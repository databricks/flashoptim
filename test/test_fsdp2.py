# Copyright 2026 Databricks AI Research authors

"""
Step-by-step FSDP2 numerical accuracy testing.

Compares FSDP2 training against a non-distributed baseline at each step,
using unshard()/full_tensor() to materialize full parameters for comparison.

Each pytest test case = one optimizer. The inner function loops over all
settings, reshard options, and dtype/ecc/quant configs within a single
mp.spawn call, amortizing process creation + NCCL init/destroy overhead.
"""

from typing import Optional

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from test_training import _create_simple_model
from test_utils import (
    ADAMW_CONFIG,
    DIST_DTYPE_ECC_QUANT_CONFIGS,
    LION_CONFIG,
    SGDM_CONFIG,
    STRICT_TOLERANCES,
    OptimizerTestConfig,
    Tolerances,
    check_tensor_similarity,
    dtype_ecc_quant_id,
    get_tolerances,
)
from torch.distributed.fsdp import FSDPModule, fully_shard

from flashoptim import enable_gradient_release


class DeterministicShardedDataset:
    """Dataset supporting both global (baseline) and per-rank (FSDP2) access.

    For mathematical equivalence between FSDP2 and single-GPU training:
    - FSDP2: Each rank processes B samples, gradients all-reduced (averaged)
    - Single-GPU: Processes B * N samples total

    With MSELoss(reduction='mean'), mean_of_means == global_mean, so:
    - FSDP2 gradient = (1/N) * sum(grad_rank_i) where each grad_rank_i is from B samples
    - Single-GPU gradient = gradient from N*B samples
    These are mathematically equivalent.
    """

    def __init__(
        self,
        n_samples: int,
        d_in: int,
        d_out: int,
        batch_size: int,
        world_size: int,
        seed: int = 0,
    ):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n_samples, d_in, generator=g)
        self.y = self.x.sum(dim=1, keepdim=True).expand(-1, d_out) + 0.1 * torch.randn(
            n_samples, d_out, generator=g
        )
        self.batch_size = batch_size
        self.world_size = world_size
        self.n_samples = n_samples

    def get_global_batch(
        self, step: int
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get concatenated batch for non-distributed baseline (B * world_size samples)."""
        start = step * self.batch_size * self.world_size
        end = start + self.batch_size * self.world_size
        if end > self.n_samples:
            return None, None  # No more data
        return self.x[start:end], self.y[start:end]

    def get_rank_batch(
        self, step: int, rank: int
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get batch for specific rank (B samples)."""
        global_start = step * self.batch_size * self.world_size
        rank_start = global_start + rank * self.batch_size
        rank_end = rank_start + self.batch_size
        if rank_end > self.n_samples:
            return None, None
        return self.x[rank_start:rank_end], self.y[rank_start:rank_end]


# ============================================================================
# FSDP2-Specific Configuration
# ============================================================================

# reshard_after_forward options to test
# True = reshard after forward (FULL_SHARD equivalent, saves memory)
# False = keep unsharded after forward (SHARD_GRAD_OP equivalent)
RESHARD_AFTER_FORWARD_OPTIONS = [True, False]

# ============================================================================
# Test Configuration
# ============================================================================

# Training settings: weight_decay and gradient accumulation combinations
SETTINGS = [
    {"name": "no_wd_no_grad_accum", "weight_decay": 0.0, "grad_accum": 1},
    {"name": "grad_accum", "weight_decay": 0.0, "grad_accum": 2},
    {"name": "weight_decay", "weight_decay": 1e-4, "grad_accum": 1},
    {
        "name": "grad_release_no_grad_accum",
        "weight_decay": 0.0,
        "grad_accum": 1,
        "grad_release": True,
    },
]

# Optimizer configs from test_utils.py
_OPT_CONFIGS = [SGDM_CONFIG, LION_CONFIG, ADAMW_CONFIG]


# ============================================================================
# Helper Functions
# ============================================================================


def get_full_tensor(param: torch.Tensor) -> torch.Tensor:
    if hasattr(param, "full_tensor"):
        return param.full_tensor()
    return param


def _check_params_synced_fsdp2(model: nn.Module) -> list[str]:
    """Completes ALL all_reduce ops before returning to prevent rank divergence."""
    errors: list[str] = []
    for name, p in model.named_parameters():
        p_full = get_full_tensor(p)
        p_max = p_full.detach().clone()
        p_min = p_full.detach().clone()
        dist.all_reduce(p_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(p_min, op=dist.ReduceOp.MIN)
        if not torch.equal(p_max, p_min):
            abs_diff = (p_max - p_min).abs().max().item()
            rel_diff = ((p_max - p_min).abs() / (p_max.abs() + 1e-8)).max().item()
            errors.append(
                f"Parameters not synchronized across FSDP2 ranks.\n"
                f"Parameter: {name}\n"
                f"Max absolute diff: {abs_diff:.2e}\n"
                f"Max relative diff: {rel_diff:.2e}"
            )
    return errors


def _check_for_nan_inf(
    tensors: dict[str, torch.Tensor],
    context: str,
    rank: int,
) -> list[str]:
    errors: list[str] = []
    for name, tensor in tensors.items():
        if tensor is None:
            continue

        # Handle DTensor
        t = get_full_tensor(tensor)

        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()

        if has_nan or has_inf:
            nan_count = torch.isnan(t).sum().item()
            inf_count = torch.isinf(t).sum().item()
            max_val = (
                t[torch.isfinite(t)].abs().max().item()
                if torch.isfinite(t).any()
                else float("nan")
            )
            min_val = (
                t[torch.isfinite(t)].abs().min().item()
                if torch.isfinite(t).any()
                else float("nan")
            )

            errors.append(
                f"[Rank {rank}] {context}: {name} contains non-finite values!\n"
                f"  NaN count: {nan_count} / {t.numel()}\n"
                f"  Inf count: {inf_count} / {t.numel()}\n"
                f"  Finite value range: [{min_val:.6e}, {max_val:.6e}]\n"
                f"  Tensor dtype: {t.dtype}, shape: {t.shape}\n"
                f"  This often happens with float16 when master_weight_bits=None\n"
                f"  Consider using master_weight_bits=24 or 32 for float16 parameters"
            )
    return errors


def unshard_for_comparison(model: nn.Module):
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.unshard()


def reshard_after_comparison(model: nn.Module):
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def compare_weights_fsdp2(
    model_baseline: Optional[nn.Module],  # None on ranks > 0
    model_fsdp2: nn.Module,
    rank: int,
    step: int,
    tol: Tolerances = STRICT_TOLERANCES,
) -> list[str]:
    """Completes ALL collective ops on all ranks before returning."""
    # Unshard to access full parameters
    unshard_for_comparison(model_fsdp2)

    errors = []
    try:
        # Verify all FSDP2 ranks have synchronized parameters
        errors.extend(_check_params_synced_fsdp2(model_fsdp2))

        # All ranks must participate in get_full_tensor (collective op)
        # Only rank 0 performs the comparison (baseline only exists there)
        baseline_params = (
            dict(model_baseline.named_parameters())
            if rank == 0 and model_baseline is not None
            else {}
        )
        fsdp2_params = dict(model_fsdp2.named_parameters())

        for name, p_fsdp2 in fsdp2_params.items():
            # All ranks call get_full_tensor (collective)
            p_fsdp2_full = get_full_tensor(p_fsdp2)

            # Only rank 0 compares with baseline
            if rank == 0 and name in baseline_params:
                p_baseline = baseline_params[name]
                errors.extend(
                    check_tensor_similarity(
                        p_baseline,
                        p_fsdp2_full,
                        f"Parameter '{name}' at step {step}",
                        rank,
                        tol=tol,
                    )
                )

            # Sync all ranks after each parameter to prevent race conditions
            dist.barrier()
    finally:
        reshard_after_comparison(model_fsdp2)

    return errors


def get_test_params():
    """One entry per optimizer; settings/reshard batched inside mp.spawn."""
    return [{"opt_config": opt} for opt in _OPT_CONFIGS]


def _get_test_id(cfg: dict) -> str:
    return cfg["opt_config"].name


# ============================================================================
# Main Test Implementation
# ============================================================================


def _run_single_fsdp2_config(
    rank: int,
    world_size: int,
    opt_config: OptimizerTestConfig,
    setting: dict,
    reshard_after_forward: bool,
    dtype: torch.dtype,
    ecc_bytes: int,
    quantized: bool,
    seed: int,
) -> None:
    """Errors are collected (never raised mid-loop) to prevent rank divergence."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    weight_decay = setting["weight_decay"]
    num_microbatches = setting["grad_accum"]

    # Build opt_kwargs from config
    opt_kwargs: dict = {
        "quantize": quantized,
        "master_weight_bits": ecc_bytes,
    }
    if dtype != torch.float32:
        opt_kwargs["dtype"] = dtype

    # Test parameters
    d_in, d_out = 10, 5
    batch_size = 8  # Per-rank batch size
    num_steps = 3
    lr = 0.001

    model_dtype = dtype
    tol = get_tolerances(model_dtype, quantized)
    opt_kwargs_clean = {k: v for k, v in opt_kwargs.items() if k != "dtype"}

    # ===== DATASET CREATION =====
    dataset = DeterministicShardedDataset(
        n_samples=128,
        d_in=d_in,
        d_out=d_out,
        batch_size=batch_size,
        world_size=world_size,
        seed=seed,
    )

    # Collect all errors — never raise mid-loop to avoid rank divergence
    all_errors: list[str] = []

    # ===== MODEL CREATION =====
    # FSDP2 model on ALL ranks
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_fsdp2 = _create_simple_model(d_in, d_out).to(device=device, dtype=model_dtype)

    # Check for NaN/Inf before wrapping
    param_dict = {name: p.data for name, p in model_fsdp2.named_parameters()}
    all_errors.extend(
        _check_for_nan_inf(param_dict, "After model initialization", rank)
    )

    # CRITICAL: Barrier before wrapping
    dist.barrier()

    # Apply FSDP2 sharding
    fully_shard(model_fsdp2, reshard_after_forward=reshard_after_forward)

    # Verify model is now an FSDPModule
    assert isinstance(model_fsdp2, FSDPModule), (
        "Model should be FSDPModule after fully_shard"
    )

    # Single-GPU baseline ONLY on rank 0 (NO DDP wrapper)
    if rank == 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model_baseline = _create_simple_model(d_in, d_out).to(
            device=device, dtype=model_dtype
        )
    else:
        model_baseline = None

    # Verify all FSDP2 ranks have identical initialization
    dist.barrier()
    unshard_for_comparison(model_fsdp2)
    try:
        all_errors.extend(_check_params_synced_fsdp2(model_fsdp2))

        # Verify baseline and FSDP2 have identical initial weights
        if rank == 0:
            assert model_baseline is not None
            for (name_b, p_b), (name_f, p_f) in zip(
                model_baseline.named_parameters(), model_fsdp2.named_parameters()
            ):
                assert name_b == name_f, (
                    f"Parameter name mismatch: baseline='{name_b}', fsdp2='{name_f}'"
                )
                p_f_full = get_full_tensor(p_f)
                all_errors.extend(
                    check_tensor_similarity(
                        p_b, p_f_full, f"Initial weight '{name_b}'", rank
                    )
                )
    finally:
        reshard_after_comparison(model_fsdp2)

    # ===== OPTIMIZER CREATION =====
    opt_factory_kwargs = {
        "lr": lr,
        "weight_decay": weight_decay,
        **opt_kwargs_clean,
    }

    # Create optimizer for FSDP2 model (all ranks)
    opt_fsdp2 = opt_config.factory(
        model_fsdp2.parameters(),
        **opt_factory_kwargs,
    )

    # Create optimizer for baseline model (rank 0 only)
    if rank == 0:
        assert model_baseline is not None
        opt_baseline = opt_config.factory(
            model_baseline.parameters(),
            **opt_factory_kwargs,
        )
    else:
        opt_baseline = None

    is_grad_release = setting.get("grad_release", False)
    if is_grad_release:
        enable_gradient_release(model_fsdp2, opt_fsdp2)
        if rank == 0:
            assert model_baseline is not None
            enable_gradient_release(model_baseline, opt_baseline)

    loss_fn = nn.MSELoss()  # reduction='mean' is default - critical for equivalence

    # ===== INTERLEAVED TRAINING LOOP =====
    model_fsdp2.train()
    if rank == 0:
        assert model_baseline is not None
        model_baseline.train()

    for step in range(num_steps):
        # Get per-rank batch for FSDP2
        xb_rank_raw, yb_rank_raw = dataset.get_rank_batch(step, rank)
        if xb_rank_raw is None or yb_rank_raw is None:
            break
        xb_rank = xb_rank_raw.to(device=device, dtype=model_dtype)
        yb_rank = yb_rank_raw.to(device=device, dtype=model_dtype)

        # Get global batch for baseline (rank 0 only)
        xb_global: Optional[torch.Tensor] = None
        yb_global: Optional[torch.Tensor] = None
        if rank == 0:
            xb_global_raw, yb_global_raw = dataset.get_global_batch(step)
            assert xb_global_raw is not None and yb_global_raw is not None
            xb_global = xb_global_raw.to(device=device, dtype=model_dtype)
            yb_global = yb_global_raw.to(device=device, dtype=model_dtype)

        # Sync before forward
        dist.barrier()

        # ===== Forward + Backward (FSDP2) =====
        for mb_idx in range(num_microbatches):
            if num_microbatches > 1:
                mb_size = batch_size // num_microbatches
                xb_mb = xb_rank[mb_idx * mb_size : (mb_idx + 1) * mb_size]
                yb_mb = yb_rank[mb_idx * mb_size : (mb_idx + 1) * mb_size]
            else:
                xb_mb, yb_mb = xb_rank, yb_rank

            # For gradient accumulation in FSDP2, use set_requires_gradient_sync
            is_last_microbatch = mb_idx == num_microbatches - 1
            if num_microbatches > 1:
                # Disable gradient sync for all but the last microbatch
                model_fsdp2.set_requires_gradient_sync(is_last_microbatch)

            loss_fsdp2 = loss_fn(model_fsdp2(xb_mb), yb_mb)
            loss_fsdp2.backward()

        # Re-enable gradient sync if we disabled it
        if num_microbatches > 1:
            model_fsdp2.set_requires_gradient_sync(True)

        # Capture FSDP2 gradients IMMEDIATELY after backward (before any unshard/reshard)
        # FSDP2 stores gradients on sharded params; they get cleared after unshard
        if not is_grad_release:
            fsdp2_grads: dict[str, torch.Tensor] = {}
            for name, p in model_fsdp2.named_parameters():
                if p.grad is not None:
                    # Get the full gradient from the sharded gradient tensor
                    fsdp2_grads[name] = get_full_tensor(p.grad).clone()

        # ===== Forward + Backward (Baseline - rank 0 only, processes GLOBAL batch) =====
        if rank == 0:
            assert model_baseline is not None
            assert xb_global is not None and yb_global is not None
            global_batch_size = batch_size * world_size
            for mb_idx in range(num_microbatches):
                if num_microbatches > 1:
                    mb_size = global_batch_size // num_microbatches
                    xb_mb = xb_global[mb_idx * mb_size : (mb_idx + 1) * mb_size]
                    yb_mb = yb_global[mb_idx * mb_size : (mb_idx + 1) * mb_size]
                else:
                    xb_mb, yb_mb = xb_global, yb_global

                loss_baseline = loss_fn(model_baseline(xb_mb), yb_mb)
                loss_baseline.backward()

        # Barrier to sync FSDP2 gradient all-reduce completion
        dist.barrier()

        # ===== COMPARISON POINT: Gradients After Backward =====
        if not is_grad_release:
            # Check FSDP2 gradients for NaN/Inf (using captured gradients)
            all_errors.extend(
                _check_for_nan_inf(
                    fsdp2_grads, f"After backward pass (step {step})", rank
                )
            )

            # Check baseline gradients for NaN/Inf (rank 0 only)
            if rank == 0:
                assert model_baseline is not None
                baseline_grad_dict = {
                    name: p.grad
                    for name, p in model_baseline.named_parameters()
                    if p.grad is not None
                }
                all_errors.extend(
                    _check_for_nan_inf(
                        baseline_grad_dict,
                        f"After backward pass baseline (step {step})",
                        rank,
                    )
                )

            # Compare gradients using captured FSDP2 gradients
            if rank == 0:
                assert model_baseline is not None
                for name, p_baseline in model_baseline.named_parameters():
                    grad_baseline = p_baseline.grad
                    grad_fsdp2 = fsdp2_grads.get(name)
                    if grad_baseline is not None and grad_fsdp2 is not None:
                        all_errors.extend(
                            check_tensor_similarity(
                                grad_baseline,
                                grad_fsdp2,
                                f"Gradient '{name}' at step {step}",
                                rank,
                                tol=tol,
                            )
                        )

        # ===== Optimizer Step =====
        opt_fsdp2.step()
        opt_fsdp2.zero_grad(set_to_none=True)
        if rank == 0:
            assert opt_baseline is not None
            opt_baseline.step()
            opt_baseline.zero_grad(set_to_none=True)

        # Check FSDP2 parameters after optimizer step
        unshard_for_comparison(model_fsdp2)
        param_dict_after_step = {
            name: p.data for name, p in model_fsdp2.named_parameters()
        }
        all_errors.extend(
            _check_for_nan_inf(
                param_dict_after_step, f"After optimizer step (step {step})", rank
            )
        )
        reshard_after_comparison(model_fsdp2)

        # Compare weights after optimizer step
        dist.barrier()
        weight_errors = compare_weights_fsdp2(
            model_baseline, model_fsdp2, rank, step, tol=tol
        )
        all_errors.extend(weight_errors)

    # ===== FINAL VERIFICATION =====
    dist.barrier()
    unshard_for_comparison(model_fsdp2)
    try:
        all_errors.extend(_check_params_synced_fsdp2(model_fsdp2))
    finally:
        reshard_after_comparison(model_fsdp2)

    # Synchronize error status across ranks - if any rank has errors, all should fail
    has_errors = torch.tensor([1 if all_errors else 0], device=device)
    dist.all_reduce(has_errors, op=dist.ReduceOp.MAX)

    # All ranks should raise if any rank had errors
    if has_errors.item() > 0:
        if all_errors:
            raise AssertionError(
                f"[Rank {rank}] Numerical mismatches detected:\n"
                + "\n".join(all_errors)
            )
        else:
            # This rank had no errors but another rank did
            raise AssertionError(
                f"[Rank {rank}] Another rank reported numerical mismatches"
            )

    # ===== CLEANUP =====
    # Free FSDP2 model and optimizer state before next config to avoid
    # leaking sharding state across sequential FSDP2 instances.
    del model_fsdp2, opt_fsdp2
    if rank == 0:
        del model_baseline, opt_baseline
    torch.cuda.empty_cache()
    dist.barrier()


def _run_fsdp2_accuracy_test(
    rank: int, world_size: int, test_config: dict, seed: int
) -> None:
    opt_config: OptimizerTestConfig = test_config["opt_config"]

    for setting in SETTINGS:
        # Gradient release doesn't support grad accumulation — release hooks fire
        # per-parameter before gradients are synchronized across microbatches.
        if setting.get("grad_release", False) and setting["grad_accum"] > 1:
            continue

        for reshard_after_forward in RESHARD_AFTER_FORWARD_OPTIONS:
            reshard_str = "reshard" if reshard_after_forward else "no_reshard"
            for dtype, ecc_bytes, quantized in DIST_DTYPE_ECC_QUANT_CONFIGS:
                config_id = dtype_ecc_quant_id((dtype, ecc_bytes, quantized))
                if rank == 0:
                    print(
                        f"  Running {opt_config.name} {config_id} "
                        f"with {setting['name']}, {reshard_str}..."
                    )
                _run_single_fsdp2_config(
                    rank,
                    world_size,
                    opt_config,
                    setting,
                    reshard_after_forward,
                    dtype,
                    ecc_bytes,
                    quantized,
                    seed,
                )
                if rank == 0:
                    print(
                        f"  [OK] {opt_config.name} {config_id} "
                        f"with {setting['name']}, {reshard_str}"
                    )


@pytest.mark.parametrize("seed", [0], ids=lambda s: f"seed{s}")
@pytest.mark.parametrize(
    "test_config",
    get_test_params(),
    ids=_get_test_id,
)
def test_fsdp2_numerical_accuracy(
    test_config: dict,
    seed: int,
    fsdp2_runner,
) -> None:
    fsdp2_runner(_run_fsdp2_accuracy_test, test_config, seed)


# ============================================================================
# FSDP2 fp32 state dict roundtrip test
# ============================================================================


def _run_fsdp2_fp32_state_dict_roundtrip(
    rank: int,
    world_size: int,
    opt_config: OptimizerTestConfig,
    seed: int,
) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    d_in, d_out = 10, 5
    dtype = torch.bfloat16

    # Create FSDP2 model
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = _create_simple_model(d_in, d_out).to(device=device, dtype=dtype)
    dist.barrier()
    fully_shard(model)

    # Create optimizer with ECC enabled
    opt = opt_config.factory(model.parameters(), lr=0.01, master_weight_bits=24)

    # Train a few steps to populate ECC state
    loss_fn = nn.MSELoss()
    for _ in range(3):
        x = torch.randn(8, d_in, device=device, dtype=dtype)
        y = torch.randn(8, d_out, device=device, dtype=dtype)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad()

    # get_fp32_model_state_dict: exercises reconstruct_fp32_param with DTensors
    fp32_state = opt.get_fp32_model_state_dict(model)
    for v in fp32_state.values():
        assert v.dtype == torch.float32

    # Perturb weights
    unshard_for_comparison(model)
    for p in model.parameters():
        p.data.fill_(0.0)
    reshard_after_comparison(model)

    # set_fp32_model_state_dict: exercises compute_ecc_bits with DTensors
    opt.set_fp32_model_state_dict(model, fp32_state)

    # Roundtrip: restored fp32 state should match original
    fp32_state_after = opt.get_fp32_model_state_dict(model)
    for name in fp32_state:
        torch.testing.assert_close(fp32_state[name], fp32_state_after[name])


@pytest.mark.parametrize("seed", [0], ids=lambda s: f"seed{s}")
@pytest.mark.parametrize("opt_config", _OPT_CONFIGS, ids=lambda c: c.name)
def test_fsdp2_fp32_state_dict_roundtrip(
    opt_config: OptimizerTestConfig,
    seed: int,
    fsdp2_runner,
) -> None:
    fsdp2_runner(_run_fsdp2_fp32_state_dict_roundtrip, opt_config, seed)


# ============================================================================
# FSDP2 DCP (distributed checkpoint) state dict roundtrip test
# ============================================================================

_DCP_CONFIGS = [
    (False, None),  # uncompressed, no ECC
    (False, 24),  # uncompressed, ECC 24-bit
    (True, None),  # compressed, no ECC
    (True, 24),  # compressed, ECC 24-bit
    (True, 32),  # compressed, ECC 32-bit
]


def _dcp_config_id(cfg: tuple[bool, int | None]) -> str:
    compress, ecc = cfg
    prefix = "compressed" if compress else "uncompressed"
    suffix = f"ecc{ecc}" if ecc else "no_ecc"
    return f"{prefix}_{suffix}"


def _run_fsdp2_dcp_state_dict_roundtrip(
    rank: int,
    world_size: int,
    opt_config: OptimizerTestConfig,
    compress: bool,
    master_weight_bits: int | None,
    ckpt_dir: str,
    seed: int,
) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    d_in, d_out = 32, 16
    dtype = torch.bfloat16
    lr = 0.001

    opt_kwargs: dict = {
        "lr": lr,
        "compress_state_dict": compress,
        "master_weight_bits": master_weight_bits,
    }

    # --- Create model + optimizer, train 3 steps ---
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = _create_simple_model(d_in, d_out).to(device=device, dtype=dtype)
    dist.barrier()
    fully_shard(model)

    opt = opt_config.factory(model.parameters(), **opt_kwargs)
    loss_fn = nn.MSELoss()

    g = torch.Generator(device=device).manual_seed(seed + rank)
    for _ in range(3):
        x = torch.randn(8, d_in, device=device, dtype=dtype, generator=g)
        y = torch.randn(8, d_out, device=device, dtype=dtype, generator=g)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    # --- Save optimizer state via DCP ---
    saved_osd = get_optimizer_state_dict(model, opt)
    dcp.save({"optimizer": saved_osd}, checkpoint_id=ckpt_dir)
    dist.barrier()

    # --- Load into a fresh template and verify roundtrip ---
    loaded_osd = get_optimizer_state_dict(model, opt)
    dcp.load({"optimizer": loaded_osd}, checkpoint_id=ckpt_dir)

    # Verify every state tensor survived the DCP roundtrip exactly
    for fqn in saved_osd["state"]:
        saved_state = saved_osd["state"][fqn]
        loaded_state = loaded_osd["state"][fqn]
        for key in saved_state:
            v_saved = saved_state[key]
            v_loaded = loaded_state[key]
            if isinstance(v_saved, torch.Tensor) and v_saved.dim() > 0:
                vs = v_saved.to_local() if hasattr(v_saved, "to_local") else v_saved
                vl = v_loaded.to_local() if hasattr(v_loaded, "to_local") else v_loaded
                assert torch.equal(vs, vl), (
                    f"[Rank {rank}] state[{fqn}].{key} changed after "
                    f"DCP roundtrip: max diff = "
                    f"{(vs.float() - vl.float()).abs().max().item()}"
                )

    # Apply loaded state and verify optimizer can still step
    set_optimizer_state_dict(model, opt, loaded_osd)

    x = torch.randn(8, d_in, device=device, dtype=dtype)
    y = torch.randn(8, d_out, device=device, dtype=dtype)
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    # Verify parameters are still finite after step
    unshard_for_comparison(model)
    try:
        for name, p in model.named_parameters():
            p_full = get_full_tensor(p)
            assert torch.isfinite(p_full).all(), (
                f"[Rank {rank}] Parameter '{name}' has non-finite values "
                f"after DCP load + step"
            )
    finally:
        reshard_after_comparison(model)

    del model, opt
    torch.cuda.empty_cache()
    dist.barrier()


@pytest.mark.parametrize("seed", [0], ids=lambda s: f"seed{s}")
@pytest.mark.parametrize("dcp_config", _DCP_CONFIGS, ids=_dcp_config_id)
@pytest.mark.parametrize("opt_config", _OPT_CONFIGS, ids=lambda c: c.name)
def test_fsdp2_dcp_state_dict_roundtrip(
    opt_config: OptimizerTestConfig,
    dcp_config: tuple[bool, int | None],
    seed: int,
    fsdp2_runner,
    tmp_path,
) -> None:
    compress, master_weight_bits = dcp_config
    fsdp2_runner(
        _run_fsdp2_dcp_state_dict_roundtrip,
        opt_config,
        compress,
        master_weight_bits,
        str(tmp_path / "ckpt"),
        seed,
    )
