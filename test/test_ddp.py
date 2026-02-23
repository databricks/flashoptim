# Copyright 2026 Databricks AI Research authors

"""DDP numerical accuracy: verify DDP training matches non-DDP at each step.

Comparison helpers return error lists (never raise mid-loop) so that all ranks
complete the same collectives before any rank exits.  Error status is
synchronized across ranks at the end.
"""

import contextlib

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from test_training import ToyDataset, _create_simple_model
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from flashoptim import enable_gradient_release

# Training settings: weight_decay and gradient accumulation combinations
SETTINGS = [
    {"name": "no_wd_no_grad_accum", "weight_decay": 0.0, "grad_accum": 1},
    {"name": "grad_accum", "weight_decay": 0.0, "grad_accum": 2},
    {"name": "weight_decay", "weight_decay": 1e-4, "grad_accum": 1},
]

_OPT_CONFIGS = [SGDM_CONFIG, LION_CONFIG, ADAMW_CONFIG]


# ---------------------------------------------------------------------------
# Helpers — return error lists instead of raising to avoid rank divergence
# ---------------------------------------------------------------------------


def _check_params_synced(model: nn.Module) -> list[str]:
    errors: list[str] = []
    for name, p in model.named_parameters():
        p_max = p.detach().clone()
        p_min = p.detach().clone()
        dist.all_reduce(p_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(p_min, op=dist.ReduceOp.MIN)
        if not torch.equal(p_max, p_min):
            abs_diff = (p_max - p_min).abs().max().item()
            rel_diff = ((p_max - p_min).abs() / (p_max.abs() + 1e-8)).max().item()
            errors.append(
                f"Parameters not synchronized across DDP ranks.\n"
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

        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()

        if has_nan or has_inf:
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            max_val = (
                tensor[torch.isfinite(tensor)].abs().max().item()
                if torch.isfinite(tensor).any()
                else float("nan")
            )
            min_val = (
                tensor[torch.isfinite(tensor)].abs().min().item()
                if torch.isfinite(tensor).any()
                else float("nan")
            )

            errors.append(
                f"[Rank {rank}] {context}: {name} contains non-finite values!\n"
                f"  NaN count: {nan_count} / {tensor.numel()}\n"
                f"  Inf count: {inf_count} / {tensor.numel()}\n"
                f"  Finite value range: [{min_val:.6e}, {max_val:.6e}]\n"
                f"  Tensor dtype: {tensor.dtype}, shape: {tensor.shape}"
            )
    return errors


def compare_gradients(
    model_nonddp: nn.Module,
    model_ddp: DDP,
    rank: int,
    step: int,
    tol: Tolerances = STRICT_TOLERANCES,
) -> list[str]:
    errors: list[str] = []

    # Verify all DDP ranks have identical gradients (all-reduce check)
    for name, p in model_ddp.module.named_parameters():
        if p.grad is not None:
            grad_max = p.grad.detach().clone()
            grad_min = p.grad.detach().clone()
            dist.all_reduce(grad_max, op=dist.ReduceOp.MAX)
            dist.all_reduce(grad_min, op=dist.ReduceOp.MIN)
            if not torch.equal(grad_max, grad_min):
                abs_diff = (grad_max - grad_min).abs().max().item()
                rel_diff = (
                    ((grad_max - grad_min).abs() / (grad_max.abs() + 1e-8)).max().item()
                )
                errors.append(
                    f"[Step {step}] Gradients not synchronized across DDP ranks.\n"
                    f"Parameter: {name}\n"
                    f"Max absolute diff: {abs_diff:.2e}\n"
                    f"Max relative diff: {rel_diff:.2e}"
                )

    # Rank 0: compare DDP gradients with non-DDP
    if rank == 0:
        for p_nonddp, p_ddp in zip(
            model_nonddp.parameters(), model_ddp.module.parameters()
        ):
            if p_nonddp.grad is not None and p_ddp.grad is not None:
                errors.extend(
                    check_tensor_similarity(
                        p_nonddp.grad,
                        p_ddp.grad,
                        f"Gradient at step {step}",
                        rank,
                        tol=tol,
                    )
                )

    return errors


def compare_weights(
    model_nonddp: nn.Module,
    model_ddp: DDP,
    rank: int,
    step: int,
    tol: Tolerances = STRICT_TOLERANCES,
) -> list[str]:
    # Verify all DDP ranks have synchronized parameters
    errors = _check_params_synced(model_ddp.module)

    # Rank 0: compare with non-DDP
    if rank == 0:
        for p_nonddp, p_ddp in zip(
            model_nonddp.parameters(), model_ddp.module.parameters()
        ):
            errors.extend(
                check_tensor_similarity(
                    p_nonddp,
                    p_ddp,
                    f"Parameter at step {step}",
                    rank,
                    tol=tol,
                )
            )

    return errors


def get_test_params():
    """OPT_CONFIGS x SETTINGS; dtype/ecc/quant batched inside mp.spawn."""
    return [{"opt_config": opt, "setting": s} for opt in _OPT_CONFIGS for s in SETTINGS]


def _get_test_id(cfg: dict) -> str:
    return f"{cfg['setting']['name']}_{cfg['opt_config'].name}"


def _run_single_ddp_config(
    rank: int,
    world_size: int,
    opt_config: OptimizerTestConfig,
    setting: dict,
    dtype: torch.dtype,
    ecc_bytes: int,
    quantized: bool,
    seed: int,
) -> None:
    """Run a single DDP accuracy test for one (dtype, ecc_bytes, quantized) config."""
    device = torch.device(f"cuda:{rank}")

    weight_decay = setting["weight_decay"]
    num_microbatches = setting["grad_accum"]

    # Build opt_kwargs from config
    opt_kwargs: dict = {
        "quantize": quantized,
        "master_weight_bits": ecc_bytes,
    }
    if dtype != torch.float32:
        opt_kwargs["dtype"] = dtype

    model_dtype = dtype
    tol = get_tolerances(model_dtype, quantized)
    opt_kwargs_clean = {k: v for k, v in opt_kwargs.items() if k != "dtype"}

    # Set seeds for reproducibility (all ranks)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Test parameters
    d_in, d_out = 10, 5
    batch_size = 8  # Per-rank batch size
    num_steps = 3  # Use fewer steps for faster testing
    lr = 0.001
    dataset = ToyDataset(n=128, d_in=d_in, d_out=d_out, seed=seed)

    # Collect all errors — never raise mid-loop to avoid rank divergence
    all_errors: list[str] = []

    # ===== MODEL CREATION (ALL RANKS) =====
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_nonddp = _create_simple_model(d_in, d_out).to(
        dtype=model_dtype, device=device
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_ddp_unwrapped = _create_simple_model(d_in, d_out).to(
        dtype=model_dtype, device=device
    )

    if rank == 0:
        for p_nonddp, p_ddp in zip(
            model_nonddp.parameters(), model_ddp_unwrapped.parameters()
        ):
            all_errors.extend(
                check_tensor_similarity(p_nonddp, p_ddp, "Initial parameters", rank)
            )

    all_errors.extend(
        _check_for_nan_inf(
            dict(model_ddp_unwrapped.named_parameters()),
            "After model initialization",
            rank,
        )
    )

    dist.barrier()
    model_ddp = DDP(model_ddp_unwrapped, device_ids=[device.index])

    dist.barrier()
    all_errors.extend(_check_params_synced(model_ddp.module))

    # ===== OPTIMIZER CREATION (ALL RANKS) =====
    opt_factory_kwargs = {
        "lr": lr,
        "weight_decay": weight_decay,
        **opt_kwargs_clean,
    }

    # Create non-DDP optimizer using factory from OptimizerTestConfig
    opt_nonddp = opt_config.factory(
        model_nonddp.parameters(),
        **opt_factory_kwargs,
    )

    opt_ddp = opt_config.factory(
        model_ddp.parameters(),
        **opt_factory_kwargs,
    )

    # ===== DATALOADER CREATION (ALL RANKS) =====
    loader_nonddp = DataLoader(
        dataset,
        batch_size=batch_size * world_size,
        shuffle=False,
        drop_last=True,
    )

    sampler_ddp = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True,
    )
    loader_ddp = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler_ddp,
    )

    loss_fn = nn.MSELoss()

    # ===== INTERLEAVED TRAINING LOOP =====
    model_nonddp.train()
    model_ddp.train()

    sampler_ddp.set_epoch(0)

    step = 0
    for (xb_nonddp, yb_nonddp), (xb_ddp, yb_ddp) in zip(loader_nonddp, loader_ddp):
        if step >= num_steps:
            break

        xb_nonddp = xb_nonddp.to(device=device, dtype=model_dtype)
        yb_nonddp = yb_nonddp.to(device=device, dtype=model_dtype)
        xb_ddp = xb_ddp.to(device=device, dtype=model_dtype)
        yb_ddp = yb_ddp.to(device=device, dtype=model_dtype)

        # ===== COMPARISON POINT 1: Weights Before Forward =====
        dist.barrier()
        all_errors.extend(compare_weights(model_nonddp, model_ddp, rank, step, tol=tol))

        # ===== Forward + Backward (Non-DDP) =====
        for mb_idx in range(num_microbatches):
            if num_microbatches > 1:
                mb_size = (batch_size * world_size) // num_microbatches
                xb_mb = xb_nonddp[mb_idx * mb_size : (mb_idx + 1) * mb_size]
                yb_mb = yb_nonddp[mb_idx * mb_size : (mb_idx + 1) * mb_size]
            else:
                xb_mb, yb_mb = xb_nonddp, yb_nonddp

            yhat = model_nonddp(xb_mb)
            loss = loss_fn(yhat, yb_mb)
            loss.backward()

        # ===== Forward + Backward (DDP) =====
        for mb_idx in range(num_microbatches):
            if num_microbatches > 1:
                mb_size = batch_size // num_microbatches
                xb_mb = xb_ddp[mb_idx * mb_size : (mb_idx + 1) * mb_size]
                yb_mb = yb_ddp[mb_idx * mb_size : (mb_idx + 1) * mb_size]
            else:
                xb_mb, yb_mb = xb_ddp, yb_ddp

            is_last_mb = mb_idx == num_microbatches - 1
            sync_ctx = contextlib.nullcontext() if is_last_mb else model_ddp.no_sync()

            with sync_ctx:
                yhat = model_ddp(xb_mb)
                loss = loss_fn(yhat, yb_mb)
                loss.backward()

        # ===== COMPARISON POINT 2: Gradients After Backward =====
        dist.barrier()

        ddp_grad_dict = {
            name: p.grad
            for name, p in model_ddp.module.named_parameters()
            if p.grad is not None
        }
        all_errors.extend(
            _check_for_nan_inf(
                ddp_grad_dict, f"After backward pass (step {step})", rank
            )
        )

        if rank == 0:
            nonddp_grad_dict = {
                name: p.grad
                for name, p in model_nonddp.named_parameters()
                if p.grad is not None
            }
            all_errors.extend(
                _check_for_nan_inf(
                    nonddp_grad_dict,
                    f"After backward pass non-DDP (step {step})",
                    rank,
                )
            )

        all_errors.extend(
            compare_gradients(model_nonddp, model_ddp, rank, step, tol=tol)
        )

        # ===== Optimizer Step =====
        opt_nonddp.step()
        opt_ddp.step()
        opt_nonddp.zero_grad(set_to_none=True)
        opt_ddp.zero_grad(set_to_none=True)

        all_errors.extend(
            _check_for_nan_inf(
                dict(model_ddp.module.named_parameters()),
                f"After optimizer step (step {step})",
                rank,
            )
        )

        step += 1

    # ===== FINAL VERIFICATION =====
    dist.barrier()
    all_errors.extend(_check_params_synced(model_ddp.module))

    # ===== SYNC ERROR STATUS =====
    # All ranks must agree on whether to raise so they exit together.
    has_errors = torch.tensor([1 if all_errors else 0], device=device)
    dist.all_reduce(has_errors, op=dist.ReduceOp.MAX)

    if has_errors.item() > 0:
        if all_errors:
            raise AssertionError(
                f"[Rank {rank}] DDP accuracy errors:\n" + "\n".join(all_errors)
            )
        else:
            raise AssertionError(
                f"[Rank {rank}] Another rank reported DDP accuracy errors"
            )

    # ===== CLEANUP =====
    # Explicitly tear down DDP wrapper and free GPU memory before next config
    # to avoid leaking reducer/hook state across sequential DDP instances.
    del model_ddp, opt_ddp, opt_nonddp
    torch.cuda.empty_cache()
    dist.barrier()


def _run_ddp_accuracy_test(
    rank: int, world_size: int, test_config: dict, seed: int
) -> None:
    opt_config: OptimizerTestConfig = test_config["opt_config"]
    setting: dict = test_config["setting"]

    for dtype, ecc_bytes, quantized in DIST_DTYPE_ECC_QUANT_CONFIGS:
        config_id = dtype_ecc_quant_id((dtype, ecc_bytes, quantized))
        if rank == 0:
            print(f"  Running {opt_config.name} {config_id} with {setting['name']}...")
        _run_single_ddp_config(
            rank,
            world_size,
            opt_config,
            setting,
            dtype,
            ecc_bytes,
            quantized,
            seed,
        )
        if rank == 0:
            print(f"  [OK] {opt_config.name} {config_id} with {setting['name']}")


@pytest.mark.parametrize("seed", [0], ids=lambda s: f"seed{s}")
@pytest.mark.parametrize(
    "test_config",
    get_test_params(),
    ids=_get_test_id,
)
def test_ddp_bitwise_accuracy(
    test_config: dict,
    seed: int,
    ddp_runner,
) -> None:
    """DDP training matches non-DDP with step-by-step allclose verification."""
    ddp_runner(_run_ddp_accuracy_test, test_config, seed)


def _run_ddp_gradient_release_rejected(rank: int, world_size: int) -> None:
    from flashoptim import FlashAdamW

    device = torch.device(f"cuda:{rank}")
    model = _create_simple_model(d_in=32, d_out=16).to(device)
    model = DDP(model, device_ids=[rank])
    opt = FlashAdamW(model.parameters(), lr=1e-3)

    with pytest.raises(TypeError, match="DistributedDataParallel"):
        enable_gradient_release(model, opt)

    dist.barrier()


def test_ddp_gradient_release_rejected(ddp_runner) -> None:
    """enable_gradient_release() must reject DDP models with a clear error."""
    ddp_runner(_run_ddp_gradient_release_rejected)
