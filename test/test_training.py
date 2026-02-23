# Copyright 2026 Databricks AI Research authors


import copy
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from test_utils import (
    ADAMW_CONFIG,
    LION_CONFIG,
    SGDM_CONFIG,
    OptimizerTestConfig,
    nmse,
)
from torch.utils.data import DataLoader, Dataset

# Test seeds for reproducibility
SEEDS = list(range(5))

# Optimizer configurations for testing
_OPT_CONFIGS = [LION_CONFIG, SGDM_CONFIG, ADAMW_CONFIG]

# AMP test thresholds (~3x worst observed, rounded to nice numbers).
# autocast modes (fp16/bf16) are exact because the fp32 optimizer sees
# fp32 master weights — only gradscaler and bf16_native introduce drift.
AMP_MIN_CORRELATION = 0.9999
AMP_MAX_NMSE_GRADSCALER = 5e-4  # worst observed ~1.3e-4
AMP_MAX_NMSE_BF16_NATIVE = 1e-4  # worst observed ~3.6e-5
AMP_MAX_NMSE_AUTOCAST = 1e-6  # worst observed ~0 (exact for fp32 optimizer)


@pytest.fixture(params=_OPT_CONFIGS, ids=[config.name for config in _OPT_CONFIGS])
def opt_config(request: pytest.FixtureRequest) -> OptimizerTestConfig:
    """Fixture that provides optimizer configurations for testing."""
    return request.param


def seed_id(seed: int) -> str:
    """Generate readable ID for seed values."""
    return f"seed{seed}"


def amp_mode_id(mode: str | None) -> str:
    """Generate readable ID for AMP modes."""
    return mode if mode else "no_amp"


class ToyDataset(Dataset):
    """Simple synthetic dataset for training tests."""

    def __init__(self, n: int, d_in: int, d_out: int, seed: int = 0) -> None:
        """
        Args:
            n: Number of samples
            d_in: Input dimension
            d_out: Output dimension
            seed: Random seed for reproducibility
        """
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, d_in, generator=g)
        # Simple linear relationship with noise
        self.y = self.x.sum(dim=1, keepdim=True).expand(-1, d_out) + 0.1 * torch.randn(
            n, d_out, generator=g
        )

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def _compute_loss_nmse(losses_baseline: list[float], losses_test: list[float]) -> float:
    assert len(losses_baseline) == len(losses_test), (
        "Loss trajectories must have same length"
    )

    baseline = np.array(losses_baseline)
    test = np.array(losses_test)

    mse = np.mean((test - baseline) ** 2)
    var_baseline = np.var(baseline)

    # Handle edge case where baseline has no variance
    if var_baseline < 1e-10:
        return mse

    return mse / var_baseline


def _compute_loss_correlation(
    losses_baseline: list[float], losses_test: list[float]
) -> float:
    assert len(losses_baseline) == len(losses_test), (
        "Loss trajectories must have same length"
    )

    baseline = np.array(losses_baseline)
    test = np.array(losses_test)

    # Use numpy's corrcoef which returns correlation matrix
    # [0, 1] is the correlation between the two arrays
    corr_matrix = np.corrcoef(baseline, test)
    return float(corr_matrix[0, 1])


def _create_simple_model(d_in: int, d_out: int, hidden_dim: int = 16) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_in, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, d_out),
    )


# (quantize, master_weight_bits) — representative checkpoint configurations.
# master_weight_bits=24 requires bf16 model (ECC is meaningless for fp32).
_CKPT_CONFIGS = [
    (False, None),  # fp32, no quantization, no ECC
    (True, None),  # fp32, quantized, no ECC
    (True, 24),  # bf16, quantized, 24-bit ECC
]


def ckpt_id(config: tuple[bool, int | None]) -> str:
    quantize, mwb = config
    q = "quant" if quantize else "noquant"
    ecc = "noECC" if mwb is None else f"ecc{mwb}b"
    return f"{q}_{ecc}"


def _prepare_batches(
    dataset: ToyDataset,
    num_steps: int,
    batch_size: int = 8,
    device: str = "cuda",
    model_dtype: torch.dtype = torch.float32,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    while len(batches) < num_steps:
        for xb, yb in loader:
            if len(batches) >= num_steps:
                break
            batches.append(
                (
                    xb.to(device=device, dtype=model_dtype),
                    yb.to(device=device, dtype=model_dtype),
                )
            )
    return batches


def _train_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    start: int,
    end: int,
) -> list[float]:
    model.train()
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    for i in range(start, end):
        xb, yb = batches[i]
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def _run_single_process_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: Dataset,
    num_steps: int,
    batch_size: int = 8,
    amp_mode: str | None = None,
    device: str = "cuda",
) -> list[float]:
    model = model.to(device)
    model.train()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    loss_fn = nn.MSELoss()

    # Setup AMP
    use_autocast = amp_mode in ("fp16_autocast", "bf16_autocast", "fp16_gradscaler")
    autocast_dtype = (
        torch.float16
        if amp_mode in ("fp16_autocast", "fp16_gradscaler")
        else torch.bfloat16
    )
    use_gradscaler = amp_mode == "fp16_gradscaler"

    scaler = torch.amp.GradScaler("cuda") if use_gradscaler else None

    losses = []
    step = 0
    while step < num_steps:
        for xb, yb in loader:
            if step >= num_steps:
                break

            xb = xb.to(device)
            yb = yb.to(device)

            # For bf16_native, cast inputs to bf16
            if amp_mode == "bf16_native":
                xb = xb.to(dtype=torch.bfloat16)
                yb = yb.to(dtype=torch.bfloat16)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass with optional autocast
            if use_autocast:
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    yhat = model(xb)
                    loss = loss_fn(yhat, yb)
            else:
                yhat = model(xb)
                loss = loss_fn(yhat, yb)

            # Backward pass with optional gradient scaling
            if use_gradscaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            step += 1

    return losses


# ============================================================================
# AMP Tests
# ============================================================================


@pytest.mark.parametrize("seed", SEEDS, ids=seed_id)
@pytest.mark.parametrize(
    "amp_mode",
    ["fp16_autocast", "bf16_autocast", "fp16_gradscaler", "bf16_native"],
    ids=amp_mode_id,
)
def test_amp_training(
    opt_config: OptimizerTestConfig,
    seed: int,
    amp_mode: str,
) -> None:
    """Test AMP training without DDP."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = "cuda"
    d_in, d_out = 10, 5
    dataset = ToyDataset(n=128, d_in=d_in, d_out=d_out, seed=seed)
    num_steps = 20

    # Use smaller learning rate for more stable numerical behavior
    lr = 0.001

    # Create shared initial weights so both runs start identically
    init_model = _create_simple_model(d_in, d_out).to(device)
    init_state = copy.deepcopy(init_model.state_dict())
    del init_model

    # Run baseline (fp32, no AMP)
    model_baseline = _create_simple_model(d_in, d_out).to(device)
    model_baseline.load_state_dict(init_state)
    optimizer_baseline = opt_config.factory(
        model_baseline.parameters(),
        lr=lr,
        quantize=False,
        check_numerics=False,
    )
    losses_baseline = _run_single_process_training(
        model=model_baseline,
        optimizer=optimizer_baseline,
        dataset=dataset,
        num_steps=num_steps,
        batch_size=8,
        amp_mode=None,
        device=device,
    )

    # Run with AMP (same initial weights)
    model_test = _create_simple_model(d_in, d_out).to(device)
    model_test.load_state_dict(init_state)
    if amp_mode == "bf16_native":
        model_test = model_test.to(dtype=torch.bfloat16)

    optimizer_test = opt_config.factory(
        model_test.parameters(),
        lr=lr,
        quantize=False,
        check_numerics=False,
    )
    losses_test = _run_single_process_training(
        model=model_test,
        optimizer=optimizer_test,
        dataset=dataset,
        num_steps=num_steps,
        batch_size=8,
        amp_mode=amp_mode,
        device=device,
    )

    # Verify loss decreases
    assert losses_test[-1] < losses_test[0], "Loss should decrease during training"

    # Compare loss trajectory with baseline
    nmse = _compute_loss_nmse(losses_baseline, losses_test)
    corr = _compute_loss_correlation(losses_baseline, losses_test)
    print(
        f"\nAMP ({opt_config.name}, {amp_mode}): NMSE={nmse:.6f}, Correlation={corr:.6f}"
    )

    # Correlation should be very high - loss trajectories should follow same pattern
    # With smaller LR (0.001), numerical stability is excellent
    assert corr >= AMP_MIN_CORRELATION, (
        f"Loss trajectory correlation too low: {corr:.6f} (min: {AMP_MIN_CORRELATION})"
    )

    if amp_mode == "fp16_gradscaler":
        threshold = AMP_MAX_NMSE_GRADSCALER
    elif amp_mode == "bf16_native":
        threshold = AMP_MAX_NMSE_BF16_NATIVE
    else:  # fp16_autocast, bf16_autocast
        threshold = AMP_MAX_NMSE_AUTOCAST

    assert nmse < threshold, (
        f"Loss trajectory NMSE too high: {nmse:.6f} (threshold: {threshold})"
    )


# ============================================================================
# Checkpoint Tests
# ============================================================================


@pytest.mark.parametrize("seed", SEEDS[:2], ids=seed_id)
@pytest.mark.parametrize("ckpt_config", _CKPT_CONFIGS, ids=ckpt_id)
def test_joint_model_optimizer_checkpoint(
    opt_config: OptimizerTestConfig,
    seed: int,
    ckpt_config: tuple[bool, int],
) -> None:
    """Save model + optimizer to disk, load into fresh instances, verify
    training continues identically."""
    quantize, master_weight_bits = ckpt_config
    N, M = 5, 5
    total = N + M
    d_in, d_out = 10, 5

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model_dtype = torch.bfloat16 if master_weight_bits in (24, 32) else torch.float32
    dataset = ToyDataset(n=128, d_in=d_in, d_out=d_out, seed=seed)
    batches = _prepare_batches(dataset, total, model_dtype=model_dtype)

    model = _create_simple_model(d_in, d_out).to("cuda", dtype=model_dtype)
    opt = opt_config.factory(
        model.parameters(),
        lr=0.01,
        quantize=quantize,
        master_weight_bits=master_weight_bits,
        check_numerics=False,
    )
    _train_steps(model, opt, batches, 0, N)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save both to disk
        torch.save(model.state_dict(), f"{tmpdir}/model.pt")
        torch.save(opt.state_dict(), f"{tmpdir}/opt.pt")

        # Continue original
        losses_orig = _train_steps(model, opt, batches, N, total)

        # Load into fresh instances
        model_fresh = _create_simple_model(d_in, d_out).to("cuda", dtype=model_dtype)
        model_fresh.load_state_dict(torch.load(f"{tmpdir}/model.pt", weights_only=True))
        opt_fresh = opt_config.factory(
            model_fresh.parameters(),
            lr=0.01,
            quantize=quantize,
            master_weight_bits=master_weight_bits,
            check_numerics=False,
        )
        opt_fresh.load_state_dict(torch.load(f"{tmpdir}/opt.pt", weights_only=False))

    losses_fresh = _train_steps(model_fresh, opt_fresh, batches, N, total)

    loss_tol = 5e-3 if quantize else 1e-3
    min_cossim = 0.99 if quantize else 0.9999
    max_nmse = 1e-3 if quantize else 1e-5

    for i, (lo, lf) in enumerate(zip(losses_orig, losses_fresh)):
        rel = abs(lo - lf) / (abs(lo) + 1e-12)
        assert rel < loss_tol, (
            f"Step {N + i}: orig={lo:.6f} fresh={lf:.6f} reldiff={rel:.6f}"
        )
    for (n, po), (_, pf) in zip(
        model.named_parameters(), model_fresh.named_parameters()
    ):
        cs = torch.cosine_similarity(po.ravel(), pf.ravel(), dim=0).item()
        assert cs > min_cossim, f"{n}: cossim={cs:.6f}"
        assert nmse(po.ravel(), pf.ravel()) < max_nmse, f"{n}: nmse too high"
