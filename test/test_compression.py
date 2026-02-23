# Copyright 2026 Databricks AI Research authors

import pytest
import torch
from test_utils import _FLOAT_DTYPES, dtype_id, shape_id

from flashoptim.optimizers import (
    _MaybeQuantizedTensor,
    dequantize,
    quantize,
)

SEEDS = list(range(10))

# Shapes chosen to exercise group-size alignment edge cases for compression
_COMPRESSION_PARAM_SHAPES = [
    (1, 5),
    (1, 12),
    (1, 16),
    (1, 17),
    (1, 32),
    (17, 32),
    (1024, 16),
]


def _seed_id(seed: int) -> str:
    return f"seed{seed}"


def _make_test_tensor(
    test_case: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    signed: bool,
    gen: torch.Generator,
) -> torch.Tensor:
    device = gen.device
    if test_case == "zeros":
        x = torch.zeros(shape, device=device, dtype=dtype)
    elif test_case == "max_vals":
        max_val = 127.0 if signed else 255.0
        x = torch.full(shape, max_val, device=device, dtype=dtype)
    elif test_case == "small_vals":
        x = torch.rand(shape, device=device, dtype=dtype, generator=gen) * 0.01
    elif test_case == "mixed_scales":
        x = torch.randn(shape, device=device, dtype=dtype, generator=gen)
        split_point = max(1, shape[0] // 2)
        x[:split_point] *= 0.1
        x[split_point:] *= 2.0
    else:  # random
        x = torch.randn(shape, device=device, dtype=dtype, generator=gen)

    if not signed:
        x = x.abs()
    return x


def _check_roundtrip_quality(
    x: torch.Tensor,
    x_reconstructed: torch.Tensor,
    test_case: str,
) -> None:
    x_flat = x.ravel()
    x_recon_flat = x_reconstructed.to(x.dtype).ravel()

    if test_case == "zeros":
        assert torch.all(x_recon_flat == 0), "Zero tensor should remain zero"
        return

    cos_sim = torch.cosine_similarity(
        x_flat.float(), x_recon_flat.float(), dim=0
    ).item()
    assert cos_sim > 0.99, f"Cosine similarity {cos_sim} too low"

    mae_val = (x_recon_flat - x_flat).abs().mean().item()
    absmax = x_flat.abs().max().item()
    # Expected max MAE: half a quantization bin width with 1.6x tolerance
    max_mae = (2 * absmax) / 255.0 / 2 * 1.6
    assert mae_val <= max_mae, f"MAE {mae_val} too high (max {max_mae})"


# (test_case, seed) pairs — deterministic cases only need seed=0
_ROUNDTRIP_CASES = [
    ("zeros", 0),
    ("max_vals", 0),
    *[("random", s) for s in SEEDS],
    *[("small_vals", s) for s in SEEDS],
    *[("mixed_scales", s) for s in SEEDS],
]


def _case_seed_id(case_seed: tuple[str, int]) -> str:
    return f"{case_seed[0]}_seed{case_seed[1]}"


@pytest.mark.parametrize(
    "test_case,seed",
    _ROUNDTRIP_CASES,
    ids=[_case_seed_id(c) for c in _ROUNDTRIP_CASES],
)
@pytest.mark.parametrize(
    "N,D",
    _COMPRESSION_PARAM_SHAPES,
    ids=[shape_id(s) for s in _COMPRESSION_PARAM_SHAPES],
)
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES, ids=dtype_id)
@pytest.mark.parametrize("signed", [True, False], ids=["signed", "unsigned"])
def test_quantize_dequantize_roundtrip(
    test_case: str,
    seed: int,
    N: int,
    D: int,
    dtype: torch.dtype,
    signed: bool,
):
    """Test that dequantize(quantize(x)) ≈ x for various inputs and configurations."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = _make_test_tensor(test_case, (N, D), dtype, signed, gen)

    quantized, scales = quantize(x, signed=signed)
    x_reconstructed = dequantize(quantized, scales, signed=signed)

    _check_roundtrip_quality(x, x_reconstructed, test_case)


@pytest.mark.parametrize("signed", [True, False], ids=["signed", "unsigned"])
def test_maybe_quantized_tensor_quantized_state(signed: bool):
    """Test that _MaybeQuantizedTensor stores quantized state when try_quantize=True."""
    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    if not signed:
        x = x.abs()

    mqt = _MaybeQuantizedTensor(None, try_quantize=True, signed=signed)
    mqt.set_data(x)

    assert mqt.is_quantized()
    assert mqt.quantized is not None
    assert mqt.scales is not None
    assert mqt.data is None

    expected_int_dtype = torch.int8 if signed else torch.uint8
    assert mqt.quantized.dtype == expected_int_dtype
    assert mqt.scales.dtype == torch.float16


@pytest.mark.parametrize("dtype", _FLOAT_DTYPES, ids=dtype_id)
def test_maybe_quantized_tensor_passthrough(dtype: torch.dtype):
    """Test that _MaybeQuantizedTensor with try_quantize=False preserves data exactly."""
    x = torch.randn(32, 64, device="cuda", dtype=dtype)

    mqt = _MaybeQuantizedTensor(None, try_quantize=False, signed=True)
    mqt.set_data(x)

    assert not mqt.is_quantized()
    assert mqt.data is not None
    assert mqt.quantized is None

    x_out = mqt.materialize()
    torch.testing.assert_close(x_out, x.float())


# ---------------------------------------------------------------------------
# Non-aligned tail group test
# ---------------------------------------------------------------------------

# Sizes whose numel is NOT a multiple of GROUP_SIZE (32) or BLOCK_SIZE_N (1024)
_NONALIGNED_SIZES = [33, 65, 100, 999, 1000, 1025, 2000, 4097]

GROUP_SIZE = 32


@pytest.mark.parametrize(
    "N", _NONALIGNED_SIZES, ids=[f"N{n}" for n in _NONALIGNED_SIZES]
)
@pytest.mark.parametrize("signed", [True, False], ids=["signed", "unsigned"])
def test_quantize_nonaligned_tail_group_error(N: int, signed: bool):
    """Verify that tail (partial) groups have comparable error to interior groups.

    When numel is not a multiple of GROUP_SIZE, the last quantization group
    contains some zero-padded elements. This test ensures that the padding
    does not degrade reconstruction quality for the real tail elements.
    """
    gen = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randn(N, device="cuda", dtype=torch.float32, generator=gen) * 5.0
    if not signed:
        x = x.abs()

    quantized, scales = quantize(x, signed=signed)
    x_recon = dequantize(quantized, scales, signed=signed)

    tail_size = N % GROUP_SIZE
    if tail_size == 0:
        return  # no partial tail group

    err = (x - x_recon).abs()
    interior_err = err[:-tail_size]
    tail_err = err[-tail_size:]

    interior_max = interior_err.max().item()
    tail_max = tail_err.max().item()
    assert tail_max <= max(interior_max * 2.0, 1e-6), (
        f"Tail group error ({tail_max:.6e}) much worse than interior ({interior_max:.6e}) "
        f"for N={N}, signed={signed}"
    )

    # The tail group's scale should match the real data's absmax
    tail_data = x[-tail_size:]
    expected_absmax = tail_data.abs().max().item()
    actual_scale = scales[-1].item()
    assert abs(actual_scale - expected_absmax) / max(expected_absmax, 1e-12) < 0.01, (
        f"Tail scale {actual_scale} does not match expected absmax {expected_absmax}"
    )
