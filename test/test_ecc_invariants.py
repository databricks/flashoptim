# Copyright 2026 Databricks AI Research authors

"""Test ECC invariants across all FP32 mantissa values.

This test verifies that error correction code (ECC) maintains key invariants:
1. If no-ecc has zero error, then 8-bit and 16-bit ECC also have zero error
2. If no-ecc has non-zero error, then ECC variants have strictly less error
3. 16-bit ECC is always at least as good as 8-bit ECC
"""

from typing import Union

import pytest
import torch
import triton
import triton.language as tl

from flashoptim.optimizers import (
    _apply_error_correction,
    _compute_ecc_bits,
)


@triton.jit
def check_ecc_invariants_kernel(
    # Inputs
    sign_ptr,
    exponent_ptr,
    total_mantissas,
    # Outputs - exception counts for 7 invariants
    exception_counts_ptr,
    # Constants
    NARROW_DTYPE: tl.constexpr,
    NUM_MANTISSA_BITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Check ECC invariants across all mantissa values.

    Invariants checked (let e0=no_ecc_error, e8=8bit_error, e16=16bit_error, r0=relative_error, r8=relative_error_8bit, r16=relative_error_16bit):
    1. perfect_downcast_8bit: e0 == 0 -> e8 == 0
    2. perfect_downcast_16bit: e0 == 0 -> e16 == 0
    3. 16bit_no_worse_zero: e8 == 0 -> e16 == 0
    4. 16bit_no_worse_always: e16 <= e8 (always)
    5. 8bit_corrects_representable: r0 >= 1/127 -> r8 < 1/127
    6. 16bit_corrects_representable: r0 >= 1/32767 -> r16 < 1/32767
    7. 16bit_better_in_middle_range: 1/32767 <= r0 < 1/127 -> e16 < e8

    Stores exception count (not bitstrings) to minimize I/O.
    """
    pid = tl.program_id(0)

    # Load configuration
    sign = tl.load(sign_ptr)
    exponent = tl.load(exponent_ptr)

    # Calculate mantissa values for this block
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mantissa_vals = block_start + offsets
    mask = mantissa_vals < total_mantissas

    # Construct FP32 values from components
    mantissa_float = mantissa_vals.to(tl.float32) / (1 << 23)

    if exponent == -127:  # FP32 denormal case
        fp32_vals = sign * tl.math.exp2(-126.0) * mantissa_float
    else:  # Normal case
        fp32_vals = (
            sign * tl.math.exp2(exponent.to(tl.float32)) * (1.0 + mantissa_float)
        )

    # Downcast to narrow dtype (no ECC)
    narrow_vals = fp32_vals.to(NARROW_DTYPE)

    # Mask out infinity values (overflow from narrow downcasting)
    is_finite = tl.abs(narrow_vals) != float("inf")
    mask = mask & is_finite

    # Compute 8-bit ECC
    ecc8_bits = _compute_ecc_bits(
        fp32_vals,
        narrow_vals,
        tl.int8,
        NUM_MANTISSA_BITS,
        127,
    )
    fp32_recon_ecc8 = _apply_error_correction(
        narrow_vals, ecc8_bits, NUM_MANTISSA_BITS, 127
    )

    # Compute 16-bit ECC
    ecc16_bits = _compute_ecc_bits(
        fp32_vals,
        narrow_vals,
        tl.int16,
        NUM_MANTISSA_BITS,
        32767,
    )
    fp32_recon_ecc16 = _apply_error_correction(
        narrow_vals, ecc16_bits, NUM_MANTISSA_BITS, 32767
    )

    # Calculate absolute errors
    no_ecc_error = tl.abs(narrow_vals.to(tl.float64) - fp32_vals.to(tl.float64))
    ecc8_error = tl.abs(fp32_recon_ecc8.to(tl.float64) - fp32_vals.to(tl.float64))
    ecc16_error = tl.abs(fp32_recon_ecc16.to(tl.float64) - fp32_vals.to(tl.float64))

    # Compute relative errors
    fp64_abs = tl.abs(fp32_vals).to(tl.float64)
    safe_denom = tl.where(fp64_abs > 0.0, fp64_abs, 1.0)
    relative_error = no_ecc_error / safe_denom
    relative_error_8bit = ecc8_error / safe_denom
    relative_error_16bit = ecc16_error / safe_denom

    # Check invariants and count exceptions
    # Invariant 1: perfect_downcast_8bit: e0 == 0 -> e8 == 0
    exception1 = (no_ecc_error == 0.0) & (ecc8_error > 0.0)
    count1 = tl.sum(tl.where(mask & exception1, 1, 0).to(tl.int32), axis=0)

    # Invariant 2: perfect_downcast_16bit: e0 == 0 -> e16 == 0
    exception2 = (no_ecc_error == 0.0) & (ecc16_error > 0.0)
    count2 = tl.sum(tl.where(mask & exception2, 1, 0).to(tl.int32), axis=0)

    # Invariant 3: 16bit_no_worse_zero: e8 == 0 -> e16 == 0
    exception3 = (ecc8_error == 0.0) & (ecc16_error > 0.0)
    count3 = tl.sum(tl.where(mask & exception3, 1, 0).to(tl.int32), axis=0)

    # Invariant 4: 16bit_no_worse_always: e16 <= e8 (always)
    exception4 = ecc16_error > ecc8_error
    count4 = tl.sum(tl.where(mask & exception4, 1, 0).to(tl.int32), axis=0)

    # Invariant 5: 8bit_corrects_representable: r0 >= 1/127 -> r8 < 1/127
    threshold_8bit_ecc = 1.0 / 127.0
    representable_8bit = relative_error >= threshold_8bit_ecc
    exception5 = representable_8bit & (relative_error_8bit >= threshold_8bit_ecc)
    count5 = tl.sum(tl.where(mask & exception5, 1, 0).to(tl.int32), axis=0)

    # Invariant 6: 16bit_corrects_representable: r0 >= 1/32767 -> r16 < 1/32767
    threshold_16bit_ecc = 1.0 / 32767.0
    representable_16bit = relative_error >= threshold_16bit_ecc
    exception6 = representable_16bit & (relative_error_16bit >= threshold_16bit_ecc)
    count6 = tl.sum(tl.where(mask & exception6, 1, 0).to(tl.int32), axis=0)

    # Invariant 7: 16bit_better_in_middle_range: 1/32767 <= r0 < 1/127 -> e16 < e8
    in_middle_range = representable_16bit & ~representable_8bit
    exception7 = in_middle_range & (ecc16_error >= ecc8_error)
    count7 = tl.sum(tl.where(mask & exception7, 1, 0).to(tl.int32), axis=0)

    # Atomic updates for aggregation across blocks
    tl.atomic_add(exception_counts_ptr + 0, count1)
    tl.atomic_add(exception_counts_ptr + 1, count2)
    tl.atomic_add(exception_counts_ptr + 2, count3)
    tl.atomic_add(exception_counts_ptr + 3, count4)
    tl.atomic_add(exception_counts_ptr + 4, count5)
    tl.atomic_add(exception_counts_ptr + 5, count6)
    tl.atomic_add(exception_counts_ptr + 6, count7)


_INVARIANT_NAMES = [
    "perfect_downcast_8bit",
    "perfect_downcast_16bit",
    "16bit_no_worse_zero",
    "16bit_no_worse_always",
    "8bit_corrects_representable",
    "16bit_corrects_representable",
    "16bit_better_in_middle_range",
]


def check_ecc_invariants(
    sign: int,
    exponent: int,
    narrow_dtype: str,
    device: torch.device = torch.device("cuda"),
) -> dict[str, int]:
    """Check ECC invariants for a given sign/exponent combination.

    Args:
        sign: Sign of the FP32 value (1 or -1)
        exponent: Signed exponent (-127 to 128)
        narrow_dtype: Narrow dtype to use ("bf16" or "fp16")
        device: CUDA device

    Returns:
        Dictionary mapping invariant name to exception count.
    """
    if narrow_dtype == "bf16":
        triton_narrow_dtype = tl.bfloat16
        num_mantissa_bits = 7
    else:  # fp16
        triton_narrow_dtype = tl.float16
        num_mantissa_bits = 10

    total_mantissas = 1 << 23  # 2^23 = 8,388,608

    sign_tensor = torch.tensor(sign, dtype=torch.float32, device=device)
    exponent_tensor = torch.tensor(exponent, dtype=torch.int32, device=device)
    exception_counts = torch.zeros(7, dtype=torch.int32, device=device)

    BLOCK_SIZE = 1024
    grid = ((total_mantissas + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    check_ecc_invariants_kernel[grid](
        sign_tensor,
        exponent_tensor,
        total_mantissas,
        exception_counts,
        NARROW_DTYPE=triton_narrow_dtype,
        NUM_MANTISSA_BITS=num_mantissa_bits,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    counts_cpu = exception_counts.cpu().numpy()
    return {name: int(counts_cpu[i]) for i, name in enumerate(_INVARIANT_NAMES)}


def _assert_invariants_within_tolerance(
    narrow_dtype: str,
    exponent_range: Union[range, list[int]],
    tolerance: dict[tuple[str, str], int],
) -> None:
    """Run ECC invariant checks for all signs/exponents and assert within tolerance.

    Args:
        narrow_dtype: "bf16" or "fp16"
        exponent_range: Range of exponents to test
        tolerance: Map of (narrow_dtype, invariant_name) -> max allowed exceptions
            per (sign, exponent) pair. Missing keys default to 0.
    """
    for sign in [1, -1]:
        for exponent in exponent_range:
            exceptions = check_ecc_invariants(sign, exponent, narrow_dtype)
            for invariant_name, count in exceptions.items():
                max_allowed = tolerance.get((narrow_dtype, invariant_name), 0)
                assert count <= max_allowed, (
                    f"Invariant '{invariant_name}' has {count} exceptions "
                    f"(max allowed: {max_allowed}) "
                    f"for {narrow_dtype.upper()}, sign={sign}, exponent={exponent}"
                )


@pytest.mark.parametrize("narrow_dtype", ["bf16", "fp16"])
def test_ecc_invariants_normal(narrow_dtype: str):
    """Test ECC invariants for normal FP32 values.

    Tests all normal exponent values:
    - BF16: exponents -126 to 127 (254 exponents)
    - FP16: exponents -14 to 15 (30 exponents)

    For each exponent, sweeps all 2^23 mantissa values.
    """
    if narrow_dtype == "bf16":
        exponent_range = range(-126, 128)
    else:  # fp16
        exponent_range = range(-14, 16)

    tolerance = {
        ("bf16", "16bit_no_worse_zero"): 32,
        ("bf16", "16bit_no_worse_always"): 32,
        ("bf16", "16bit_better_in_middle_range"): 32129,
        ("fp16", "16bit_better_in_middle_range"): 235656,
    }

    _assert_invariants_within_tolerance(narrow_dtype, exponent_range, tolerance)


@pytest.mark.parametrize("narrow_dtype", ["bf16", "fp16"])
def test_ecc_invariants_denormal(narrow_dtype: str):
    """Test ECC invariants for denormal FP32 values.

    Tests all denormal exponent values:
    - BF16: exponent -127 (1 denormal exponent)
    - FP16: exponents -30 to -15 (16 denormal exponents)

    For each exponent, sweeps all 2^23 mantissa values.
    """
    if narrow_dtype == "bf16":
        exponent_range = [-127]
    else:  # fp16
        exponent_range = range(-30, -14)

    tolerance = {
        ("bf16", "8bit_corrects_representable"): 8192,
        ("bf16", "16bit_corrects_representable"): 25,
        ("bf16", "16bit_better_in_middle_range"): 24319,
        ("fp16", "16bit_no_worse_zero"): 8192,
        ("fp16", "16bit_no_worse_always"): 16383,
        ("fp16", "8bit_corrects_representable"): 7656268,
        ("fp16", "16bit_corrects_representable"): 7602176,
        ("fp16", "16bit_better_in_middle_range"): 123952,
    }

    _assert_invariants_within_tolerance(narrow_dtype, exponent_range, tolerance)
