# Copyright 2026 Databricks AI Research authors

import dataclasses
import functools
from collections.abc import Callable
from typing import Optional

import torch
from reference import ReferenceAdamW, ReferenceLion, ReferenceSGDW
from torch.optim.adam import Adam as TorchAdam
from torch.optim.adamw import AdamW as TorchAdamW
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD as TorchSGD

from flashoptim import FlashAdam, FlashAdamW, FlashLion, FlashSGD, FlashSGDW

# Common dtypes and constants
_FLOAT_DTYPES = [torch.bfloat16, torch.float16, torch.float32]
_MASTER_WEIGHT_BITS = [None, 24, 32]
_WEIGHT_DECAY_VALUES = [0, 0.1]

_DTYPE_WIDTHS = {
    torch.int8: 1,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
}

# Common parameter shapes for testing
_MANY_PARAM_SHAPES = [(1, 1), (1, 2), (17, 23), (64, 32), (1, 1000)]
# CUDA-only configurations: (dtype, ecc_bits, quantized)
# Used for tests that assume fused=True or don't vary fused
DTYPE_ECC_QUANT_CONFIGS = [
    # float32 configurations (ECC not effective, so only ecc=None)
    (torch.float32, None, False),  # fp32, no ECC, unquantized
    (torch.float32, None, True),  # fp32, no ECC, quantized
    # float16 configurations (ECC effective for 24 and 32 bits)
    (torch.float16, None, False),  # fp16, no ECC, unquantized
    (torch.float16, None, True),  # fp16, no ECC, quantized
    (torch.float16, 24, False),  # fp16, 24-bit ECC, unquantized
    (torch.float16, 24, True),  # fp16, 24-bit ECC, quantized
    (torch.float16, 32, False),  # fp16, 32-bit ECC, unquantized
    (torch.float16, 32, True),  # fp16, 32-bit ECC, quantized
    # bfloat16 configurations (ECC effective for 24 and 32 bits)
    (torch.bfloat16, None, False),  # bf16, no ECC, unquantized
    (torch.bfloat16, None, True),  # bf16, no ECC, quantized
    (torch.bfloat16, 24, False),  # bf16, 24-bit ECC, unquantized
    (torch.bfloat16, 24, True),  # bf16, 24-bit ECC, quantized
    (torch.bfloat16, 32, False),  # bf16, 32-bit ECC, unquantized
    (torch.bfloat16, 32, True),  # bf16, 32-bit ECC, quantized
]  # Total: 14 configurations

# Reduced configs for distributed tests (DDP/FSDP2).
# Distributed correctness (gradient all-reduce, parameter sync) is orthogonal to
# dtype/ecc/quant — those dimensions are already fully covered by single-GPU tests.
DIST_DTYPE_ECC_QUANT_CONFIGS = [
    (torch.float32, None, False),  # fp32, no ECC, unquantized
    (torch.float32, None, True),  # fp32, no ECC, quantized
    (torch.bfloat16, 24, True),  # bf16, 24-bit ECC, quantized
    (torch.bfloat16, 32, True),  # bf16, 32-bit ECC, quantized
]

# CUDA-only configurations with fused dimension: (dtype, ecc_bits, quantized, fused)
# Used for tests that need to compare fused vs unfused
DTYPE_ECC_QUANT_FUSED_CONFIGS = [
    # All DTYPE_ECC_QUANT_CONFIGS with fused=True
    *[
        (dtype, ecc_bits, quant, True)
        for dtype, ecc_bits, quant in DTYPE_ECC_QUANT_CONFIGS
    ],
    # Unfused baselines
    (torch.float32, None, False, False),  # fp32, no ECC, unquantized, unfused
    (torch.bfloat16, 24, False, False),  # bf16, 24-bit ECC, unquantized, unfused
    (torch.bfloat16, 32, False, False),  # bf16, 32-bit ECC, unquantized, unfused
]  # Total: 17 configurations (14 fused + 3 unfused)

_DESCENDS_PARAM_SHAPES = [(1, 16), (17, 23), (32, 32)]


# Optimizer test configuration
@dataclasses.dataclass
class OptimizerTestConfig:
    """Configuration for optimizer testing.

    Attributes:
        name: Human-readable name for the optimizer (e.g., "FlashLion")
        factory: Factory function for the FlashOptim optimizer being tested
        reference_factory: Factory for the reference/torch optimizer used as baseline.
            Points to Torch* where an equivalent exists, or Reference* for custom WD semantics.
        state_var_names: Names of optimizer state variables (excludes error bits)
    """

    name: str
    factory: Callable[..., Optimizer]
    reference_factory: Callable[..., Optimizer]
    state_var_names: list[str]
    decouple_lr: bool = False


# Common optimizer configurations
LION_CONFIG = OptimizerTestConfig(
    name="FlashLion",
    factory=functools.partial(FlashLion, betas=(0.5, 0.75)),
    reference_factory=functools.partial(ReferenceLion, betas=(0.5, 0.75)),
    state_var_names=["exp_avg"],
)

SGDM_CONFIG = OptimizerTestConfig(
    name="FlashSGDM",
    factory=functools.partial(FlashSGD, momentum=0.9),
    reference_factory=functools.partial(TorchSGD, momentum=0.9),
    state_var_names=["momentum_buffer"],
)

SGD_ZERO_MOM_CONFIG = OptimizerTestConfig(
    name="FlashSGD_ZeroMom",
    factory=functools.partial(FlashSGD, momentum=0.0),
    reference_factory=functools.partial(TorchSGD, momentum=0.0),
    state_var_names=[],
)

SGDM_NESTEROV_CONFIG = OptimizerTestConfig(
    name="FlashSGDM_Nesterov",
    factory=functools.partial(FlashSGD, momentum=0.9, nesterov=True),
    reference_factory=functools.partial(TorchSGD, momentum=0.9, nesterov=True),
    state_var_names=["momentum_buffer"],
)

ADAMW_CONFIG = OptimizerTestConfig(
    name="FlashAdamW",
    factory=functools.partial(FlashAdamW, betas=(0.9, 0.999), eps=1e-8),
    reference_factory=functools.partial(TorchAdamW, betas=(0.9, 0.999), eps=1e-8),
    state_var_names=["exp_avg", "exp_avg_sq"],
)

ADAM_L2_CONFIG = OptimizerTestConfig(
    name="FlashAdam_L2",
    factory=functools.partial(FlashAdam, betas=(0.9, 0.999), eps=1e-8),
    reference_factory=functools.partial(TorchAdam, betas=(0.9, 0.999), eps=1e-8),
    state_var_names=["exp_avg", "exp_avg_sq"],
)

# decouple_lr=True variants — fully LR-decoupled weight decay
LION_DECOUPLE_LR_CONFIG = OptimizerTestConfig(
    name="FlashLion_DecoupleLR",
    factory=functools.partial(FlashLion, betas=(0.5, 0.75), decouple_lr=True),
    reference_factory=functools.partial(
        ReferenceLion, betas=(0.5, 0.75), decouple_lr=True
    ),
    state_var_names=["exp_avg"],
    decouple_lr=True,
)

ADAMW_DECOUPLE_LR_CONFIG = OptimizerTestConfig(
    name="FlashAdamW_DecoupleLR",
    factory=functools.partial(
        FlashAdamW, betas=(0.9, 0.999), eps=1e-8, decouple_lr=True
    ),
    reference_factory=functools.partial(
        ReferenceAdamW, betas=(0.9, 0.999), eps=1e-8, decouple_lr=True
    ),
    state_var_names=["exp_avg", "exp_avg_sq"],
    decouple_lr=True,
)

SGDMW_CONFIG = OptimizerTestConfig(
    name="FlashSGDMW",
    factory=functools.partial(FlashSGDW, momentum=0.9),
    reference_factory=functools.partial(ReferenceSGDW, momentum=0.9),
    state_var_names=["momentum_buffer"],
)

SGDM_DECOUPLE_LR_CONFIG = OptimizerTestConfig(
    name="FlashSGDM_DecoupleLR",
    factory=functools.partial(FlashSGDW, momentum=0.9, decouple_lr=True),
    reference_factory=functools.partial(ReferenceSGDW, momentum=0.9, decouple_lr=True),
    state_var_names=["momentum_buffer"],
    decouple_lr=True,
)


# ID generation functions with type hints
def dtype_id(dtype: torch.dtype) -> str:
    """Generate readable ID for torch dtypes."""
    return str(dtype).split(".")[-1]  # torch.bfloat16 -> bfloat16


def shape_id(n_d: tuple[int, int]) -> str:
    """Generate readable ID for N,D shape pairs."""
    n, d = n_d
    return f"{n}x{d}"


def weight_decay_id(wd: float) -> str:
    """Generate readable ID for weight_decay values."""
    if wd == 0:
        return "no_wd"
    return f"wd{wd}"


def lr_id(lr: float) -> str:
    """Generate readable ID for learning rate values."""
    if lr >= 1:
        return f"lr{lr:.0f}"
    elif lr >= 0.001:
        return f"lr{lr:.3f}"
    else:
        return f"lr{lr:.0e}"


def master_weight_bits_id(mb: Optional[int]) -> str:
    """Generate readable ID for master_weight_bits values."""
    if mb is None:
        return "noECC"
    return f"{mb}bit"


def compress_state_dict_id(compressed: bool) -> str:
    """Generate readable ID for compress_state_dict boolean."""
    return "compressed" if compressed else "uncompressed"


def quantized_state_id(quantized: bool) -> str:
    """Generate readable ID for quantized_state boolean."""
    return "quantized" if quantized else "unquantized"


def dtype_ecc_quant_id(combo: tuple[torch.dtype, Optional[int], bool]) -> str:
    """Generate readable ID for (dtype, ecc_bits, quantized) tuples."""
    dtype, ecc_bits, quantized = combo
    quant_str = "quant" if quantized else "unquant"
    ecc_str = "noECC" if ecc_bits is None else f"{ecc_bits}b"
    return f"{dtype_id(dtype)}_{ecc_str}_{quant_str}"


def dtype_ecc_quant_fused_id(
    combo: tuple[torch.dtype, Optional[int], bool, bool],
) -> str:
    """Generate readable ID for (dtype, ecc_bits, quantized, fused) tuples."""
    dtype, ecc_bits, quantized, fused = combo
    quant_str = "quant" if quantized else "unquant"
    fused_str = "fused" if fused else "unfused"
    ecc_str = "noECC" if ecc_bits is None else f"{ecc_bits}b"
    return f"{dtype_id(dtype)}_{ecc_str}_{quant_str}_{fused_str}"


def nmse(
    vals_true: torch.Tensor, vals_hat: torch.Tensor, norm_how: str = "l2_sq"
) -> float:
    """Normalized Mean Squared Error between two tensors.

    Args:
        vals_true: Ground truth tensor
        vals_hat: Approximation tensor
        norm_how: Normalization method — "l2_sq" (default) or "var"

    Returns:
        NMSE as a float scalar
    """
    diffs = vals_true - vals_hat
    mse = (diffs * diffs).mean()
    if norm_how == "var":
        return (mse / vals_true.var()).item()
    return (mse / (vals_true * vals_true).mean()).item()


# Minimum number of elements for cosine similarity to be meaningful.
# Small tensors (e.g. bias vectors) produce noisy cossim values.
MIN_NUMEL_FOR_COSSIM = 16


@dataclasses.dataclass
class Tolerances:
    """Tolerance thresholds for distributed numerical comparisons.

    Attributes:
        rtol: Relative tolerance for allclose
        atol: Absolute tolerance for allclose
        min_cossim: Minimum cosine similarity (1.0 = identical)
        max_nmse: Maximum normalized MSE (0.0 = identical)
    """

    rtol: float
    atol: float
    min_cossim: float
    max_nmse: float


# Strict tolerances for "should be identical" comparisons (e.g., initial params,
# DDP-vs-FSDP2 equivalence). Suitable for float32 unquantized comparisons.
STRICT_TOLERANCES = Tolerances(
    rtol=1e-5,
    atol=1e-7,
    min_cossim=0.99999,
    max_nmse=1e-8,
)


def check_tensor_similarity(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    name: str,
    rank: int = 0,
    tol: Tolerances = STRICT_TOLERANCES,
) -> list[str]:
    """Check two tensors are numerically close via allclose, cosine similarity, and NMSE.

    Returns:
        List of error messages (empty if all checks pass)
    """
    errors: list[str] = []

    if not torch.allclose(tensor1, tensor2, rtol=tol.rtol, atol=tol.atol):
        max_diff = (tensor1 - tensor2).abs().max().item()
        rel_diff = ((tensor1 - tensor2).abs() / (tensor2.abs() + 1e-8)).max().item()
        errors.append(
            f"[Rank {rank}] {name} not close. Max abs diff: {max_diff:.2e}, "
            f"Max rel diff: {rel_diff:.2e}, rtol={tol.rtol}, atol={tol.atol}"
        )

    t1_flat = tensor1.flatten().float()
    t2_flat = tensor2.flatten().float()

    if t1_flat.numel() >= MIN_NUMEL_FOR_COSSIM:
        cs = torch.cosine_similarity(t1_flat.unsqueeze(0), t2_flat.unsqueeze(0)).item()
        if cs < tol.min_cossim:
            errors.append(f"[Rank {rank}] {name} cossim={cs:.6f} < {tol.min_cossim}")

    nm = nmse(t1_flat, t2_flat)
    if nm > tol.max_nmse:
        errors.append(f"[Rank {rank}] {name} nmse={nm:.2e} > {tol.max_nmse}")

    return errors


def get_tolerances(model_dtype: torch.dtype, quantized: bool) -> Tolerances:
    """Get appropriate tolerances based on dtype and quantization.

    Lower precision dtypes (bfloat16, float16) and quantized optimizers need
    more relaxed tolerances due to:
    - Reduced mantissa precision (bfloat16 has 7 bits, float16 has 10 bits vs float32's 23)
    - Quantization errors in optimizer state storage
    - Non-associativity of floating-point operations

    Args:
        model_dtype: Model's data type (float32, bfloat16, float16)
        quantized: Whether the optimizer uses quantization

    Returns:
        Tolerances dataclass with rtol, atol, min_cossim, max_nmse
    """
    # cossim/NMSE thresholds set empirically with ~3x headroom over worst observed.
    if quantized:
        if model_dtype in (torch.bfloat16, torch.float16):
            return Tolerances(rtol=1.5e-2, atol=2e-3, min_cossim=0.9999, max_nmse=1e-3)
        elif model_dtype == torch.float32:
            return Tolerances(rtol=5e-4, atol=5e-4, min_cossim=0.99999, max_nmse=1e-5)
        else:
            raise ValueError(f"Unsupported dtype for tolerance: {model_dtype}")
    else:
        if model_dtype in (torch.bfloat16, torch.float16):
            return Tolerances(rtol=2e-3, atol=2e-4, min_cossim=0.9999, max_nmse=1e-3)
        elif model_dtype == torch.float32:
            return Tolerances(rtol=5e-6, atol=2e-7, min_cossim=0.99999, max_nmse=1e-8)
        else:
            raise ValueError(f"Unsupported dtype for tolerance: {model_dtype}")


# Usage patterns for pytest parametrize ids:
# 1. Single parameter matching function signature: ids=dtype_id (callable)
# 2. Multiple parameters or tuple parameters: ids=[func(item) for item in item_list] (list comprehension)
# 3. When function signature doesn't match pytest's calling pattern: use list comprehension
