"""FlashOptim: Efficient PyTorch optimizers with quantization and Triton kernels."""

from .optimizers import (
    FlashAdam,
    FlashAdamW,
    FlashLion,
    FlashOptimizer,
    FlashSGD,
    FlashSGDW,
    GradientReleaseHandle,
    NumericsError,
    cast_model,
    compute_ecc_bits,
    enable_gradient_release,
    reconstruct_fp32_param,
)

__version__ = "0.1.2"

__all__ = [
    "FlashAdam",
    "FlashAdamW",
    "FlashLion",
    "FlashOptimizer",
    "FlashSGD",
    "FlashSGDW",
    "GradientReleaseHandle",
    "NumericsError",
    "compute_ecc_bits",
    "cast_model",
    "enable_gradient_release",
    "reconstruct_fp32_param",
]
