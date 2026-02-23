# SPDX-FileCopyrightText: Copyright 2026 Databricks, Inc.
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "NumericsError",
    "FlashOptimizer",
    "FlashLion",
    "FlashSGD",
    "FlashSGDW",
    "FlashAdam",
    "FlashAdamW",
    "GradientReleaseHandle",
    "cast_model",
    "compute_ecc_bits",
    "enable_gradient_release",
    "reconstruct_fp32_param",
]

import abc
import functools
import math
import os
import warnings
import weakref
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Literal, Optional, cast

import torch
import torch.nn as nn
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


class NumericsError(RuntimeError):
    """The optimizer detected that the learning rate is too small to
    meaningfully update weights at their current magnitude and dtype."""


_DTYPE_WIDTHS = {
    torch.int8: 1,
    torch.int16: 2,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
}
_EXPONENT_BITS = {
    torch.float16: 5,
    torch.bfloat16: 8,
    torch.float32: 8,
}
_NUM_MANTISSA_BITS = {
    torch.float16: 10,
    torch.bfloat16: 7,
    torch.float32: 23,
}

_TORCH_DTYPE_TO_TRITON_DTYPE = {
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
}

_VALID_MASTER_WEIGHT_BITS = (None, 24, 32)
_BITS_TO_BYTES = {None: 0, 24: 3, 32: 4}


@dataclass(frozen=True)
class QuantizedTensorSpec:
    signed: bool = True
    sqrt: bool = False
    softsign: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.signed, bool):
            raise ValueError(f"signed must be bool; got {type(self.signed)}")
        if not isinstance(self.sqrt, bool):
            raise ValueError(f"sqrt must be bool; got {type(self.sqrt)}")
        if not isinstance(self.softsign, bool):
            raise ValueError(f"softsign must be bool; got {type(self.softsign)}")


# most triton helpers go at the bottom, but this one is up here since we
# to write `@triton.autotune(configs=_generate_configs(), ...)`
def _generate_configs() -> list[triton.Config]:
    params = product(
        [2, 4, 8],  # num_warps
        [4, 8],  # elems_per_thread
        [1],  # num_stages
    )
    return [
        triton.Config(
            {"BLOCK_SIZE_N": numel_per_thread * 32 * num_warps},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for num_warps, numel_per_thread, num_stages in params
    ]


class _MaybeQuantizedTensor:
    """Helper class so optimizers don't have to know quantization details.

    Important points about this class:
    * It handles CPU tensors not being quantized
    * It knows how to save + load state dicts, handling both the quantized
        and not quantized cases
    * It implements some parts of the torch.Tensor interface that we need,
        but is not intended to be a full torch.Tensor replacement
    """

    def __init__(
        self,
        data: Optional[torch.Tensor],
        try_quantize: bool = True,
        signed: bool = True,
        sqrt: bool = False,
        softsign: bool = True,
    ):
        super().__init__()
        self.data: Optional[torch.Tensor] = None
        self.quantized: Optional[torch.Tensor] = None
        self.scales: Optional[torch.Tensor] = None
        self._try_quantize = try_quantize and torch.cuda.is_available()
        self._signed = signed
        self._sqrt = sqrt
        self._softsign = softsign

        if data is not None:
            self.set_data(data)

    @staticmethod
    def _quantized_vals_key(name: str) -> str:
        return f"{name}::quantized"

    @staticmethod
    def _quantized_scales_key(name: str) -> str:
        return f"{name}::scales"

    def state_dict(
        self, name: str, allow_quantized: bool = False
    ) -> dict[str, torch.Tensor]:
        if self.is_quantized() and allow_quantized:
            return {
                _MaybeQuantizedTensor._quantized_vals_key(name): self.quantized,
                _MaybeQuantizedTensor._quantized_scales_key(name): self.scales,
            }
        return {name: self.materialize().to(dtype=torch.bfloat16)}

    @classmethod
    def from_state_dict(
        cls,
        d: dict[str, torch.Tensor],
        name: str,
        try_quantize: bool = True,
        signed: bool = True,
        sqrt: bool = False,
        softsign: bool = True,
    ) -> "_MaybeQuantizedTensor":
        obj = cls(
            None, try_quantize=try_quantize, signed=signed, sqrt=sqrt, softsign=softsign
        )
        obj.load_state_dict(d, name)
        return obj

    def load_state_dict(self, d: dict[str, torch.Tensor], name: str) -> None:
        # we allow other keys in the state dict for convenience, so you can
        # just pass the whole opt state for a parameter
        d = {k: v for k, v in d.items() if k == name or k.startswith(f"{name}::")}
        if name in d:
            if len(d) != 1:
                raise ValueError(
                    f"If state dict specifies {name}, it must not "
                    + f"specify other keys. Got {list(d.keys())}"
                )
            self.set_data(d[name])
            return

        target_dtype = torch.int8 if self._signed else torch.uint8
        self.quantized = d[_MaybeQuantizedTensor._quantized_vals_key(name)].to(
            dtype=target_dtype
        )
        self.scales = d[_MaybeQuantizedTensor._quantized_scales_key(name)].to(
            dtype=torch.float16
        )

    def set_data(self, data: torch.Tensor) -> None:
        if self._try_quantize:
            if not data.is_cuda:
                raise NotImplementedError(
                    f"Attempting to quantize a non-CUDA {data.dtype} tensor "
                    + f"on device {data.device} with shape {data.shape}."
                )
            self.data = None
            self.quantized, self.scales = quantize(
                data, signed=self._signed, sqrt=self._sqrt, softsign=self._softsign
            )
        else:
            self.data = data.to(dtype=torch.float32)
            self.quantized = None
            self.scales = None

    def is_quantized(self) -> bool:
        return self.data is None

    def materialize(self) -> torch.Tensor:
        if not self.is_quantized():
            return self.data
        return dequantize(
            self.quantized,
            self.scales,
            signed=self._signed,
            sqrt=self._sqrt,
            softsign=self._softsign,
        )

    @property  # property to mirror Tensor interface
    def is_cuda(self) -> bool:
        if self.is_quantized():
            return self.quantized.is_cuda
        return self.data.is_cuda

    @property  # property to mirror Tensor interface
    def shape(self) -> tuple[int, ...]:
        if self.is_quantized():
            return tuple(self.quantized.shape)
        return tuple(self.data.shape)

    def numel(self) -> int:
        if self.is_quantized():
            return self.quantized.numel()
        return self.data.numel()

    def __repr__(self):
        return (
            f"{self.__class__.__name__} quantized={self.is_quantized()} signed={self._signed} "
            + f"shape={self.shape}"
        )

    @property
    def kernel_tensor(self) -> torch.Tensor:
        """Returns the appropriate tensor representation for kernel operations.

        Returns quantized int8/uint8 tensor if quantized, else fp32 data tensor.
        """
        if self.is_quantized():
            assert self.quantized is not None
            return self.quantized
        assert self.data is not None
        return self.data

    @property
    def kernel_scales_or_self(self) -> torch.Tensor:
        """Returns scales for kernel operations, or the tensor itself as a dummy.

        Returns fp16 scales tensor if quantized, otherwise returns the main tensor
        to serve as an unused dummy pointer (Triton kernels require valid pointers
        even for unused parameters).
        """
        if self.is_quantized():
            assert self.scales is not None
            return self.scales
        return self.kernel_tensor


# ======================================= FlashOptimizer Base Class and Helpers


def _log2_min_expressible_step_size(
    dtype: torch.dtype, maxabs: float, master_bytewidth: int
) -> float:
    # Here's the idea with this function: any given value has a certain
    # exponent, and each exponent is associated with a fixed numerical
    # resolution. This resolution is 2^(-num_mantissa_bits) * exponent_pow2,
    # where exponent_pow2 is the greatest power of two that the value in
    # question exceeds. The "value in question" here, maxabs, is the value in
    # a tensor with the largest absolute value, since this value will have
    # the lowest resolution.
    #
    # Now consider two subtleties:
    #   1) It suffices to have steps that are just barely larger than
    #   the resolution, since such steps are enough to bump us up or down to
    #   an adjacent value.
    #   2) We may have error correction bits, which effectively give us a
    #   dtype that has the same sign bit and exponent bits as the original
    #   dtype, but with more mantissa bits.
    #
    master_bits = max(_DTYPE_WIDTHS[dtype], master_bytewidth) * 8
    num_mantissa_bits = master_bits - 1 - _EXPONENT_BITS[dtype]
    num_mantissa_bits += math.log2(0.49)  # step just needs to round to next value
    maxabs_exponent = math.floor(math.log2(maxabs))
    return maxabs_exponent - num_mantissa_bits


class FlashOptimizer(torch.optim.Optimizer, abc.ABC):
    """Base class for our optimized optimizers.

    Implements five (optional) speed and space optimizations:
        - Quantized optimizer states, which use 8.5 bits per parameter instead
            of 32. The extra .5 bits is from storing scale factors.
            Savings:
                -23.5 bits (~3 bytes) of RAM per parameter
                -Slightly faster optimizer steps
            Caveats:
                -Should increase training loss by a factor of <1.001x, but
                your mileage may vary.

        - Compressed state dicts, which store quantized optimizer states.
            Savings:
                -2 or ~3 bytes of checkpoint size per parameter.
            Caveats:
                -Quantized state dicts are incompatible with FSDP if you need
                 unless you resume with the exact same sharding. But even with
                 these turned off, the optimizer states are saved as bf16
                 instead of f32 whenever quantize=True, so you still save
                 2 bytes/param.

        - Fused optimizer steps, which put all the logic in one kernel.
            Savings:
                -No space saved, but program runs faster.
            Caveats:
                -No drawbacks, but many torch optimizers already have fused
                steps; those implementations just lack the other optimizations.

        - Error correction bits, which let us decouple the master weight bitwidth
            from the bitwidth used in model.parameters(). This lets you just
            keep your parameters in bf16 and not have a separate cast in
            the forward pass. It also lets you use 24-bit master weights instead
            of 32-bit if you have a large enough learning rate to support this.
            Savings:
                -1 byte of RAM per param with 24-bit master weights
                -Likely faster forward and/or backward passes, depending on
                mixed precision autocast and sharding settings
            Caveats:
                -You need a high enough ratio of update size to parameter
                size to get away with using narrower master weights, or else
                updates won't change the master weights. You need your steps
                to consistently be of size |param|*2^(-num_mantissa_bits+1).

        - Gradient release, meaning we update each parameter as soon as its
            gradient is ready in the backward pass and then free its gradient.
            This lets us avoid materializing more than a handful of gradients at
            a time.
            Savings:
                -4 bytes of RAM per parameter with (typical) 4-byte gradients.
            Caveats:
                -Gradient release is incompatible with global gradient clipping.

    With all of these optimizations, you can go from a typical:
        -8 bytes of optimizer state per parameter
        -4 bytes of master weight per parameter
        -4 bytes of gradient per parameter
        = 16 bytes/param, plus casting overhead in forward and backward passes
    to:
        -1 byte of optimizer state per parameter
        -3 bytes of master weight per parameter
        -0 bytes of gradient per parameter
        = 4 bytes/param, plus no casting overhead

    Args:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        weight decay: Weights are multiplied by 1 - `weight_decay` after
            each optimizer step. Note that we use decoupled weight decay,
            meaning that this decay does not contribute to the momentum.
        compress_state_dict: if True, this optimizer's `state_dict` will
            include quantized optimizer states. Otherwise, the optimizer
            states are converted to bfloat16 Tensors matching the shapes of
            their corresponding parameters. The former uses ~8.5 bits per
            parameter while the latter uses 16 bits per parameter. However,
            the former is less thoroughly tested and will not work with
            FSDP or other weight sharding approaches.
        quantize: If True (default), optimizer states (momentum,
            variance) are quantized to INT8, reducing memory from 32 to
            ~8.5 bits per parameter. Set to False to keep states in FP32
            for debugging convergence. Requires CUDA.
        master_weight_bits: Effective precision (in bits) for master weights.
            When set to 24 or 32, the optimizer stores error-correction bits
            alongside each low-precision parameter, enabling higher effective
            master weight precision without a full FP32 copy. For bf16
            parameters, 24 (default) adds an INT8 correction tensor
            (1 byte/param) and 32 adds an INT16 correction tensor
            (2 bytes/param). Set to None to disable and use native parameter
            precision. Supported: {None, 24, 32}.
        check_numerics: If true, will check that the learning rate is large
            enough to perturb the largest values in each tensor to a different
            bit string (i.e., not ignore the update). This calculation takes
            into account each parameter's dtype, `master_weight_bits`, the
            current learning rate, and a *cached* measurement of the largest
            absolute value in each tensor. You must call
            `recompute_param_stats()` to update this cached value; to avoid
            overhead, it is only called automatically during the first step
            and after loading a state dict.
    .. NOTE:

        Because `check_numerics` recomputes the parameter standard deviations
        during load_state_dict(), it is possible that saving and reloading a
        checkpoint might result in a new `NumericsError` error appearing. Call
        recompute_param_stats() before saving a checkpoint to avoid this
        discrepancy, or simply set `check_numerics=False`.

    .. NOTE:

        Parameters within the same param_group must have the same dtype.
        Violating this rule can cause undefined behavior when loading a state
        dict.

    .. NOTE:

        Do not change the dtypes of parameters between saving and loading a
        state dict if you also try to load the error correction bits. This
        results in undefined behavior.

    .. NOTE:

        Due to assumptions in torch's Optimizer class, you cannot currently
        load a state dict that lacks error correction bits if this optimizer
        instance is set to use them.

    Raises:
        ValueError - If the hyperparameters fail sanity checks, such as
            having a learning rate greater than zero.
        NotImplementedError - If any of `quantize`, `compress_state_dict`,
            or `error_correction` are `True` and either a) there is no CUDA
            device, or b) step() is executed on a non-CUDA parameter.
        NumericsError - If check_numerics is True and estimates
            that the learning rate is too small to alter the weights.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        quantize: bool = True,
        compress_state_dict: bool = False,
        master_weight_bits: Literal[24, 32] | None = 24,
        check_numerics: bool = False,
        fused: bool = True,
        # defaults is here so subclasses can pass custom hparams to their
        # param groups
        defaults: Optional[dict[str, Any]] = None,
    ):
        if compress_state_dict and not quantize:
            raise ValueError("compress_state_dict=True requires quantize=True")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if master_weight_bits not in _VALID_MASTER_WEIGHT_BITS:
            raise ValueError(
                f"Master weight bits {master_weight_bits} "
                + f"invalid; must be one of {_VALID_MASTER_WEIGHT_BITS}"
            )

        # Convert public API bits to internal byte representation once
        master_bytewidth = _BITS_TO_BYTES[master_weight_bits]

        if not torch.cuda.is_available():
            needs_cuda = " requires a CUDA device."
            if quantize:
                raise NotImplementedError("Quantization" + needs_cuda)
            if compress_state_dict:
                raise NotImplementedError("Quantized state dict" + needs_cuda)
            if master_weight_bits is not None:
                raise NotImplementedError("Using error correction bits" + needs_cuda)
            if fused:
                raise NotImplementedError("Fused optimizer step" + needs_cuda)

        self._fused = fused

        self._quantize = quantize
        self._master_bytewidth = master_bytewidth
        self._compress_state_dict = compress_state_dict
        self._check_numerics = check_numerics
        defaults = defaults or {}
        defaults.update(
            {
                "lr": lr,
                "initial_lr": lr,
                "master_bytewidth": master_bytewidth,
                "quantize": quantize,
            }
        )
        # Keyed by id(param) — stores lightweight metadata (e.g. maxabs)
        # that should NOT be serialized in state_dict.
        self._transient_state: dict[int, dict[str, Any]] = {}

        # Gradient-release state (set by enable_gradient_release, cleared by GradientReleaseHandle.remove)
        self._gradient_release = False

        super().__init__(params, defaults=defaults)

        # Validate that ECC is meaningful when master_weight_bits is 24 or 32
        _allow = os.environ.get("FLASHOPTIM_ALLOW_INEFFECTIVE_MASTER_WEIGHT_BITS")
        if master_weight_bits is not None and not _allow:
            all_params_fp32 = all(
                p.dtype == torch.float32
                for group in self.param_groups
                for p in group["params"]
            )
            if all_params_fp32:
                raise ValueError(
                    f"master_weight_bits={master_weight_bits} has no effect when all "
                    "parameters are fp32, since fp32 already uses 32 bits. "
                    "No error correction would be applied in this scenario. "
                    "Please omit master_weight_bits (or set it to None)."
                )

    def __repr__(self) -> str:
        header = f"{self.__class__.__name__} (\n"
        header += "FlashOptim Config\n"
        flash_config = {
            "fused": self._fused,
            "quantize": self._quantize,
            "master_bytewidth": self._master_bytewidth,
            "compress_state_dict": self._compress_state_dict,
            "check_numerics": self._check_numerics,
            "gradient_release": self._gradient_release,
        }
        for key, value in flash_config.items():
            header += f"    {key}: {value}\n"
        for i, group in enumerate(self.param_groups):
            header += "\n"
            header += f"Parameter Group {i}\n"
            for key in sorted(group.keys()):
                if key != "params":
                    header += f"    {key}: {group[key]}\n"
        header += ")"
        return header

    @abc.abstractmethod
    def _quantized_state_spec(self) -> dict[str, QuantizedTensorSpec]: ...

    @staticmethod
    def _get_tensor_for_stats(tensor: torch.Tensor) -> torch.Tensor:
        """Get tensor suitable for stats computation, handling FSDP2 DTensors.

        FSDP2's fully_shard() converts parameters to DTensors (distributed tensors).
        Some operations like .abs().max().item() don't work directly on DTensors.
        This helper extracts the full tensor for stats computation.

        Args:
            tensor: Parameter tensor (may be DTensor or regular Tensor)

        Returns:
            Regular tensor suitable for stats computation
        """
        if hasattr(tensor, "full_tensor"):
            return tensor.full_tensor()
        return tensor

    @staticmethod
    def _get_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Get local tensor from potentially sharded DTensor.

        FSDP2's fully_shard() converts parameters to DTensors. Triton kernels
        cannot operate on DTensors directly - they need the local tensor shard.
        This helper extracts the local tensor for kernel operations.

        The returned tensor shares storage with the DTensor, so in-place
        modifications will affect the original DTensor.

        Args:
            tensor: Parameter tensor (may be DTensor or regular Tensor)

        Returns:
            Local tensor suitable for kernel operations
        """
        if hasattr(tensor, "to_local"):
            return tensor.to_local()
        return tensor

    def _recompute_stats_for_param(self, p: torch.Tensor) -> None:
        p_for_stats = self._get_tensor_for_stats(p)
        maxabs = (
            float(p_for_stats.detach().abs().max().item()) if p_for_stats.numel() else 0
        )
        self._transient_state.setdefault(id(p), {})["maxabs"] = maxabs

    def recompute_param_stats(self) -> None:
        """Writes each parameter's maximum absolute value to its transient state dict.

        Call this to ensure that the numerics checks that run at the next step
        reflect the latest parameter statistics, instead of stale, cached
        statistics.
        """
        for group in self.param_groups:
            for p in group["params"]:
                assert isinstance(p, torch.Tensor)
                self._recompute_stats_for_param(p)

    @torch.no_grad()
    def step_param(
        self,
        p: torch.Tensor,
        group: dict[str, Any] | None = None,
    ) -> None:
        if group is None:
            group = self._find_group(p)

        quantize = group.get("quantize", self._quantize)
        if quantize and not p.is_cuda:
            raise NotImplementedError(
                f"Can't use quantization with param on {p.device} "
                + f"({p.shape}, {p.dtype}). If you need to use this class "
                + "without a CUDA device, try creating it with "
                + "quantize=False."
            )

        self._ensure_state_initialized(p, hparams=group)
        param_state = self.state[p]

        # it's possible to pass in tensors of size 0 if param is smaller than
        # world size; but these params still need their states initialized for
        # distributed checkpointing logic, so only exit the function here
        if p.numel() < 1:
            return
        # similarly, distributed checkpoint logic needs state initialized for
        # every tensor in every group's params, even if it doesn't actually
        # need a gradient (as of torch 2.3). So don't early exit until here.
        if not p.requires_grad or p.grad is None:
            return

        # with the current learning rate and cached param statistics, should
        # we expect to hit numerical issues?
        master_bytewidth = group["master_bytewidth"]
        if self._check_numerics:
            self._check_param_numerics(
                p, lr=group["lr"], master_bytewidth=master_bytewidth
            )

        # FSDP2 support: extract local tensors from DTensors for Triton kernels.
        # DTensors are distributed tensors that can't be passed to Triton directly.
        # to_local() returns a view of the local shard, so in-place ops work correctly.
        p_local = self._get_local_tensor(p)
        grad_local = self._get_local_tensor(p.grad) if p.grad is not None else None

        # Triton kernels use flat pointer arithmetic (ptr + offset) which
        # assumes contiguous memory. Non-contiguous tensors (e.g. transposed
        # params from tied weights) would silently read/write wrong elements.
        if self._fused:
            if not p_local.is_contiguous():
                raise RuntimeError(
                    f"FlashOptimizer fused kernels require contiguous parameters, "
                    f"but got param with shape {p_local.shape} and stride {p_local.stride()}. "
                    f"Call .contiguous() on the parameter or use fused=False."
                )
            if grad_local is not None and not grad_local.is_contiguous():
                raise RuntimeError(
                    f"FlashOptimizer fused kernels require contiguous gradients, "
                    f"but got grad with shape {grad_local.shape} and stride {grad_local.stride()}. "
                    f"Use fused=False for non-contiguous gradients."
                )

        errors: torch.Tensor | None = None
        if "error_bits" in param_state:
            errors = param_state["error_bits"]

        self._do_step(
            param=p_local,
            grad=grad_local,
            errors=errors,
            param_state=param_state,
            hparams=group,
        )

    def _find_group(self, p: torch.Tensor) -> dict[str, Any]:
        for group in self.param_groups:
            if any(param is p for param in group["params"]):
                return group
        raise ValueError(
            f"Parameter {p.shape} {p.dtype} not managed by this optimizer."
        )

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> torch.Tensor | None:
        if self._gradient_release:
            warnings.warn(
                "optimizer.step() is a no-op while enable_gradient_release() is active — "
                "parameters are stepped by gradient hooks.",
                stacklevel=2,
            )
            return None
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                self.step_param(p, group)
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self._gradient_release:
            warnings.warn(
                "optimizer.zero_grad() is a no-op while enable_gradient_release() is active — "
                "gradients are freed by gradient hooks.",
                stacklevel=2,
            )
            return
        super().zero_grad(set_to_none=set_to_none)

    def _load_state_for_param(
        self,
        param: torch.Tensor,
        opt_state: dict[torch.Tensor, Any],
        hparams: dict[str, Any],
    ) -> dict[str, Any]:
        param_state = opt_state[param]
        new_state = {}

        # State dict change 1 of 3: Copy non-quantized keys (like step)
        quantized_keys = self._quantized_state_spec()
        for key, value in param_state.items():
            if not any(key == qk or key.startswith(f"{qk}::") for qk in quantized_keys):
                new_state[key] = value

        # State dict change 2 of 3: quantize state if needed.
        quantize = hparams.get("quantize", self._quantize)
        for key_quant, spec in quantized_keys.items():
            if any(k.startswith(key_quant) for k in param_state):
                # The keys can either be just "exp_avg" or
                # "exp_avg::quantized" and "exp_avg::scales", depending on
                # whether we saved it as quantized or not. The former case
                # also gives us interop with non-quantized optimizers.
                qtensor = _MaybeQuantizedTensor.from_state_dict(
                    param_state,
                    name=key_quant,
                    try_quantize=quantize,
                    signed=spec.signed,
                    sqrt=spec.sqrt,
                    softsign=spec.softsign,
                )
                new_state[key_quant] = qtensor

        # State dict change 3 of 3: load+check error correction state if present
        master_bytewidth = hparams.get("master_bytewidth")
        if master_bytewidth is None:
            return new_state  # never load errors if master_bytewidth not saved
        master_bytewidth = int(master_bytewidth)

        # If no errors saved, we're done
        errs = param_state.get("error_bits", None)
        if errs is None:
            return new_state

        # Handle errors that had to be upcast to workaround torch quirks; we
        # assume this happened anytime the raw errors tensor we read in has
        # too wide a dtype for the param and saved master weight bits.
        # NOTE: this can yield incorrect behavior if the user changes the param
        # dtype between checkpoint save and load, but...that's a bizarre edge
        # case and I'm not sure why anyone would do that. And if you really
        # need that, you can just not load the opt state from the checkpoint.
        param_bytewidth = _DTYPE_WIDTHS[param.dtype]
        need_err_bytewidth = master_bytewidth - param_bytewidth
        orig_err_dtype = errs.dtype
        if need_err_bytewidth == 1 and errs.dtype != torch.int8:
            errs = errs.to(dtype=torch.int8)
        elif need_err_bytewidth == 2 and errs.dtype != torch.int16:
            errs = errs.view(dtype=torch.int16)

        # sanity check bytewidths before returning
        if param_bytewidth + _DTYPE_WIDTHS[errs.dtype] != master_bytewidth:
            raise ValueError(
                f"Loading errors of dtype {orig_err_dtype} for "
                + f"param of dtype {param.dtype} cannot yield "
                + f"master_weight_bits={master_bytewidth * 8}."
            )

        new_state["error_bits"] = errs
        return new_state

    def _state_dict_for_param(
        self,
        param_number: int,
        opt_state: dict[int, Any],
        hparams: dict[str, Any],
        param_dtype_map: dict[int, torch.dtype],
    ) -> dict[str, Any]:
        # Make a copy so that we don't mutate our self.state. `opt_state`
        # isn't the same as self.state, but its consituent dicts are
        # the same as those in self.state; i.e., `opt_state[p] is self.state[p]`
        param_state = dict(opt_state[param_number].items())

        # State dict change 1 of 2: undo quantization if needed.
        for key_quant in self._quantized_state_spec().keys():
            if key_quant in param_state:  # true if we've taken any steps
                # If the user hasn't opted into storing compressed state dicts
                # we have to make sure our states are regular torch.Tensors.
                # This is mostly needed to make FSDP happy in the case that
                # we want to resume training with a number of devices where
                #   (param numel / device count) % quantization group size != 0
                # for any param.
                qtensor = param_state.pop(key_quant)
                assert isinstance(qtensor, _MaybeQuantizedTensor)
                param_state.update(
                    qtensor.state_dict(
                        name=key_quant, allow_quantized=self._compress_state_dict
                    )
                )

        # State dict change 2 of 2: have error-correction state appease FSDP.
        if "error_bits" in param_state:
            # FSDP assumes that all non-scalar opt states have the
            # same shape as the params. Since Optimizer also casts these
            # states to the param dtype, we have to carefully change the
            # dtypes if the errors and params don't have the same bitwidth.
            # The goal here is to make Optimizer's cast op a no-op.
            param_dtype = param_dtype_map[param_number]
            param_bytewidth = _DTYPE_WIDTHS[param_dtype]
            error_bytewidth = hparams["master_bytewidth"] - param_bytewidth
            errs = param_state["error_bits"]
            if error_bytewidth == param_bytewidth:
                param_state["error_bits"] = errs.view(dtype=param_dtype)
            elif error_bytewidth == 1 and param_bytewidth == 2:
                # uint8 -> [b]f16 conversion is lossless, so this is safe
                param_state["error_bits"] = errs.to(dtype=param_dtype)
            else:
                raise NotImplementedError(
                    f"Cannot save {error_bytewidth}B error correction tensor "
                    + f"for {param_bytewidth}B parameter. This is because FSDP "
                    + "requires that non-scalar state variables have the same "
                    + "shape as their param, while optim.Optimizer auto-casts "
                    + "state tensors to the param dtype before calling "
                    + "load_state_dict()."
                    + "It is not possible to work around both of these "
                    + "behaviors simultaneously without information loss "
                    + "(or far more implementation complexity)."
                )
        return param_state

    def __setstate__(self, state: dict[str, dict[Any, Any]]) -> None:
        opt_state = state["state"]
        param_groups = state["param_groups"]
        # Can't assert because dict[...] is not a concrete type.
        opt_state = cast(dict[torch.Tensor, Any], opt_state)
        param_groups = cast(list[dict[str, Any]], param_groups)
        for group in param_groups:
            for param in group["params"]:
                assert isinstance(param, torch.Tensor)
                opt_state[param] = self._load_state_for_param(
                    param, opt_state, hparams=group
                )
            # set default master_bytewidth here so _load_state_for_param()
            # knows that this field came from the state dict if it's present
            # and isn't just our own default value
            group.setdefault("master_bytewidth", self._master_bytewidth)
        super().__setstate__(state)

    def state_dict(self):
        d = super().state_dict()
        opt_state = d["state"]
        param_groups = d["param_groups"]

        # Build param index -> dtype mapping without mutating self.state
        param_dtype_map: dict[int, torch.dtype] = {}
        for orig_group, saved_group in zip(self.param_groups, param_groups):
            for param, idx in zip(orig_group["params"], saved_group["params"]):
                param_dtype_map[idx] = param.dtype

        for group in param_groups:
            for param_number in group["params"]:
                assert isinstance(param_number, int)
                opt_state[param_number] = self._state_dict_for_param(
                    param_number,
                    opt_state=opt_state,
                    hparams=group,
                    param_dtype_map=param_dtype_map,
                )

        return d

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state, ensuring backward compatibility with vanilla PyTorch optimizers.

        Vanilla PyTorch optimizers (like AdamW) don't include 'initial_lr' in their
        state dicts unless a learning rate scheduler has been used. This method ensures
        'initial_lr' is present in all param_groups after loading, defaulting to the
        current 'lr' value if missing.
        """
        super().load_state_dict(state_dict)
        # Ensure initial_lr is present for all param_groups
        for group in self.param_groups:
            if "initial_lr" not in group:
                group["initial_lr"] = group["lr"]

    def _ensure_state_initialized(
        self, p: torch.Tensor, hparams: dict[str, Any]
    ) -> None:
        # Part 1: initialize + check numerics stats if needed.
        maybe_std = self._transient_state.get(id(p), {}).get("maxabs")
        if self._check_numerics and maybe_std is None:
            self._recompute_stats_for_param(p)

        state = self.state[p]

        # Part 2: initialize quantized state vars if needed.
        # FSDP2 support: state tensors must be created from local tensors, not DTensors.
        # This ensures each rank has state for its local parameter shard.
        p_local = self._get_local_tensor(p)

        quantize = hparams.get("quantize", self._quantize)
        for key_quant, spec in self._quantized_state_spec().items():
            if key_quant not in state:
                state[key_quant] = _MaybeQuantizedTensor(
                    self._initial_value_for_state_var(p_local, key_quant),
                    try_quantize=quantize,
                    signed=spec.signed,
                    sqrt=spec.sqrt,
                    softsign=spec.softsign,
                )

        # Part 3: initialize master weight error correction if needed.
        master_bytewidth = hparams["master_bytewidth"]
        num_err_bytes = master_bytewidth - _DTYPE_WIDTHS[p.dtype]
        if state.get("error_bits") is None and num_err_bytes > 0:
            if num_err_bytes == 1:
                errors = torch.zeros_like(p_local, dtype=torch.int8)
            else:
                errors = torch.zeros_like(p_local, dtype=torch.int16)
            state["error_bits"] = errors

    def _initial_value_for_state_var(
        self, param: torch.Tensor, key: str
    ) -> torch.Tensor:
        """Hook to let subclasses do something besides zero-initialize state."""
        del key
        return torch.zeros_like(param)

    def _min_step_size_relative_to_lr(self) -> float:
        return 1.0  # true for LION and SignSGD; too large for most subclasses

    def get_fp32_model_state_dict(
        self,
        model: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Reconstruct fp32 model state dict from optimizer state.

        This method is useful for loading fp32 weights for inference when using FlashOptim
        with ECC-compressed master weights (master_weight_bits is not None).

        For parameters with ECC (master_weight_bits is not None):
        - Applies ECC reconstruction using error_bits

        For parameters without ECC (master_weight_bits=None):
        - Returns parameter upcast to fp32

        Args:
            model: The model being optimized. If using DDP, pass the DDP-wrapped model
                   and this method will handle unwrapping.

        Returns:
            State dict with all parameters in fp32 precision.

        Example:
            >>> model_fp32 = timm.create_model(..., dtype=torch.float32)
            >>> fp32_state = optimizer.get_fp32_model_state_dict(model)
            >>> model_fp32.load_state_dict(fp32_state)
            >>> # Now model_fp32 has reconstructed fp32 weights for inference
        """
        # Handle DDP wrapper
        model_to_use = model.module if hasattr(model, "module") else model

        # Start with the full model state dict (includes parameters and buffers)
        fp32_state = {
            k: v.detach().float() for k, v in model_to_use.state_dict().items()
        }

        # Replace parameters with ECC-reconstructed fp32 versions where available
        for name, param in model_to_use.named_parameters():
            state = self.state.get(param, {})
            if "error_bits" in state:
                # Has ECC - reconstruct fp32 and replace
                # FSDP2: extract local tensor from DTensor for Triton kernels
                local_param = self._get_local_tensor(param.detach())
                fp32_state[name] = reconstruct_fp32_param(
                    local_param, state["error_bits"]
                )

        return fp32_state

    def set_fp32_model_state_dict(
        self,
        model: torch.nn.Module,
        fp32_state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Set model weights and optimizer ECC state from fp32 state dict.

        Inverse of get_fp32_model_state_dict(). For each parameter with ECC enabled:
        - Casts fp32 weights to the parameter's dtype and updates the model
        - Computes and stores ECC error_bits in optimizer state

        Args:
            model: The model being optimized. If using DDP, pass the DDP-wrapped model
                   and this method will handle unwrapping.
            fp32_state_dict: State dict with fp32 weights to load

        Example:
            >>> # Save fp32 weights during training
            >>> fp32_state = optimizer.get_fp32_model_state_dict(model)
            >>> torch.save(fp32_state, "checkpoint_fp32.pt")
            >>>
            >>> # Later, load fp32 weights back into model + optimizer
            >>> fp32_state = torch.load("checkpoint_fp32.pt")
            >>> optimizer.set_fp32_model_state_dict(model, fp32_state)
        """
        # Handle DDP wrapper
        model_to_use = model.module if hasattr(model, "module") else model

        # Update each parameter and its ECC state
        for name, param in model_to_use.named_parameters():
            if name not in fp32_state_dict:
                continue

            fp32_value = fp32_state_dict[name]

            # FSDP2: extract local tensors from DTensors. Triton kernels and
            # in-place copy_ cannot operate on DTensors directly.
            local_param = self._get_local_tensor(param.data)
            local_fp32 = self._get_local_tensor(fp32_value)

            # Ensure fp32_value is on the same device as param
            if local_fp32.device != local_param.device:
                local_fp32 = local_fp32.to(local_param.device)

            # Cast to parameter dtype and update model parameter
            local_param.copy_(local_fp32.to(param.dtype))

            # Update optimizer ECC state if present
            state = self.state.get(param, {})
            if "error_bits" in state:
                state["error_bits"] = compute_ecc_bits(local_fp32, local_param)

    def _check_param_numerics(
        self, p: torch.Tensor, lr: float, master_bytewidth: int
    ) -> None:
        if lr == 0.0:
            return  # dummy steps are allowed
        maxabs = self._transient_state.get(id(p), {}).get("maxabs", 0)
        if maxabs <= 0:
            return
        if not math.isfinite(maxabs):
            # Parameter has infinite values - this indicates numerical instability
            # from the loaded state. Skip numerics check to allow the step to proceed.
            return
        min_step_needed = _log2_min_expressible_step_size(
            dtype=p.dtype, maxabs=maxabs, master_bytewidth=master_bytewidth
        )

        effective_bits = max(_DTYPE_WIDTHS[p.dtype], master_bytewidth) * 8
        min_step_guaranteed = math.log2(lr * self._min_step_size_relative_to_lr())
        if min_step_needed > min_step_guaranteed:
            raise NumericsError(
                f"Learning rate {lr:.3g} yields steps that may be too small "
                + f"to alter the weights for dtype {p.dtype} with "
                + f"master_weight_bits={effective_bits} "
                + f"and weight maximum absolute value {maxabs} "
                + f"(min step size {2**min_step_guaranteed:.5g} < "
                + f"resolution {2**min_step_needed:.5g}). "
                + "Consider increasing your learning rate, decreasing your "
                + "weight magnitudes, or increasing master_weight_bits. "
                + "Set check_numerics=False to disable this check. If "
                + "you're getting this error after loading a checkpoint, "
                + "you can call recompute_param_stats() throughout training "
                + "to fail faster and avoid this surprise."
            )

    @abc.abstractmethod
    def _do_step(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        errors: Optional[torch.Tensor],
        param_state: dict[str, Any],
        hparams: dict[str, Any],
    ) -> None:
        """Performs the optimizer step on a given parameter.

        Args:
            param: The parameter tensor to update.
            grad: The gradient of the parameter to use for the update.
            errors: Optional error correction bits for the parameter.
            state: The optimizer state for the parameter (as opposed to the
                optimizer state for the entire optimizer, which is self.state).
            hparams: Hyperparameters for the optimizer step, such as learning
                rate, betas, etc. These are taken from the parameter group.
        """
        ...


# ================================================================ LION


class FlashLion(FlashOptimizer):
    """FlashOptimizer implementation of the LION optimizer.

    This optimizer is a drop-in replacement for the regular LION optimizer
    with decoupled weight decay, but uses less memory, writes smaller
    checkpoints, and offers almost-numerically-identical convergence.

    You can load checkpoints from regular LION (as implemented in LLMFoundry*)
    with this class. Regular LION can load checkpoints from this class as
    long as they were not saved with `compress_state_dict=True`.

    See the LION paper (https://arxiv.org/abs/2302.06675) for details about
    the algorithm itself.

    See FlashOptimizer for details about the non-LION specific arguments
    and behavior of this class.

    * https://github.com/mosaicml/llm-foundry/blob/6c0d864916cf943314c47061e62efb381083e394/llmfoundry/optim/lion.py#L19
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0,
        decouple_lr: bool = False,
        **kwargs: Any,
    ):
        if not 0.0 <= betas[0] <= 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] <= 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        self._decouple_lr = decouple_lr
        defaults = {"betas": betas, "weight_decay": weight_decay}
        super().__init__(params=params, lr=lr, defaults=defaults, **kwargs)

    def _quantized_state_spec(self) -> dict[str, QuantizedTensorSpec]:
        return {"exp_avg": QuantizedTensorSpec(signed=True)}

    def _do_step(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        errors: Optional[torch.Tensor],
        param_state: dict[str, Any],
        hparams: dict[str, Any],
    ) -> None:
        # Decoupled weight decay: param *= 1 - wd * lr
        # With decouple_lr: param *= 1 - wd * (lr / initial_lr)
        weight_decay = hparams["weight_decay"]
        if self._decouple_lr:
            weight_decay *= hparams["lr"] / hparams["initial_lr"]
        else:
            weight_decay *= hparams["lr"]

        momentums = param_state["exp_avg"]
        beta1 = hparams["betas"][0]
        beta2 = hparams["betas"][1]
        lr = hparams["lr"]

        if errors is not None:
            if errors.dtype not in [torch.int8, torch.int16]:
                raise ValueError("Errors needs to be int8 or int16")

        if self._fused:
            is_quantized = momentums.is_quantized()
            _fused_momentum_step(
                momentum=momentums.kernel_tensor,
                scales_f16=momentums.kernel_scales_or_self,
                param=param,
                grad=grad,
                errors=errors,
                lr=lr,
                mom_coef=beta2,
                update_coef=beta1,
                weight_decay=weight_decay,
                do_lion=True,
                quantize_optim_states=is_quantized,
            )
            return

        # unfused LION step
        mom_f32 = momentums.materialize().to(dtype=torch.float32)
        grad_f32 = grad.to(dtype=torch.float32)
        update = mom_f32.lerp(grad_f32, 1 - beta1).sign_()

        param_f32 = _read_param_fp32(param, errors)

        if weight_decay > 0:
            param_f32.mul_(1.0 - weight_decay)
        param_f32.add_(update, alpha=-lr)

        param.copy_(param_f32.to(param.dtype))
        if errors is not None:
            errors.copy_(compute_ecc_bits(param_f32, param))

        mom_f32.lerp_(grad_f32, 1.0 - beta2)
        momentums.set_data(mom_f32)


# ================================================================ SGDM


class FlashSGD(FlashOptimizer):
    """Drop-in replacement for torch.optim.SGD with L2 regularisation.

    See torch.optim.SGD and FlashOptimizer for argument details.
    For decoupled weight decay, use :class:`FlashSGDW`.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.001,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        **kwargs: Any,
    ):
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        self._nesterov = nesterov
        self._decoupled = False
        self._decouple_lr = False
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
        }
        super().__init__(params, lr=lr, defaults=defaults, **kwargs)

    def _quantized_state_spec(self) -> dict[str, QuantizedTensorSpec]:
        use_momentum = any(g["momentum"] > 0.0 for g in self.param_groups)
        spec = QuantizedTensorSpec(signed=True)
        return {"momentum_buffer": spec} if use_momentum else {}

    def _do_step(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        errors: Optional[torch.Tensor],
        param_state: dict[str, Any],
        hparams: dict[str, Any],
    ) -> None:
        """Perform SGD update on a single parameter."""
        # Only check if grad is None. Don't check param.requires_grad because:
        # 1. param may be a local tensor from _get_local_tensor() whose
        #    requires_grad is False even when the original DTensor requires grad
        # 2. If grad is not None, we have a valid gradient to apply
        if grad is None:
            return

        # read hparams
        lr = hparams["lr"]
        mom_coef = hparams["momentum"]
        update_coef = 1.0 - hparams["dampening"]
        weight_decay = hparams["weight_decay"]
        decoupled = self._decoupled

        # Decoupled weight decay
        if decoupled:
            if self._decouple_lr:
                weight_decay *= lr / hparams["initial_lr"]
            else:
                weight_decay *= lr

        # No momentum -> no momentum buffer in state, so the fused Triton
        # kernel (which always reads/writes a momentum tensor) cannot be used.
        # We handle this case entirely in Python, including ECC.
        if mom_coef == 0.0:
            grad_f32 = grad.to(dtype=torch.float32)

            param_f32 = _read_param_fp32(param, errors)

            if not decoupled and weight_decay > 0:
                grad_f32.add_(param_f32, alpha=weight_decay)
            if decoupled and weight_decay > 0:
                param_f32.mul_(1 - weight_decay)
            param_f32.add_(grad_f32, alpha=-lr)

            param.copy_(param_f32.to(param.dtype))
            if errors is not None:
                errors.copy_(compute_ecc_bits(param_f32, param))
            return

        # if we get to here, we're using momentum
        mom = param_state["momentum_buffer"]

        # run the optimizer step with momentum
        if self._fused:
            is_quantized = mom.is_quantized()
            _fused_momentum_step(
                momentum=mom.kernel_tensor,
                scales_f16=mom.kernel_scales_or_self,
                param=param,
                grad=grad,
                errors=errors,
                lr=lr,
                mom_coef=mom_coef,
                update_coef=update_coef,
                weight_decay=weight_decay,
                decoupled=decoupled,
                nesterov=self._nesterov,
                quantize_optim_states=is_quantized,
            )
        else:
            grad_f32 = grad.to(dtype=torch.float32)

            param_f32 = _read_param_fp32(param, errors)

            if not decoupled and weight_decay > 0:
                grad_f32.add_(param_f32, alpha=weight_decay)

            mom_f32 = mom.materialize().to(dtype=torch.float32)
            mom_f32.mul_(mom_coef).add_(grad_f32, alpha=update_coef)
            if decoupled and weight_decay > 0:
                param_f32.mul_(1 - weight_decay)
            if self._nesterov:
                param_f32.add_(grad_f32 + mom_f32 * mom_coef, alpha=-lr)
            else:
                param_f32.add_(mom_f32, alpha=-lr)

            param.copy_(param_f32.to(param.dtype))
            if errors is not None:
                errors.copy_(compute_ecc_bits(param_f32, param))
            mom.set_data(mom_f32)


def _ecc_kernel_params(
    errors: torch.Tensor | None, param: torch.Tensor
) -> tuple[bool, int, torch.Tensor, "tl.constexpr"]:
    if errors is None:
        return False, 0, param, tl.int8
    elem_sz = errors.element_size()
    if elem_sz == 1:
        return True, 127, errors, tl.int8
    if elem_sz == 2:
        return True, 32767, errors, tl.int16
    raise ValueError(f"Errors must have width 1 or 2, not {elem_sz}.")


def _read_param_fp32(param: torch.Tensor, errors: torch.Tensor | None) -> torch.Tensor:
    if errors is not None:
        return reconstruct_fp32_param(param, errors)
    return param.to(dtype=torch.float32)


def _fused_momentum_step(
    momentum: torch.Tensor,
    scales_f16: torch.Tensor,
    param: torch.Tensor,
    grad: torch.Tensor,
    errors: Optional[torch.Tensor],
    lr: float,
    mom_coef: float,
    update_coef: float,
    weight_decay: float,
    decoupled: bool = False,
    nesterov: bool = False,
    do_lion: bool = False,
    quantize_optim_states: bool = True,
    group_size: int = 32,
) -> None:
    N = param.numel()
    if N == 0:
        return
    use_ecc, signed_max_val, errors, signed_error_t = _ecc_kernel_params(errors, param)

    grid = functools.partial(_make_grid, N)
    _triton_momentum_kernel[grid](
        momentum,
        scales_f16,
        param,
        grad,
        errors,
        N,
        lr,
        mom_coef,
        update_coef,
        weight_decay,
        GROUP_SIZE=group_size,
        DO_LION=do_lion,
        NESTEROV=nesterov,
        DECOUPLED_WEIGHT_DECAY=decoupled,
        PARAM_DTYPE=_TORCH_DTYPE_TO_TRITON_DTYPE[param.dtype],
        USE_ECC=use_ecc,
        QUANTIZE_OPTIM_STATES=quantize_optim_states,
        SIGNED_ERROR_T=signed_error_t,
        NUM_MANTISSA_BITS=_NUM_MANTISSA_BITS[param.dtype],
        SIGNED_MAX_VAL=signed_max_val,
        BLOCK_SIZE_N=1024,
    )


# NOTE: no @triton.autotune here. Autotune re-runs the kernel with different
# configs, but this kernel does in-place read-modify-write on params/momentum/ECC,
# so repeated runs would apply the update multiple times and corrupt state.
# BLOCK_SIZE_N is hardcoded at the call site instead.
@triton.jit
def _triton_momentum_kernel(
    mom_ptr: "Any",  # pointer to momentum state (int8 if quantized, fp32/bf16/fp16 otherwise)
    scales_ptr: "Any",  # pointer to scales for quantization/dequantization (only used if QUANTIZE_OPTIM_STATES)
    param_ptr: "Any",  # pointer to parameter array
    grad_ptr: "Any",  # pointer to bf16 update array
    ecc_ptr: "Any",  # pointer to error correction bits (optional)
    N: int,  # total number of elements
    lr: float,  # learning rate scalar
    mom_coef: float,  # momentum persistence (SGD: momentum; LION: beta2)
    update_coef: float,  # update mixing coefficient (SGD: 1-dampening; LION: beta1)
    weight_decay: float,  # weight decay scalar
    GROUP_SIZE: tl.constexpr,  # number of elements per quantization group
    DO_LION: tl.constexpr,  # whether to use LION update logic
    NESTEROV: tl.constexpr,  # whether to use Nesterov momentum
    DECOUPLED_WEIGHT_DECAY: tl.constexpr,  # whether to use decoupled weight decay
    PARAM_DTYPE: tl.constexpr,  # dtype of the parameter
    USE_ECC: tl.constexpr,  # whether to use error correction bits
    QUANTIZE_OPTIM_STATES: tl.constexpr,  # whether optimizer states are quantized to int8
    SIGNED_ERROR_T: tl.constexpr,  # signed error type for ECC
    NUM_MANTISSA_BITS: tl.constexpr,  # mantissa bits in narrow dtype
    SIGNED_MAX_VAL: tl.constexpr,  # max value for signed quantization
    BLOCK_SIZE_N: tl.constexpr,  # numel processed per block per iteration
):
    # --- Grid Stride Loop Setup ---
    pid_block = tl.program_id(axis=0)
    num_blocks_launched = tl.num_programs(axis=0)
    total_num_blocks = tl.cdiv(N, BLOCK_SIZE_N)
    num_groups_per_block: tl.constexpr = BLOCK_SIZE_N // GROUP_SIZE

    # --- Process each `BLOCK_SIZE_N` chunk ---
    for block_idx in range(pid_block, total_num_blocks, num_blocks_launched):
        # Compute offsets for data in this block
        block_base_offset = block_idx * BLOCK_SIZE_N
        absolute_offsets = block_base_offset + tl.arange(0, BLOCK_SIZE_N)
        scales_base_offset = block_base_offset // GROUP_SIZE
        scales_offsets = scales_base_offset + tl.arange(0, num_groups_per_block)

        # Load gradients, momentums, and params
        mask = absolute_offsets < N
        grad = tl.load(grad_ptr + absolute_offsets, mask=mask, other=0.0)
        grad = grad.to(tl.float32)
        param = tl.load(param_ptr + absolute_offsets, mask=mask, other=0.0)
        if USE_ECC:
            ecc = tl.load(ecc_ptr + absolute_offsets, mask=mask, other=0.0)
            param = _apply_error_correction(
                param,
                ecc,
                NUM_MANTISSA_BITS=NUM_MANTISSA_BITS,
                SIGNED_MAX_VAL=SIGNED_MAX_VAL,
            )
        else:
            param = param.to(tl.float32)
        if not DO_LION and not DECOUPLED_WEIGHT_DECAY:  # coupled weight decay
            grad += param * weight_decay

        # Load and dequantize (if needed) the momentum
        if QUANTIZE_OPTIM_STATES:
            mom_i8 = tl.load(mom_ptr + absolute_offsets, mask=mask, other=0)
            scales_mask = scales_offsets < tl.cdiv(N, GROUP_SIZE)
            scales = tl.load(scales_ptr + scales_offsets, mask=scales_mask, other=1.0)
            # Dequantize the momentum (dequant -> inverse_softsign -> scale)
            mom_f32 = mom_i8.to(tl.float32)
            mom_groups = mom_f32.reshape((num_groups_per_block, GROUP_SIZE))
            # 1. q / 127 = transformed value in [-1, 1]
            transformed = mom_groups / 127.0
            # 2. Inverse softsign: x = y / (2 - |y|)
            normalized = transformed / (2.0 - tl.abs(transformed))
            # 3. Multiply by scale
            mom_f32 = (normalized * scales.to(tl.float32)[:, None]).reshape(
                (BLOCK_SIZE_N,)
            )
        else:
            # Load momentum directly in full precision
            mom_f32 = tl.load(mom_ptr + absolute_offsets, mask=mask, other=0.0)
            mom_f32 = mom_f32.to(tl.float32)

        # Update the momentum
        if DO_LION:
            # update_coef is beta1 (update interpolation); mom_coef is beta2 (momentum EMA)
            # Compute param update before mom update
            update = (mom_f32 * update_coef) + (grad * (1.0 - update_coef))
            update = tl.where(update > 0, 1.0, tl.where(update < 0, -1.0, 0.0))
            mom_f32 = (mom_f32 * mom_coef) + (grad * (1.0 - mom_coef))
        else:
            mom_f32 = mom_f32 * mom_coef + grad * update_coef

        # Update + store the param
        if DO_LION:  # decoupled weight decay happens here
            param *= 1.0 - weight_decay
            param -= update * lr
        else:
            if DECOUPLED_WEIGHT_DECAY:  # decoupled weight decay for SGD
                param *= 1.0 - weight_decay
            if NESTEROV:
                param -= (grad + mom_coef * mom_f32) * lr
            else:
                param -= mom_f32 * lr
        param_narrow = param.to(PARAM_DTYPE)
        tl.store(param_ptr + absolute_offsets, param_narrow, mask=mask)
        if USE_ECC:
            ecc = _compute_ecc_bits(
                param,
                param_narrow,
                SIGNED_ERROR_T=SIGNED_ERROR_T,
                NUM_MANTISSA_BITS=NUM_MANTISSA_BITS,
                SIGNED_MAX_VAL=SIGNED_MAX_VAL,
            )
            tl.store(ecc_ptr + absolute_offsets, ecc, mask=mask)

        # Quantize (if needed) and store the new momentum
        if QUANTIZE_OPTIM_STATES:
            # Quantize the momentum (absmax -> normalize -> softsign -> quantize)
            mom_groups = mom_f32.reshape((num_groups_per_block, GROUP_SIZE))
            # 1. Absmax on raw values — safe for partial groups (N % BLOCK_SIZE_N != 0)
            # because zero-padded tail elements can't increase the max.
            absmaxs = tl.max(tl.abs(mom_groups), axis=1)
            absmaxs = tl.maximum(absmaxs, 1e-12)  # avoid div by zero
            # 2. Normalize to [-1, 1]
            normalized = mom_groups / absmaxs[:, None]
            # 3. Softsign: y = 2x / (1 + |x|)
            transformed = 2.0 * normalized / (1.0 + tl.abs(normalized))
            # 4. Quantize
            mom_out = (transformed * 127.0).reshape((BLOCK_SIZE_N,))
            mom_out_i8 = tl.floor(mom_out + 0.5).to(tl.int8)
            # Store updated quantized values and scales
            tl.store(mom_ptr + absolute_offsets, mom_out_i8, mask=mask)
            scales_mask = scales_offsets < tl.cdiv(N, GROUP_SIZE)
            tl.store(
                scales_ptr + scales_offsets, absmaxs.to(tl.float16), mask=scales_mask
            )
        else:
            # Store momentum directly in full precision
            tl.store(mom_ptr + absolute_offsets, mom_f32.to(PARAM_DTYPE), mask=mask)


class FlashSGDW(FlashSGD):
    """SGD with decoupled weight decay.

    Like :class:`FlashSGD` but applies weight decay as a multiplicative
    factor on the parameters rather than as L2 regularisation on the gradient.

    By default uses standard decoupled weight decay (scaled by ``lr``).
    Set ``decouple_lr=True`` for fully LR-decoupled weight decay
    (scaled by ``lr_t / lr_initial``).
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.001,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        decouple_lr: bool = False,
        quantize: bool = True,
        compress_state_dict: bool = False,
        master_weight_bits: Literal[24, 32] | None = 24,
        check_numerics: bool = False,
        fused: bool = True,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            quantize=quantize,
            compress_state_dict=compress_state_dict,
            master_weight_bits=master_weight_bits,
            check_numerics=check_numerics,
            fused=fused,
        )
        self._decoupled = True
        self._decouple_lr = decouple_lr


# ================================================================ Adam


class FlashAdam(FlashOptimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        **kwargs: Any,
    ):
        """Adam optimizer with L2 regularisation (coupled weight decay).

        Uses ``grad += weight_decay * param`` like ``torch.optim.Adam``.
        For decoupled weight decay (AdamW), use :class:`FlashAdamW`.

        Args:
            weight_decay: L2 regularisation coefficient (default 0).
        """
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] <= 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] <= 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        self._decoupled = False
        self._decouple_lr = False
        defaults = {
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params=params, lr=lr, defaults=defaults, **kwargs)

    def _ensure_state_initialized(
        self, p: torch.Tensor, hparams: dict[str, Any]
    ) -> None:
        super()._ensure_state_initialized(p, hparams)
        state = self.state[p]
        # note: pytorch optimizers use float32 for checkpointed params
        if "step" not in state:
            state["step"] = torch.tensor(0, dtype=torch.float32, device="cpu")

    def _min_step_size_relative_to_lr(self) -> float:
        # Returns a pragmatic heuristic for the minimum effective step size relative to the
        # learning rate.

        # Unlike SGD or Lion where the step magnitude is strictly `1.0 * lr`, Adam's step
        # is element-wise and scaled by the ratio of the first moment to the square root of
        # the second moment (|m| / sqrt(v)). Because this ratio can theoretically be arbitrarily
        # close to zero, there is no true mathematical lower bound.

        # We return 1e-1 as a safety buffer. This forces the numerics check to
        # assume Adam will routinely scale updates down to a fraction of the base LR. It acts
        # as a strict safeguard to warn users before silent underflow occurs on the largest
        # weights, without requiring expensive per-element analysis.
        return 0.1  # or 0.01, depending on how strict you want to be

    def _quantized_state_spec(self) -> dict[str, QuantizedTensorSpec]:
        return {
            "exp_avg": QuantizedTensorSpec(signed=True, softsign=True),
            "exp_avg_sq": QuantizedTensorSpec(signed=False, sqrt=True, softsign=False),
        }

    def _do_step(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        errors: Optional[torch.Tensor],
        param_state: dict[str, Any],
        hparams: dict[str, Any],
    ) -> None:
        # read hparams
        lr = hparams["lr"]
        beta1 = hparams["betas"][0]
        beta2 = hparams["betas"][1]
        eps = hparams["eps"]
        weight_decay = hparams["weight_decay"]
        decoupled = self._decoupled

        # Decoupled weight decay: param *= (1 - wd * lr)
        # With decouple_lr: param *= (1 - wd * lr/initial_lr)
        if decoupled:
            if self._decouple_lr:
                weight_decay *= hparams["lr"] / hparams["initial_lr"]
            else:
                weight_decay *= hparams["lr"]

        if self._fused:
            exp_avg = param_state["exp_avg"]
            exp_avg_sq = param_state["exp_avg_sq"]
            is_quantized = exp_avg.is_quantized()

            # Increment step counter (matches unfused implementation)
            param_state["step"] += 1

            return _fused_adam_step(
                mom=exp_avg.kernel_tensor,
                mom_scales_f16=exp_avg.kernel_scales_or_self,
                var=exp_avg_sq.kernel_tensor,
                var_scales_f16=exp_avg_sq.kernel_scales_or_self,
                param=param,
                grad=grad,
                errors=errors,
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                step=int(param_state["step"].item()),
                weight_decay=weight_decay,
                decoupled=decoupled,
                quantize_optim_states=is_quantized,
            )

        # ------------------------ unfused Adam step
        # Get state variables
        exp_avg = param_state["exp_avg"]
        exp_avg_sq = param_state["exp_avg_sq"]

        # step is a CPU tensor (for checkpointing); 0-dim CPU tensors don't
        # cause a device sync. The fused path converts to int for Triton.
        param_state["step"] += 1
        step = param_state["step"]

        # Materialize quantized tensors to float32
        exp_avg_f32 = exp_avg.materialize().to(dtype=torch.float32)
        exp_avg_sq_f32 = exp_avg_sq.materialize().to(dtype=torch.float32)
        grad_f32 = grad.to(dtype=torch.float32)

        param_f32 = _read_param_fp32(param, errors)

        # Apply weight decay to gradient if not decoupled
        if not decoupled and weight_decay > 0:
            grad_f32 = grad_f32.add(param_f32, alpha=weight_decay)

        # Update biased first moment estimate
        exp_avg_f32.mul_(beta1).add_(grad_f32, alpha=1 - beta1)

        # Update biased second raw moment estimate
        exp_avg_sq_f32.mul_(beta2).addcmul_(grad_f32, grad_f32, value=1 - beta2)

        # Apply bias correction
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        corrected_exp_avg = exp_avg_f32 / bias_correction1
        corrected_exp_avg_sq = exp_avg_sq_f32 / bias_correction2

        # Apply decoupled weight decay before parameter update
        if decoupled and weight_decay > 0:
            param_f32.mul_(1 - weight_decay)

        # Update parameter in fp32
        denom = corrected_exp_avg_sq.sqrt().add_(eps)
        param_f32.addcdiv_(corrected_exp_avg, denom, value=-lr)

        # Write back
        param.copy_(param_f32.to(param.dtype))
        if errors is not None:
            errors.copy_(compute_ecc_bits(param_f32, param))

        # Update state tensors
        exp_avg.set_data(exp_avg_f32)
        exp_avg_sq.set_data(exp_avg_sq_f32)


class FlashAdamW(FlashAdam):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        decouple_lr: bool = False,
        quantize: bool = True,
        compress_state_dict: bool = False,
        master_weight_bits: Literal[24, 32] | None = 24,
        check_numerics: bool = False,
        fused: bool = True,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            quantize=quantize,
            compress_state_dict=compress_state_dict,
            master_weight_bits=master_weight_bits,
            check_numerics=check_numerics,
            fused=fused,
        )
        self._decoupled = True
        self._decouple_lr = decouple_lr


def _matches_keyword(name: str, keywords: list[str]) -> bool:
    segments = name.split(".")
    return any(keyword in segments for keyword in keywords)


def _fp32_input_cast_hook(
    module: nn.Module,
    args: tuple[Any, ...],
) -> tuple[Any, ...]:
    return tuple(
        a.to(torch.float32)
        if isinstance(a, torch.Tensor) and a.is_floating_point()
        else a
        for a in args
    )


def cast_model(
    model: nn.Module,
    dtype: torch.dtype = torch.bfloat16,
    selective: bool = True,
    full_precision_keywords: list[str] | None = None,
) -> None:
    """Cast model parameters and buffers to a different dtype.

    Typically used before training to reduce memory by storing parameters
    in bf16 instead of fp32. When combined with master_weight_bits, the
    optimizer maintains higher effective precision via error-correction bits.

    Args:
        model: The model to convert.
        dtype: Target dtype (default: torch.bfloat16).
        selective: If True (default), normalization layers with running statistics
            (BatchNorm, InstanceNorm) are kept in fp32 for stability.
        full_precision_keywords: Module/parameter names to keep in fp32
            (e.g., ["lm_head", "classifier"]).
    """
    # Build set of full precision parameter names if keywords provided
    full_precision_params = set()
    if full_precision_keywords:
        for name, _ in model.named_parameters():
            if _matches_keyword(name, full_precision_keywords):
                full_precision_params.add(name)

    # Get mapping of parameter/buffer to its full name
    param_to_name = {}
    for name, param in model.named_parameters():
        param_to_name[param] = name
    buffer_to_name = {}
    for name, buffer in model.named_buffers():
        buffer_to_name[buffer] = name

    # Track already-cast tensors so shared/tied parameters stay shared
    seen: dict[int, torch.Tensor] = {}

    for module in model.modules():
        # Keep norm layers with running statistics in fp32 if selective.
        # LayerNorm / RMSNorm / GroupNorm are safe in bf16 because PyTorch's
        # fused CUDA kernels accumulate reductions in fp32 internally.
        if selective and isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.SyncBatchNorm,
                nn.InstanceNorm1d,
                nn.InstanceNorm2d,
                nn.InstanceNorm3d,
            ),
        ):
            continue

        # Convert parameters and buffers to target dtype
        for param in module.parameters(recurse=False):
            # Skip if this parameter is marked for full precision
            param_name = param_to_name.get(param, "")
            if param_name in full_precision_params:
                continue
            data_id = param.data.data_ptr()
            if data_id not in seen:
                seen[data_id] = param.data.to(dtype=dtype)
            param.data = seen[data_id]

        for buffer in module.buffers(recurse=False):
            # Skip if this buffer belongs to a full precision module
            buffer_name = buffer_to_name.get(buffer, "")
            if _matches_keyword(buffer_name, full_precision_keywords or []):
                continue
            data_id = buffer.data.data_ptr()
            if data_id not in seen:
                seen[data_id] = buffer.data.to(dtype=dtype)
            buffer.data = seen[data_id]

    # Register pre-hook on full precision modules to cast inputs to fp32
    if full_precision_keywords:
        for name, module in model.named_modules():
            if _matches_keyword(name, full_precision_keywords):
                if not getattr(module, "_has_fp32_input_hook", False):
                    module.register_forward_pre_hook(_fp32_input_cast_hook)
                    module._has_fp32_input_hook = True


def _fused_adam_step(
    mom: torch.Tensor,
    mom_scales_f16: torch.Tensor,
    var: torch.Tensor,
    var_scales_f16: torch.Tensor,
    param: torch.Tensor,
    grad: torch.Tensor,
    errors: Optional[torch.Tensor],
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    decoupled: bool,
    step: int,
    quantize_optim_states: bool = True,
    group_size: int = 32,
) -> None:
    N = param.numel()
    if N == 0:
        return
    use_ecc, signed_max_val, errors, signed_error_t = _ecc_kernel_params(errors, param)

    grid = functools.partial(_make_grid, N)
    _triton_adam_kernel[grid](
        mom,
        mom_scales_f16,
        var,
        var_scales_f16,
        param,
        grad,
        errors,
        N,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        step,
        GROUP_SIZE=group_size,
        DECOUPLED_WEIGHT_DECAY=decoupled,
        PARAM_DTYPE=_TORCH_DTYPE_TO_TRITON_DTYPE[param.dtype],
        USE_ECC=use_ecc,
        QUANTIZE_OPTIM_STATES=quantize_optim_states,
        SIGNED_ERROR_T=signed_error_t,
        NUM_MANTISSA_BITS=_NUM_MANTISSA_BITS[param.dtype],
        SIGNED_MAX_VAL=signed_max_val,
        BLOCK_SIZE_N=1024,
    )


# NOTE: no @triton.autotune - see comment on _triton_momentum_kernel.
@triton.jit
def _triton_adam_kernel(
    mom_ptr: "Any",  # pointer to momentum state (int8 if quantized, fp32/bf16/fp16 otherwise)
    mom_scales_f16_ptr: "Any",  # pointer to scales for quantization/dequantization (only used if QUANTIZE_OPTIM_STATES)
    var_ptr: "Any",  # pointer to variance state (uint8 if quantized, fp32/bf16/fp16 otherwise)
    var_scales_f16_ptr: "Any",  # pointer to variance estimate scales (only used if QUANTIZE_OPTIM_STATES)
    param_ptr: "Any",  # pointer to parameter array
    grad_ptr: "Any",  # pointer to bf16 update array
    ecc_ptr: "Any",  # pointer to error correction bits (optional)
    N: int,  # total number of elements
    lr: float,  # learning rate scalar
    beta1: float,  # momentum/EMA parameter
    beta2: float,  # variance EMA parameter
    eps: float,  # epsilon for numerical stability
    weight_decay: float,  # weight decay scalar
    step: int,  # current step number for bias correction
    GROUP_SIZE: tl.constexpr,  # number of elements per quantization group
    DECOUPLED_WEIGHT_DECAY: tl.constexpr,  # bool for decoupled weight decay
    PARAM_DTYPE: tl.constexpr,  # dtype of the parameter
    USE_ECC: tl.constexpr,  # whether to use error correction bits
    QUANTIZE_OPTIM_STATES: tl.constexpr,  # whether optimizer states are quantized to int8
    SIGNED_ERROR_T: tl.constexpr,  # signed error type for ECC
    NUM_MANTISSA_BITS: tl.constexpr,  # mantissa bits in narrow dtype
    SIGNED_MAX_VAL: tl.constexpr,  # max value for signed quantization
    BLOCK_SIZE_N: tl.constexpr,  # numel processed per block per iteration
):
    # --- Grid Stride Loop Setup ---
    pid_block = tl.program_id(axis=0)
    num_blocks_launched = tl.num_programs(axis=0)
    total_num_blocks = tl.cdiv(N, BLOCK_SIZE_N)
    num_groups_per_block: tl.constexpr = BLOCK_SIZE_N // GROUP_SIZE

    # --- Process each `BLOCK_SIZE_N` chunk ---
    for block_idx in range(pid_block, total_num_blocks, num_blocks_launched):
        # Compute offsets for data in this block
        block_base_offset = block_idx * BLOCK_SIZE_N
        absolute_offsets = block_base_offset + tl.arange(0, BLOCK_SIZE_N)
        scales_base_offset = block_base_offset // GROUP_SIZE
        scales_offsets = scales_base_offset + tl.arange(0, num_groups_per_block)

        # Load gradients and params
        mask = absolute_offsets < N
        grad = tl.load(grad_ptr + absolute_offsets, mask=mask, other=0.0)
        grad = grad.to(tl.float32)
        param = tl.load(param_ptr + absolute_offsets, mask=mask, other=0.0)
        if USE_ECC:
            ecc = tl.load(ecc_ptr + absolute_offsets, mask=mask, other=0.0)
            param = _apply_error_correction(
                param,
                ecc,
                NUM_MANTISSA_BITS=NUM_MANTISSA_BITS,
                SIGNED_MAX_VAL=SIGNED_MAX_VAL,
            )
        else:
            param = param.to(tl.float32)
        if not DECOUPLED_WEIGHT_DECAY:
            grad += param * weight_decay

        # Load and dequantize (if needed) the momentum and variance estimates
        if QUANTIZE_OPTIM_STATES:
            mom_i8 = tl.load(mom_ptr + absolute_offsets, mask=mask, other=0)
            var_i8 = tl.load(var_ptr + absolute_offsets, mask=mask, other=0)
            scales_mask = scales_offsets < tl.cdiv(N, GROUP_SIZE)
            mom_scales = tl.load(
                mom_scales_f16_ptr + scales_offsets, mask=scales_mask, other=1.0
            )
            var_scales = tl.load(
                var_scales_f16_ptr + scales_offsets, mask=scales_mask, other=1.0
            )

            mom_f32 = mom_i8.to(tl.float32)
            var_f32 = var_i8.to(tl.float32)
            mom_groups = mom_f32.reshape((num_groups_per_block, GROUP_SIZE))
            var_groups = var_f32.reshape((num_groups_per_block, GROUP_SIZE))

            # Dequant -> inverse_softsign (momentum only) -> scale
            # 1. q / IN_MAX = transformed
            mom_transformed = mom_groups / 127.0
            var_transformed = var_groups / 255.0

            # 2. Inverse softsign: x = y / (2 - |y|) - applied to momentum only
            mom_normalized = mom_transformed / (2.0 - tl.abs(mom_transformed))
            # Variance does not use softsign, so use transformed directly
            var_normalized = var_transformed

            # 3. Multiply by scale
            mom_f32 = (mom_normalized * mom_scales.to(tl.float32)[:, None]).reshape(
                (BLOCK_SIZE_N,)
            )
            var_sqrt = (var_normalized * var_scales.to(tl.float32)[:, None]).reshape(
                (BLOCK_SIZE_N,)
            )
            # Square to undo the sqrt applied during quantization
            var_f32 = var_sqrt * var_sqrt
        else:
            # Load momentum and variance directly in full precision
            mom_f32 = tl.load(mom_ptr + absolute_offsets, mask=mask, other=0.0)
            var_f32 = tl.load(var_ptr + absolute_offsets, mask=mask, other=0.0)
            mom_f32 = mom_f32.to(tl.float32)
            var_f32 = var_f32.to(tl.float32)

        # Update the momentum
        mom_f32 = mom_f32 * beta1 + grad * (1.0 - beta1)

        # Update the variance estimate
        var_f32 = var_f32 * beta2 + grad * grad * (1.0 - beta2)

        # Update + store the param
        if DECOUPLED_WEIGHT_DECAY:  # decoupled weight decay happens here
            param *= 1.0 - weight_decay

        # Apply bias correction to momentum and variance estimates.
        # Loop-invariant, but precomputing CPU-side regresses ~2% (different Triton codegen).
        bias_correction1 = 1.0 - libdevice.pow(beta1, step)
        bias_correction2 = 1.0 - libdevice.pow(beta2, step)
        corrected_mom = mom_f32 / bias_correction1
        corrected_var = var_f32 / bias_correction2

        # Adam update: param -= lr * corrected_mom / (sqrt(corrected_var) + eps)
        denom = tl.sqrt(corrected_var) + eps
        param -= lr * corrected_mom / denom

        param_narrow = param.to(PARAM_DTYPE)
        tl.store(param_ptr + absolute_offsets, param_narrow, mask=mask)
        if USE_ECC:
            ecc = _compute_ecc_bits(
                param,
                param_narrow,
                SIGNED_ERROR_T=SIGNED_ERROR_T,
                NUM_MANTISSA_BITS=NUM_MANTISSA_BITS,
                SIGNED_MAX_VAL=SIGNED_MAX_VAL,
            )
            tl.store(ecc_ptr + absolute_offsets, ecc, mask=mask)

        # Quantize (if needed) and store the new momentum and variance
        if QUANTIZE_OPTIM_STATES:
            mom_groups = mom_f32.reshape((num_groups_per_block, GROUP_SIZE))
            # Apply sqrt before rescaling and softsign for variance
            var_sqrt = tl.sqrt(var_f32)
            var_groups = var_sqrt.reshape((num_groups_per_block, GROUP_SIZE))

            # Absmax -> scale -> transform
            # 1. Absmax on RAW values — safe for partial groups (N % BLOCK_SIZE_N != 0)
            # because zero-padded tail elements can't increase the max.
            mom_absmaxs = tl.max(tl.abs(mom_groups), axis=1)
            var_absmaxs = tl.max(
                var_groups, axis=1
            )  # variance always positive (sqrt applied)
            mom_absmaxs = tl.maximum(mom_absmaxs, 1e-12)
            var_absmaxs = tl.maximum(var_absmaxs, 1e-12)

            # 2. Normalize
            mom_normalized = mom_groups / mom_absmaxs[:, None]
            var_normalized = var_groups / var_absmaxs[:, None]

            # 3. Softsign: y = 2x / (1 + |x|) - applied to momentum only
            mom_transformed = 2.0 * mom_normalized / (1.0 + tl.abs(mom_normalized))
            # Variance does not use softsign, so use normalized directly
            var_transformed = var_normalized

            # 4. Quantize
            mom_out = (mom_transformed * 127.0).reshape((BLOCK_SIZE_N,))
            var_out = (var_transformed * 255.0).reshape((BLOCK_SIZE_N,))
            mom_out_i8 = tl.floor(mom_out + 0.5).to(tl.int8)
            var_out_i8 = tl.floor(var_out + 0.5).to(tl.uint8)

            # Store
            tl.store(mom_ptr + absolute_offsets, mom_out_i8, mask=mask)
            tl.store(var_ptr + absolute_offsets, var_out_i8, mask=mask)
            scales_mask = scales_offsets < tl.cdiv(N, GROUP_SIZE)
            tl.store(
                mom_scales_f16_ptr + scales_offsets,
                mom_absmaxs.to(tl.float16),
                mask=scales_mask,
            )
            tl.store(
                var_scales_f16_ptr + scales_offsets,
                var_absmaxs.to(tl.float16),
                mask=scales_mask,
            )
        else:
            # Store momentum and variance directly in full precision
            tl.store(mom_ptr + absolute_offsets, mom_f32.to(PARAM_DTYPE), mask=mask)
            tl.store(var_ptr + absolute_offsets, var_f32.to(PARAM_DTYPE), mask=mask)


# =========================================================== Triton + Wrappers

# ------------------------------------------------ Quantization


def quantize(
    x_float: torch.Tensor,
    out_int8: Optional[torch.Tensor] = None,
    out_scales: Optional[torch.Tensor] = None,
    signed: bool = True,
    group_size: int = 32,
    sqrt: bool = False,
    softsign: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    N = x_float.numel()
    out_type = tl.int8 if signed else tl.uint8
    out_max = 127.0 if signed else 255.0
    if out_int8 is None:
        out_torch_dtype = torch.int8 if signed else torch.uint8
        out_int8 = torch.empty_like(x_float, dtype=out_torch_dtype)
    if out_scales is None:
        num_scales = (N + group_size - 1) // group_size
        out_scales = torch.empty(num_scales, device=x_float.device, dtype=torch.float16)
    if N > 0:
        grid = functools.partial(_make_grid, N)
        _triton_quantize_kernel[grid](
            x_float,
            out_int8,
            out_scales,
            N,
            out_max,
            out_type,
            group_size,
            sqrt,
            softsign,
        )
    return out_int8, out_scales


def dequantize(
    in_int8: torch.Tensor,
    in_scales: torch.Tensor,
    out_float: Optional[torch.Tensor] = None,
    signed: bool = True,
    group_size: int = 32,
    sqrt: bool = False,
    softsign: bool = True,
) -> torch.Tensor:
    N = in_int8.numel()
    in_max = 127.0 if signed else 255.0
    if out_float is None:
        out_float = torch.empty_like(in_int8, dtype=torch.float32)
    if N > 0:
        grid = functools.partial(_make_grid, N)
        _triton_dequantize_kernel[grid](
            in_int8, in_scales, out_float, N, in_max, group_size, sqrt, softsign
        )
    return out_float


@triton.autotune(configs=_generate_configs(), key=["N"])
@triton.jit
def _triton_dequantize_kernel(
    q_in_ptr: "Any",  # pointer to int8 input
    scales_in_ptr: "Any",  # pointer to input scales
    x_out_ptr: "Any",  # pointer to float32 output
    N: int,  # total number of elements
    IN_MAX: tl.constexpr,  # max value for quantization (int8)
    GROUP_SIZE: tl.constexpr,  # passed as a compile-time constant
    SQRT: tl.constexpr,  # whether to square the output (undo sqrt from quantization)
    SOFTSIGN: tl.constexpr,  # whether to apply inverse softsign
    BLOCK_SIZE_N: tl.constexpr,  # numel processed by a program in one iteration
):
    # --- Grid Stride Loop Setup ---
    pid_block = tl.program_id(axis=0)
    num_blocks_launched = tl.num_programs(axis=0)
    total_num_blocks = tl.cdiv(N, BLOCK_SIZE_N)

    # Each program instance (block) iterates through its assigned blocks
    for block_idx in range(pid_block, total_num_blocks, num_blocks_launched):
        # --- Process one `BLOCK_SIZE_N` chunk ---
        block_base_offset = block_idx * BLOCK_SIZE_N
        absolute_offsets = block_base_offset + tl.arange(0, BLOCK_SIZE_N)
        scales_base_offset = block_base_offset // GROUP_SIZE
        scales_offsets = scales_base_offset + tl.arange(0, BLOCK_SIZE_N // GROUP_SIZE)

        # Load quantized data and scales
        mask = absolute_offsets < N
        q_i8 = tl.load(
            q_in_ptr + absolute_offsets,
            mask=mask,
            other=0,
            eviction_policy="evict_first",
        )

        scales_mask = scales_offsets < tl.cdiv(N, GROUP_SIZE)
        scales = tl.load(
            scales_in_ptr + scales_offsets,
            mask=scales_mask,
            other=1.0,
            eviction_policy="evict_first",
        )

        q_f32 = q_i8.to(tl.float32)
        num_groups_per_block: tl.constexpr = BLOCK_SIZE_N // GROUP_SIZE
        q_groups = q_f32.reshape((num_groups_per_block, GROUP_SIZE))
        scales_broadcast = scales[:, None]

        # Dequant -> inverse_softsign (if enabled) -> scale -> square (if sqrt was applied)
        # 1. q / IN_MAX = transformed value in [-1, 1]
        transformed = q_groups / IN_MAX

        # 2. Inverse softsign: x = y / (2 - |y|) (if enabled)
        if SOFTSIGN:
            normalized = transformed / (2.0 - tl.abs(transformed))
        else:
            normalized = transformed

        # 3. Multiply by scale
        x_f32 = (normalized * scales_broadcast).reshape((BLOCK_SIZE_N,))

        # 4. Square to undo the sqrt applied during quantization (if enabled)
        if SQRT:
            x_f32 = x_f32 * x_f32

        # Store result
        tl.store(x_out_ptr + absolute_offsets, x_f32, mask=mask)


@triton.autotune(configs=_generate_configs(), key=["N"])
@triton.jit
def _triton_quantize_kernel(
    in_ptr: "Any",  # pointer to bf16 input
    q_out_ptr: "Any",  # pointer to int8 output
    scales_out_ptr: "Any",  # pointer to output scales
    N: int,  # total number of elements
    OUT_MAX: tl.constexpr,  # max value for quantization (int8)
    OUT_DTYPE: tl.constexpr,  # output dtype (int8)
    GROUP_SIZE: tl.constexpr,  # passed as a compile-time constant
    SQRT: tl.constexpr,  # whether to apply sqrt before quantization
    SOFTSIGN: tl.constexpr,  # whether to apply softsign transformation
    BLOCK_SIZE_N: tl.constexpr,  # numel processed by a block in one iteration
):
    # --- Grid Stride Loop Setup ---
    pid_block = tl.program_id(axis=0)
    num_blocks_launched = tl.num_programs(axis=0)
    total_num_blocks = tl.cdiv(N, BLOCK_SIZE_N)
    num_groups_per_block: tl.constexpr = BLOCK_SIZE_N // GROUP_SIZE

    # Each program instance (block) iterates through its assigned conceptual blocks
    # The loop iterates from `pid_block` with a step of `num_blocks_launched`
    for block_idx in range(pid_block, total_num_blocks, num_blocks_launched):
        # --- Process one `BLOCK_SIZE_N` chunk ---
        block_base_offset = block_idx * BLOCK_SIZE_N
        absolute_offsets = block_base_offset + tl.arange(0, BLOCK_SIZE_N)
        scales_base_offset = block_base_offset // GROUP_SIZE
        scales_offsets = scales_base_offset + tl.arange(0, num_groups_per_block)

        # Load data
        mask = absolute_offsets < N
        x_bf16 = tl.load(
            in_ptr + absolute_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        x_f32 = x_bf16.to(tl.float32)

        # Apply sqrt before rescaling and softsign (if enabled)
        if SQRT:
            x_f32 = tl.sqrt(x_f32)

        # Absmax -> scale -> transform
        # 1. Absmax on RAW values (after sqrt if enabled) — safe for partial groups
        # (N % BLOCK_SIZE_N != 0) because zero-padded tail elements can't increase the max.
        groups = x_f32.reshape((num_groups_per_block, GROUP_SIZE))
        absmaxs = tl.max(tl.abs(groups), axis=1)
        absmaxs = tl.maximum(absmaxs, 1e-12)

        # 2. Normalize to [-1, 1]
        normalized = groups / absmaxs[:, None]

        # 3. Softsign: y = 2x / (1 + |x|) (if enabled)
        if SOFTSIGN:
            transformed = 2.0 * normalized / (1.0 + tl.abs(normalized))
        else:
            transformed = normalized

        # 4. Quantize
        q_f32 = (transformed * OUT_MAX).reshape((BLOCK_SIZE_N,))
        q_i8 = tl.floor(q_f32 + 0.5).to(OUT_DTYPE)

        # Store quantized values and scales
        tl.store(q_out_ptr + absolute_offsets, q_i8, mask=mask)
        scales_mask = scales_offsets < tl.cdiv(N, GROUP_SIZE)
        tl.store(scales_out_ptr + scales_offsets, absmaxs, mask=scales_mask)


# ------------------------------------------------ Error correction


@triton.jit
def _get_unbiased_exponent(f: tl.tensor, mantissa_bits: tl.constexpr) -> tl.tensor:
    """Extracts the unbiased exponent from a float in Triton.

    Handles normal and subnormal numbers for fp32, fp16, and bf16.
    """
    f = tl.abs(f)  # zero out the sign bit

    # Use appropriate bitcast type based on mantissa bits
    if mantissa_bits == 23:  # fp32
        f_bits = f.to(tl.uint32, bitcast=True)
        exponent_bias = 127
    elif mantissa_bits == 7:  # bf16
        f_bits = f.to(tl.uint16, bitcast=True)
        exponent_bias = 127
    elif mantissa_bits == 10:  # fp16
        f_bits = f.to(tl.uint16, bitcast=True)
        exponent_bias = 15
    else:
        raise ValueError(f"Unsupported mantissa_bits: {mantissa_bits}")

    # exponent_bits = (f_bits >> mantissa_bits) & (1 <<  - 1)
    exponent_bits = f_bits >> mantissa_bits
    # For subnormals (exponent_bits == 0), the effective exponent for ULP
    # calculation is the same as the smallest normal exponent (1 - bias).

    return tl.where(
        exponent_bits == 0,
        1 - exponent_bias,
        exponent_bits.to(tl.int32) - exponent_bias,
    )


@triton.jit
def _log_ulp_for_mantissa(f: tl.tensor, mantissa_bits: tl.constexpr) -> tl.tensor:
    """Calculates the gap between f and the next representable float."""
    # ULP is 2^(exponent - mantissa_bits).
    # For f == 0 or subnormals, _get_unbiased_exponent returns 1 - bias,
    # so this yields log2(smallest_subnormal) — the correct ULP for the
    # entire subnormal range (all subnormals are evenly spaced).
    exponent = _get_unbiased_exponent(f, mantissa_bits)
    return exponent - mantissa_bits


@triton.jit
def _compute_ecc_bits(
    x_f32: tl.tensor,
    x_narrow: tl.tensor,
    SIGNED_ERROR_T: tl.constexpr,
    NUM_MANTISSA_BITS: tl.constexpr,
    SIGNED_MAX_VAL: tl.constexpr,
) -> tl.tensor:
    # Compute reconstruction error we'll get in next iteration from downcasting
    x_recon = x_narrow.to(tl.float32)
    e = x_f32 - x_recon

    # This error is at most +/- the half the gap between the two nearest
    # representable floating point values. So instead of storing the absolute
    # error, we can store it as a signed fraction of this worst-case gap.
    # This is just generic signed integer quantization, but with the scale
    # based on the floating point exponent of this element instead of the
    # absmax of a group of elements.
    log_scale = _log_ulp_for_mantissa(x_narrow, NUM_MANTISSA_BITS) - 1
    # Decomposed approach for numerical stability: e * 2^(-log_scale) = (e * 2^h) * 2^(-log_scale - h)
    neg_log_scale = (-log_scale).to(tl.float32)
    h = tl.floor(neg_log_scale / 2.0)
    temp = e * tl.exp2(h)
    e_norm = temp * tl.exp2(neg_log_scale - h)
    e_clamped = tl.clamp(e_norm, -1.0, 1.0)
    scaled_e = e_clamped * SIGNED_MAX_VAL

    # round to int, accounting for triton lacking a round() function
    rounded_val = tl.floor(tl.abs(scaled_e) + 0.5)
    rounded_val = libdevice.copysign(rounded_val, scaled_e)
    return rounded_val.to(SIGNED_ERROR_T)


@triton.jit
def _apply_error_correction(
    x_narrow: tl.tensor,  # narrow floating-point dtype
    ecc: tl.tensor,  # signed int dtype
    NUM_MANTISSA_BITS: tl.constexpr,
    SIGNED_MAX_VAL: tl.constexpr,
) -> tl.tensor:
    """Reconstruct f32 value from narrower float and error correction bits."""
    x_recon = x_narrow.to(tl.float32)
    # Calculate the same scale factor used during encoding
    log_scale = _log_ulp_for_mantissa(x_narrow, NUM_MANTISSA_BITS) - 1
    # Scale the integer error back into the original floating-point range
    # Decomposed approach for numerical stability: correction * 2^log_scale = (correction * 2^h) * 2^(log_scale - h)
    correction_factor = ecc.to(tl.float32) / SIGNED_MAX_VAL
    log_scale_f32 = log_scale.to(tl.float32)
    h = tl.floor(log_scale_f32 / 2.0)
    e = correction_factor * tl.exp2(h) * tl.exp2(log_scale_f32 - h)
    return x_recon + e


@triton.jit
def _reconstruct_fp32_kernel(
    x_ptr,
    ecc_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_MANTISSA_BITS: tl.constexpr,
    SIGNED_MAX_VAL: tl.constexpr,
):
    """Kernel to reconstruct fp32 weights from narrow dtype + ECC bits."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_narrow = tl.load(x_ptr + offsets, mask=mask, other=0)
    ecc = tl.load(ecc_ptr + offsets, mask=mask, other=0)

    result_f32 = _apply_error_correction(
        x_narrow,
        ecc,
        NUM_MANTISSA_BITS,
        SIGNED_MAX_VAL,
    )
    tl.store(out_ptr + offsets, result_f32, mask=mask)


@triton.jit
def _compute_ecc_bits_kernel(
    fp32_ptr,
    narrow_ptr,
    ecc_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    SIGNED_ERROR_T: tl.constexpr,
    NUM_MANTISSA_BITS: tl.constexpr,
    SIGNED_MAX_VAL: tl.constexpr,
):
    """Kernel to compute ECC bits from fp32 and narrow dtype weights."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    fp32_val = tl.load(fp32_ptr + offsets, mask=mask, other=0.0)
    narrow_val = tl.load(narrow_ptr + offsets, mask=mask, other=0.0)

    ecc = _compute_ecc_bits(
        fp32_val,
        narrow_val,
        SIGNED_ERROR_T,
        NUM_MANTISSA_BITS,
        SIGNED_MAX_VAL,
    )
    tl.store(ecc_out_ptr + offsets, ecc, mask=mask)


def reconstruct_fp32_param(
    param: torch.Tensor, error_bits: torch.Tensor
) -> torch.Tensor:
    """Reconstruct fp32 parameter from narrow dtype parameter + ECC bits.

    This function is useful for loading fp32 weights from the optimizer state for inference,
    particularly when using FlashOptim with ECC-compressed master weights.

    Args:
        param: Parameter tensor in narrow dtype (bf16, fp16, etc.)
        error_bits: ECC tensor (int8 or int16)

    Returns:
        Reconstructed fp32 parameter tensor

    Example:
        >>> optimizer = FlashAdam(model.parameters(), master_weight_bits=24)
        >>> # After training...
        >>> for name, param in model.named_parameters():
        ...     state = optimizer.state[param]
        ...     if "error_bits" in state:
        ...         fp32_param = reconstruct_fp32_param(param, state["error_bits"])
    """
    if error_bits.dtype.is_floating_point:
        raise ValueError(f"error_bits must be int8 or int16, got {error_bits.dtype}. ")

    # Apply ECC reconstruction for int8/int16 error bits
    BLOCK_SIZE = 1024
    n = param.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    out = torch.empty(param.shape, dtype=torch.float32, device=param.device)
    signed_max_val = 127 if error_bits.element_size() == 1 else 32767

    # Ensure tensors are contiguous for view operations
    param_flat = param.contiguous().view(-1)
    error_bits_flat = error_bits.contiguous().view(-1)
    out_flat = out.view(-1)

    _reconstruct_fp32_kernel[grid](
        param_flat,
        error_bits_flat,
        out_flat,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_MANTISSA_BITS=_NUM_MANTISSA_BITS[param.dtype],
        SIGNED_MAX_VAL=signed_max_val,
    )
    return out


def compute_ecc_bits(
    fp32_param: torch.Tensor,
    narrow_param: torch.Tensor,
) -> torch.Tensor:
    """Compute ECC bits from fp32 weights and narrow dtype weights.

    Inverse of reconstruct_fp32_param. Given fp32 master weights and narrow dtype
    parameter weights, computes the error correction bits that should be stored in
    optimizer state.

    Args:
        fp32_param: Parameter tensor in fp32 dtype
        narrow_param: Parameter tensor in narrow dtype (bf16, fp16, etc.)

    Returns:
        ECC tensor (int8 or int16) containing error correction bits

    Example:
        >>> # Load fp32 weights into a bf16 model with ECC
        >>> fp32_weights = torch.load("checkpoint.pt")
        >>> model = Model(dtype=torch.bfloat16)
        >>> optimizer = FlashAdam(model.parameters(), master_weight_bits=24)
        >>> for name, param in model.named_parameters():
        ...     fp32_val = fp32_weights[name]
        ...     param.copy_(fp32_val.to(param.dtype))
        ...     ecc = compute_ecc_bits(fp32_val.to(param.device), param)
        ...     optimizer.state[param]["error_bits"] = ecc
    """
    # Determine error dtype from byte width difference
    fp32_bytes = 4
    narrow_bytes = narrow_param.element_size()
    num_err_bytes = fp32_bytes - narrow_bytes

    if num_err_bytes == 1:
        error_dtype = torch.int8
        signed_max_val = 127
        signed_error_t = tl.int8
    elif num_err_bytes == 2:
        error_dtype = torch.int16
        signed_max_val = 32767
        signed_error_t = tl.int16
    else:
        raise ValueError(
            f"Unsupported dtype combination: fp32 (4 bytes) to {narrow_param.dtype} ({narrow_bytes} bytes). "
            f"Error byte width {num_err_bytes} must be 1 or 2."
        )

    BLOCK_SIZE = 1024
    n = narrow_param.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    out = torch.empty(narrow_param.shape, dtype=error_dtype, device=narrow_param.device)

    # Ensure tensors are contiguous for view operations
    fp32_flat = fp32_param.contiguous().view(-1)
    narrow_flat = narrow_param.contiguous().view(-1)
    out_flat = out.view(-1)

    _compute_ecc_bits_kernel[grid](
        fp32_flat,
        narrow_flat,
        out_flat,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
        SIGNED_ERROR_T=signed_error_t,
        NUM_MANTISSA_BITS=_NUM_MANTISSA_BITS[narrow_param.dtype],
        SIGNED_MAX_VAL=signed_max_val,
    )
    return out


# ------------------------------------------------ Kernel Launch Helpers


@functools.cache
def _get_sm_count(device: Optional[torch.device] = None):
    if device is None:
        device = torch.device("cuda:0")
    return torch.cuda.get_device_properties(device).multi_processor_count


# Grid setup function for the kernel launch
def _make_grid(N: int, meta: dict[str, int]) -> tuple[int]:
    blocks_per_sm_target = meta.get("num_ctas", 2)
    total_num_blocks = triton.cdiv(N, meta["BLOCK_SIZE_N"])
    # Launch 'blocks_per_sm_target' per SM, but no more than total_num_blocks needed
    # And ensure at least 1 block is launched if N > 0.
    grid_dim = min(blocks_per_sm_target * _get_sm_count(), total_num_blocks)
    return (grid_dim,)


# ============================================================================
# Gradient release
# ============================================================================


@dataclass
class GradientReleaseHandle:
    """Handle returned by :func:`enable_gradient_release` to manage gradient-release hooks.

    Call :meth:`remove` to restore normal (lazy) optimizer behaviour.
    """

    _hooks: list
    _optimizer: FlashOptimizer

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._optimizer._gradient_release = False

    @property
    def active(self) -> bool:
        return len(self._hooks) > 0


def _register_plain_hooks(
    model: torch.nn.Module,
    optimizer: FlashOptimizer,
    hooks: list,
    pre_step: Callable[[torch.Tensor, dict[str, Any]], bool] | None,
    param_to_group: dict[int, dict[str, Any]],
) -> None:
    # Shared / tied parameters are safe here for two reasons:
    #   1. model.parameters() deduplicates, so each tensor gets one hook.
    #   2. register_post_accumulate_grad_hook fires once per backward after
    #      all gradient contributions are merged; the hook then sets
    #      p.grad = None, so even a hypothetical re-entry would be a no-op
    #      (the ``p.grad is None`` guard returns early).
    for p in model.parameters():
        if not p.requires_grad:
            continue
        group = param_to_group[id(p)]

        def _make_hook(p: torch.Tensor, group: dict[str, Any]) -> Callable:
            weak_opt = weakref.ref(optimizer)

            def hook(_p: torch.Tensor) -> None:
                opt = weak_opt()
                if opt is None or p.grad is None:
                    return
                if pre_step is not None and not pre_step(p, group):
                    return
                opt.step_param(p, group)
                p.grad = None

            return hook

        hooks.append(p.register_post_accumulate_grad_hook(_make_hook(p, group)))


def enable_gradient_release(
    model: torch.nn.Module,
    optimizer: FlashOptimizer,
    *,
    grad_scaler: "torch.amp.GradScaler | None" = None,
    pre_step: Callable[[torch.Tensor, dict[str, Any]], bool] | None = None,
) -> GradientReleaseHandle:
    """Attach hooks so the optimizer steps each parameter as soon as its
    gradient is ready and frees the gradient immediately.

    While active, both ``optimizer.step()`` and ``optimizer.zero_grad()``
    become no-ops (with a one-time warning) since stepping and gradient
    clearing are handled by the hooks.

    Args:
        model: The module whose parameters are being optimised.
        optimizer: A :class:`FlashOptimizer` that owns the same parameters.
        grad_scaler: *Reserved for future use.*  Passing a non-``None`` value
            raises :class:`NotImplementedError`.
        pre_step: Optional callback ``(param, group) -> bool``.  Returning
            ``False`` skips the step for that parameter.

    Returns:
        A :class:`GradientReleaseHandle` whose :meth:`~GradientReleaseHandle.remove` method
        detaches all hooks and restores normal ``step`` / ``zero_grad``
        behaviour.

    Raises:
        TypeError: If the model is wrapped in ``DistributedDataParallel`` or
            ``FullyShardedDataParallel`` (FSDP1).
        NotImplementedError: If *grad_scaler* is provided (not yet supported).
    """
    if grad_scaler is not None:
        # TODO: Per-parameter unscale approach:
        #   1. Before stepping each param, multiply p.grad by 1/scale
        #   2. Check for inf/nan in the unscaled gradient
        #   3. If inf found, skip the step for this param (or all params)
        #   4. After all params processed, call scaler.update()
        raise NotImplementedError(
            "GradScaler integration with enable_gradient_release is not yet implemented. "
            "Per-parameter unscaling requires multiplying each grad by 1/scale, "
            "checking for inf, and skipping steps on inf — this is planned but "
            "not yet available."
        )

    # Reject FSDP1 — its resharding lifecycle is incompatible with gradient
    # release.  FSDP2 (fully_shard / FSDPModule) is supported and falls
    # through to _register_plain_hooks below.
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if isinstance(model, FSDP):
        raise TypeError(
            "enable_gradient_release() does not support FullyShardedDataParallel (FSDP1) "
            "models. Use FSDP2 (fully_shard) instead, which is compatible "
            "with per-parameter gradient release."
        )

    # Reject DDP — its reducer unconditionally copies the (zeroed) bucket
    # buffer back into p.grad after the comm hook future resolves, leaking
    # gradient memory.  No Python hook fires after this copy-back, so the
    # leak cannot be prevented.  Benchmarks also show that stepping inside
    # the comm hook callback serialises the DDP bucket pipeline and hurts
    # throughput for realistic model sizes.  Use plain DDP (step/zero_grad)
    # or switch to FSDP2 for gradient release.
    from torch.nn.parallel import DistributedDataParallel as DDP

    if isinstance(model, DDP):
        raise TypeError(
            "enable_gradient_release() does not support DistributedDataParallel (DDP). "
        )

    hooks: list = []

    param_to_group: dict[int, dict[str, Any]] = {
        id(p): g for g in optimizer.param_groups for p in g["params"]
    }

    _register_plain_hooks(model, optimizer, hooks, pre_step, param_to_group)

    optimizer._gradient_release = True

    return GradientReleaseHandle(
        _hooks=hooks,
        _optimizer=optimizer,
    )
