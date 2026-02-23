# Copyright 2026 Databricks AI Research authors

import tempfile
import time
import warnings
from collections.abc import Sequence
from contextlib import nullcontext

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from test_training import (
    _CKPT_CONFIGS,
    ToyDataset,
    _create_simple_model,
    _prepare_batches,
    _train_steps,
    ckpt_id,
)
from test_utils import (
    _DESCENDS_PARAM_SHAPES,
    _DTYPE_WIDTHS,
    _FLOAT_DTYPES,
    _MANY_PARAM_SHAPES,
    _MASTER_WEIGHT_BITS,
    _WEIGHT_DECAY_VALUES,
    ADAM_L2_CONFIG,
    ADAMW_CONFIG,
    ADAMW_DECOUPLE_LR_CONFIG,
    DTYPE_ECC_QUANT_CONFIGS,
    DTYPE_ECC_QUANT_FUSED_CONFIGS,
    LION_CONFIG,
    LION_DECOUPLE_LR_CONFIG,
    SGDM_CONFIG,
    SGDM_DECOUPLE_LR_CONFIG,
    SGDM_NESTEROV_CONFIG,
    SGDMW_CONFIG,
    OptimizerTestConfig,
    ReferenceLion,
    compress_state_dict_id,
    dtype_ecc_quant_fused_id,
    dtype_ecc_quant_id,
    dtype_id,
    lr_id,
    master_weight_bits_id,
    nmse,
    quantized_state_id,
    shape_id,
    weight_decay_id,
)
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD

from flashoptim import FlashLion, cast_model
from flashoptim.optimizers import (
    _BITS_TO_BYTES,
    NumericsError,
    _log2_min_expressible_step_size,
)

cossim = torch.cosine_similarity  # avoid ugly linewraps
warnings.filterwarnings("ignore")

np.set_printoptions(linewidth=160, formatter={"float": lambda f: f"{f:5.3f}"})

# Seeds for parametrized random testing
SEEDS = list(range(3))
_OPT_CONFIGS = [LION_CONFIG, SGDM_CONFIG, ADAMW_CONFIG]
_OPT_CONFIGS_WITH_VARIANTS = [
    LION_CONFIG,
    LION_DECOUPLE_LR_CONFIG,
    SGDM_CONFIG,
    SGDM_NESTEROV_CONFIG,
    SGDM_DECOUPLE_LR_CONFIG,
    SGDMW_CONFIG,
    ADAMW_CONFIG,
    ADAMW_DECOUPLE_LR_CONFIG,
    ADAM_L2_CONFIG,
]


def seed_id(seed: int) -> str:
    """Generate readable ID for seed values."""
    return f"seed{seed}"


def w_init_id(w_init: str) -> str:
    """Generate readable ID for weight initialization strategy."""
    return f"w_{w_init}"


def grad_strategy_id(grad_strategy: str) -> str:
    """Generate readable ID for gradient strategy."""
    return f"g_{grad_strategy}"


def d_id(d: int) -> str:
    """Generate readable ID for dimension D."""
    return f"D{d}"


def make_params_with_grads(
    device: str | torch.device, dtype: torch.dtype, generator: torch.Generator
) -> list[torch.Tensor]:
    """Create parameters with gradients for testing."""
    params = []
    device = torch.device(device) if isinstance(device, str) else device
    for shape in _MANY_PARAM_SHAPES:
        p = torch.rand(
            shape, device=device, dtype=dtype, requires_grad=True, generator=generator
        )
        p.grad = torch.rand(shape, device=device, dtype=dtype, generator=generator)
        params.append(p)
    return params


@pytest.fixture(params=_OPT_CONFIGS, ids=[config.name for config in _OPT_CONFIGS])
def opt_config(request: pytest.FixtureRequest) -> OptimizerTestConfig:
    return request.param


@pytest.fixture(
    params=_OPT_CONFIGS_WITH_VARIANTS,
    ids=[config.name for config in _OPT_CONFIGS_WITH_VARIANTS],
)
def opt_config_with_variants(request: pytest.FixtureRequest) -> OptimizerTestConfig:
    return request.param


@pytest.mark.parametrize("seed", SEEDS, ids=seed_id)
@pytest.mark.parametrize(
    "N,D", _MANY_PARAM_SHAPES, ids=[shape_id(shape) for shape in _MANY_PARAM_SHAPES]
)
@pytest.mark.parametrize(
    "dtype,master_weight_bits,quantize,fused",
    DTYPE_ECC_QUANT_FUSED_CONFIGS,
    ids=[dtype_ecc_quant_fused_id(c) for c in DTYPE_ECC_QUANT_FUSED_CONFIGS],
)
def test_modifies_weights_and_momentums(
    opt_config: OptimizerTestConfig,
    seed: int,
    N: int,
    D: int,
    dtype: torch.dtype,
    master_weight_bits: int | None,
    quantize: bool,
    fused: bool,
) -> None:
    """Test that optimizer states initialize to zero before stepping and become non-zero after stepping."""
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(
        (N, D), device=device, requires_grad=False, dtype=dtype, generator=gen
    )
    W = torch.randn(
        (D, D), device=device, requires_grad=True, dtype=dtype, generator=gen
    )
    W_orig = W.detach().clone()

    opt = opt_config.factory(
        [W],
        lr=1.0,
        fused=fused,
        quantize=quantize,
        weight_decay=0.2,
        master_weight_bits=master_weight_bits,
    )

    Y = X @ W
    loss = Y.sum()
    loss.backward()
    torch.testing.assert_close(W_orig, W)  # no weight modification yet

    # Check that optimizer states materialize to zero before the first step
    # Force state initialization by calling the private method used during step
    group = opt.param_groups[0]
    hparams = {k: v for k, v in group.items() if k != "params"}
    opt._ensure_state_initialized(W, hparams=hparams)

    param_state = opt.state[W]
    for key in opt_config.state_var_names:
        momentum = param_state[key].materialize()
        assert momentum.shape == (D, D)
        assert torch.all(momentum == 0), f"State {key} should be zero before first step"

    opt.step()
    opt.zero_grad()

    with pytest.raises(AssertionError):  # opt step modified the weights
        torch.testing.assert_close(W_orig, W)

    # Every momentum should be nonzero with infinite precision, but
    # might be zero after quantization. We turn the _MaybeQuantizedTensor
    # instance into a regular torch Tensor to simplify this check.
    param_state = opt.state[W]
    for key in opt_config.state_var_names:
        momentum = param_state[key].materialize()  # or momentum-like state
        assert momentum.shape == (D, D)
        momentum = momentum.ravel()
        if momentum.numel() == 1:
            assert momentum.item() != 0
        else:
            assert torch.std(momentum).item() > 0


@pytest.mark.parametrize("seed", SEEDS, ids=seed_id)
@pytest.mark.parametrize(
    "N,D", _MANY_PARAM_SHAPES, ids=[shape_id(shape) for shape in _MANY_PARAM_SHAPES]
)
@pytest.mark.parametrize("weight_decay", _WEIGHT_DECAY_VALUES, ids=weight_decay_id)
@pytest.mark.parametrize(
    "dtype,master_weight_bits,quantize,fused",
    DTYPE_ECC_QUANT_FUSED_CONFIGS,
    ids=[dtype_ecc_quant_fused_id(c) for c in DTYPE_ECC_QUANT_FUSED_CONFIGS],
)
def test_changes_with_zero_grads(
    opt_config: OptimizerTestConfig,
    seed: int,
    N: int,
    D: int,
    weight_decay: float,
    dtype: torch.dtype,
    master_weight_bits: int | None,
    quantize: bool,
    fused: bool,
) -> None:
    """Test optimizer behavior with zero gradients, ensuring momentum stays zero and weights change only with weight decay."""
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)
    is_sgd = "SGD" in opt_config.name

    mom_should_be_zero = True
    if is_sgd and weight_decay > 0:
        mom_should_be_zero = False  # SGD includes weight decay in grad

    W = torch.rand((D, D), device=device, requires_grad=True, generator=gen)
    W_orig = W.detach().clone()

    opt = opt_config.factory(
        [W],
        fused=fused,
        quantize=quantize,
        weight_decay=weight_decay,
        master_weight_bits=master_weight_bits,
    )

    zeros_grad = torch.zeros_like(W)
    for _ in range(5):
        W.grad = zeros_grad
        opt.step()
        opt.zero_grad()

        for key in opt_config.state_var_names:
            mom = opt.state[W][key]
            if mom_should_be_zero:
                assert torch.all(mom.materialize() == 0)
                if mom.is_quantized():
                    assert torch.all(mom.quantized == 0)

            if weight_decay:
                assert torch.all(W_orig.abs() > W.abs())
            else:
                torch.testing.assert_close(W_orig, W)  # no weight modification


@pytest.mark.parametrize("seed", SEEDS, ids=seed_id)
@pytest.mark.parametrize(
    "N,D",
    _DESCENDS_PARAM_SHAPES,
    ids=[shape_id(shape) for shape in _DESCENDS_PARAM_SHAPES],
)
@pytest.mark.parametrize(
    "dtype,master_weight_bits,quantize,fused",
    DTYPE_ECC_QUANT_FUSED_CONFIGS,
    ids=[dtype_ecc_quant_fused_id(c) for c in DTYPE_ECC_QUANT_FUSED_CONFIGS],
)
def test_descends(
    opt_config: OptimizerTestConfig,
    seed: int,
    N: int,
    D: int,
    dtype: torch.dtype,
    master_weight_bits: int | None,
    quantize: bool,
    fused: bool,
) -> None:
    """Test that optimizer consistently reduces loss over multiple steps and momentum grows monotonically."""
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)

    X = torch.randn(
        (N, D), device=device, requires_grad=False, dtype=dtype, generator=gen
    )
    W = torch.randn(
        (D, D), device=device, requires_grad=True, dtype=dtype, generator=gen
    )
    Z = torch.randn(
        (N, D), device=device, requires_grad=False, dtype=dtype, generator=gen
    )

    # we use tiny beta1 so we move almost entirely in the gradient direction
    lr = 1e-2
    if "Adam" in opt_config.name:
        # Adam needs a sufficiently large LR to produce representable updates
        # in bf16/fp16 with quantized optimizer states, but not so large that
        # fp16 without master weights (0b) diverges after many steps
        if dtype == torch.float16 and master_weight_bits is None:
            lr = 5e-4
        else:
            lr = 1e-3

    opt = opt_config.factory(
        [W],
        lr=lr,
        weight_decay=0,
        quantize=quantize,
        check_numerics=False,
        fused=fused,
        master_weight_bits=master_weight_bits,
    )

    prev_loss = np.inf
    prev_momentum = None
    num_iters = 10  # CUDA only
    prev_states = {}
    for _i in range(num_iters):
        Y = (X @ W).float()
        loss = (Y - Z).square().mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

        loss_val = loss.item()
        assert loss_val < prev_loss
        prev_loss = loss_val

        # since we're getting the same batch every time and have a small
        # learning rate, our gradients should point in the same direction
        # at each step. Consequently, our momentum should grow each step.
        state_for_param = opt.state[W]
        for key in opt_config.state_var_names:
            momentum = state_for_param[key].materialize()
            assert momentum is not None and momentum.shape == W.shape
            prev_momentum = prev_states.get(key, None)
            if prev_momentum is not None:
                momentum_abs_changes = (momentum - prev_momentum).abs()
                assert momentum_abs_changes.max() > 0
            prev_states[key] = momentum.clone()  # {gpu, f32 cpu} write in place


if torch.cuda.is_available():
    import triton
    import triton.language as tl

    from flashoptim.optimizers import (
        _NUM_MANTISSA_BITS,
        _apply_error_correction,
    )

    @triton.jit
    def _apply_error_correction_kernel(
        x_ptr,
        ecc_ptr,
        out_f32_ptr,
        n_elements: int,
        BLOCK_SIZE: tl.constexpr,
        NUM_MANTISSA_BITS: tl.constexpr,
        SIGNED_MAX_VAL: tl.constexpr,
    ):
        """Kernel to apply ECC.

        Loads data, calls the device function, stores result.
        """
        pid = tl.program_id(axis=0)
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

        tl.store(out_f32_ptr + offsets, result_f32, mask=mask)

    def _apply_ecc(x: torch.Tensor, errors: torch.Tensor) -> torch.Tensor:
        BLOCK_SIZE = 128
        grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
        out = torch.empty_like(x, dtype=torch.float32)
        signed_max_val = 127 if errors.element_size() == 1 else 32767
        _apply_error_correction_kernel[grid](
            x,
            errors,
            out,
            x.numel(),
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_MANTISSA_BITS=_NUM_MANTISSA_BITS[x.dtype],
            SIGNED_MAX_VAL=signed_max_val,
        )
        return out


# Internal combinations for correctness comparison across fused/unfused and quantized/unquantized
@pytest.mark.parametrize("seed", SEEDS, ids=seed_id)
@pytest.mark.parametrize("w_init", ["cyclic", "rand"], ids=w_init_id)
@pytest.mark.parametrize(
    "grad_strategy", ["zero", "ones", "const", "rand"], ids=grad_strategy_id
)
@pytest.mark.parametrize("D", [5, 12], ids=d_id)  # vectorized and unvectorized impls
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES, ids=dtype_id)
def test_fused_unfused_unquantized_same(
    opt_config_with_variants: OptimizerTestConfig,
    seed: int,
    w_init: str,
    grad_strategy: str,
    D: int,
    dtype: torch.dtype,
) -> None:
    """Test that fused and unfused implementations produce nearly identical results across quantization modes."""
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)

    # each optimizer gets a different copy of the weight matrix to optimize
    if w_init == "cyclic":
        W0 = torch.arange(D * D, device=device, requires_grad=False)
        W0 = ((W0 // 2 % 5) - 2).reshape(D, D).to(dtype=dtype)
    elif w_init == "rand":
        W0 = (
            torch.rand(
                size=(D, D),
                device=device,
                requires_grad=False,
                dtype=dtype,
                generator=gen,
            )
            * 4
            - 2
        )
        W0 += 0.01 * torch.sign(W0)  # bound away from 0 to cap rel errors
        W0 = W0.to(dtype=dtype)
    else:
        raise ValueError("Unrecognized w_init: ", w_init)
    W_true = torch.empty_like(W0, requires_grad=True, dtype=torch.float32)
    W_uu = torch.empty_like(W0, requires_grad=True)  # unfused, unquantized
    W_uq = torch.empty_like(W0, requires_grad=True)  # unfused, quantized
    W_fu = torch.empty_like(W0, requires_grad=True)  # fused, unquantized
    W_fq = torch.empty_like(W0, requires_grad=True)  # fused, quantized
    W_fqe = torch.empty_like(W0, requires_grad=True)  # fused, quantized, ecc
    W_other = torch.empty_like(W0, requires_grad=True)
    with torch.no_grad():
        W_true.copy_(W0.to(W_true.dtype))
        W_uu.copy_(W0)
        W_uq.copy_(W0)
        W_fu.copy_(W0)
        W_fq.copy_(W0)
        W_fqe.copy_(W0)
        W_other.copy_(W0)

    # Use larger WD so effective decay (lr * wd for decoupled optimizers)
    # is representable in low-precision dtypes (bf16/fp16)
    lr = 0.1
    weight_decay = 0.1
    if opt_config_with_variants.decouple_lr:
        weight_decay *= lr  # normalize effective decay to match standard configs
    kwargs = {"lr": lr, "weight_decay": weight_decay}
    factory = opt_config_with_variants.factory
    opt_true = factory([W_true], quantize=False, **kwargs)
    opt_uu = factory(
        [W_uu],
        fused=False,
        quantize=False,
        **kwargs,
        master_weight_bits=None,
    )
    opt_uq = factory(
        [W_uq],
        fused=False,
        quantize=True,
        **kwargs,
        master_weight_bits=None,
    )
    opt_fu = factory(
        [W_fu],
        fused=True,
        quantize=False,
        **kwargs,
        master_weight_bits=None,
    )
    opt_fq = factory(
        [W_fq],
        fused=True,
        **kwargs,
        master_weight_bits=None,
    )

    mas_bits = min(32, (W_fqe.element_size() + 2) * 8)
    opt_fqe = factory(
        [W_fqe],
        fused=True,
        **kwargs,
        master_weight_bits=mas_bits,
    )

    if "SGD" in opt_config_with_variants.name:  # SGD not expected to differ from SGD
        opt_other = ReferenceLion([W_other], betas=(0.12, 0.34), lr=lr)
    else:
        opt_other = SGD([W_other], momentum=0.99, lr=lr)

    if grad_strategy == "zero":
        grads = torch.zeros_like(W0)
    elif grad_strategy == "ones":
        grads = ((torch.arange(W0.numel()) % 2) * 2 - 1).reshape(W0.shape)
        grads = grads.to(device=device)
    elif grad_strategy == "const":
        # arange makes blocks have different distros, so we can't
        # get away with bugs like always using the first scale_scale
        grads = torch.arange(
            W0.numel(), device=device, requires_grad=False, dtype=W0.dtype
        ).view(W0.shape)
    elif grad_strategy == "rand":
        grads = torch.tensor([-1])
    else:
        raise ValueError("bad grad_strategy: ", grad_strategy)

    weights_and_opts = [
        (W_true, opt_true),
        (W_uu, opt_uu),
        (W_uq, opt_uq),
        (W_fu, opt_fu),
        (W_fq, opt_fq),
        (W_fqe, opt_fqe),
        (W_other, opt_other),
    ]
    for _it in range(10):
        if grad_strategy == "rand":
            grads = torch.rand(
                W0.shape,
                device=device,
                requires_grad=False,
                dtype=W0.dtype,
                generator=gen,
            )
        for W, opt in weights_and_opts:
            assert W in opt.state or len(opt.state) == 0
            W.grad = grads.clone().to(dtype=W.dtype)
            opt.step()
            opt.zero_grad()

    # reconstruct true params from ecc bits for the ecc case
    if W_fqe.element_size() < 4:
        err_bits = opt_fqe.state[W_fqe]["error_bits"]
        W_fqe = _apply_ecc(W_fqe.detach(), err_bits)
        assert W_fqe.dtype == torch.float32
    W0_f = W0.float()
    diffs_true = (W_true.detach().float() - W0_f).ravel()
    diffs_uu = (W_uu.detach().float() - W0_f).ravel()
    diffs_uq = (W_uq.detach().float() - W0_f).ravel()
    diffs_fq = (W_fq.detach().float() - W0_f).ravel()
    diffs_fqe = (W_fqe.detach().float() - W0_f).ravel()
    diffs_other = (W_other.detach().float() - W0_f).ravel()

    if dtype != torch.bfloat16:
        min_cossim = 0.99
        max_nmse = 0.01
    else:
        min_cossim = 0.975
        max_nmse = 0.05

    # higher-precision steps should match requested precision steps when
    # there's no quantization
    assert cossim(diffs_true, diffs_uu, dim=-1) > min_cossim
    assert nmse(diffs_true, diffs_uu) < max_nmse

    # fused and unfused should be almost identical; the only differences
    # are intermediate upcasting in the fused impl
    assert cossim(diffs_uq, diffs_fq, dim=-1) > 0.99
    assert nmse(diffs_uq, diffs_fq) < 0.01

    assert cossim(diffs_true, diffs_uq, dim=-1) > min_cossim
    assert nmse(diffs_true, diffs_uq) < max_nmse

    # fused impl should be close to unfused version with no quantization
    assert cossim(diffs_true, diffs_fq, dim=-1) > min_cossim
    assert nmse(diffs_true, diffs_fq) < max_nmse

    # fused impl with ECC should also be close to "true" updates
    assert cossim(diffs_true, diffs_fqe, dim=-1) > min_cossim
    assert nmse(diffs_true, diffs_fqe) < max_nmse

    # ECC should reduce error, or at least do no worse (small atol for
    # sporadic cases where the codec doesn't improve all instances)
    atol = 2e-5
    delta = nmse(diffs_true, diffs_fqe) - nmse(diffs_true, diffs_fq)
    assert delta <= atol

    # if a different optimizer's weights aren't different from ours,
    # we haven't changed them enough to meaningfully test the optimizer logic
    if grad_strategy not in ("zero", "ones"):
        assert cossim(diffs_true, diffs_other, dim=-1) < 0.995
        assert nmse(diffs_true, diffs_other) > 0.01


@pytest.mark.parametrize("seed", SEEDS, ids=seed_id)
@pytest.mark.parametrize(
    "dtype,master_weight_bits,quantize",
    DTYPE_ECC_QUANT_CONFIGS,
    ids=[dtype_ecc_quant_id(c) for c in DTYPE_ECC_QUANT_CONFIGS],
)
@pytest.mark.parametrize("max_abs_value", [1.0, 1e2, 1e-2])
@pytest.mark.parametrize("lr", [1e-1, 1e-3, 1e-7], ids=lr_id)
def test_check_numerics(
    seed: int,
    dtype: torch.dtype,
    master_weight_bits: int | None,
    quantize: bool,
    max_abs_value: float,
    lr: float,
):
    """Test that check_numerics properly detects when learning rate is too small to modify weights."""
    # Create a parameter with the specified maximum absolute value
    gen = torch.Generator(device="cuda").manual_seed(seed)
    p = (
        torch.randint(
            2, size=(1024,), device=torch.device("cuda"), dtype=dtype, generator=gen
        )
        - 1
    )
    param = nn.Parameter(p * max_abs_value)

    # Create the optimizer
    optimizer = FlashLion(
        [param],
        lr=lr,
        check_numerics=True,
        quantize=quantize,
        master_weight_bits=master_weight_bits,
    )

    # Create a dummy loss and compute gradients
    loss = param.sum()
    loss.backward()

    # Check if the numerics check throws as expected
    need_step = _log2_min_expressible_step_size(
        dtype, max_abs_value, _BITS_TO_BYTES[master_weight_bits]
    )
    if np.log2(lr) < need_step:
        with pytest.raises(NumericsError):
            optimizer.step()
    else:
        optimizer.step()


@pytest.mark.parametrize("seed", SEEDS, ids=seed_id)
@pytest.mark.parametrize(
    "dtype,master_weight_bits,quantize",
    DTYPE_ECC_QUANT_CONFIGS,
    ids=[dtype_ecc_quant_id(c) for c in DTYPE_ECC_QUANT_CONFIGS],
)
@pytest.mark.parametrize(
    "compress_state_dict", [False, True], ids=compress_state_dict_id
)
def test_vanilla_checkpoint_interop(
    opt_config: OptimizerTestConfig,
    seed: int,
    dtype: torch.dtype,
    master_weight_bits: int | None,
    quantize: bool,
    compress_state_dict: bool,
):
    """Test that optimizer state can be loaded from vanilla PyTorch optimizers and vice versa."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    params = make_params_with_grads(device="cuda", dtype=dtype, generator=gen)

    if compress_state_dict and not quantize:
        pytest.skip("can't export compressed if unquantized")

    def _state_dicts_match(
        opt_baseline: Optimizer,
        opt_compressed: Optimizer,
        check_params: Sequence[torch.Tensor],
    ) -> bool:
        for p in check_params:
            for key in opt_config.state_var_names:
                state_vanilla = opt_baseline.state[p][key]
                state_ours = opt_compressed.state[p][key].materialize()
                cs = F.cosine_similarity(
                    state_vanilla.ravel(), state_ours.ravel(), dim=-1
                ).item()
                if cs <= 0.99:
                    return False
        return True

    # load vanilla PyTorch optimizer's state into FlashOptim optimizer
    opt_torch = opt_config.reference_factory(params, lr=0.1)
    opt_torch.step()  # per-param state is only created when we step
    opt_ours = opt_config.factory(
        params,
        lr=0.1,
        quantize=quantize,
        master_weight_bits=master_weight_bits,
        compress_state_dict=compress_state_dict,
    )
    opt_ours.load_state_dict(opt_torch.state_dict())
    assert _state_dicts_match(opt_torch, opt_ours, params)

    # load FlashOptim optimizer's state into vanilla PyTorch optimizer
    new_opt_torch = opt_config.reference_factory(params)
    with pytest.raises(KeyError) if compress_state_dict else nullcontext():
        new_opt_torch.load_state_dict(opt_ours.state_dict())
        assert _state_dicts_match(new_opt_torch, opt_ours, params)
        new_opt_torch.step()  # make sure the vanilla PyTorch optimizer at least runs

    opt_ours.step()  # make sure the FlashOptim optimizer at least runs


@pytest.mark.parametrize("seed", SEEDS, ids=seed_id)
@pytest.mark.parametrize("quantized_state", [False, True], ids=quantized_state_id)
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES, ids=dtype_id)
@pytest.mark.parametrize(
    "master_weight_bits", _MASTER_WEIGHT_BITS, ids=master_weight_bits_id
)
def test_state_dict_save_load(
    opt_config: OptimizerTestConfig,
    seed: int,
    quantized_state: bool,
    dtype: torch.dtype,
    master_weight_bits: int | None,
):
    """Test that optimizer state can be saved and loaded correctly, preserving quantized and error correction data."""
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)
    params = make_params_with_grads(device=device, dtype=dtype, generator=gen)

    # create optimizer and have it step so that state gets populated
    opt = opt_config.factory(
        params,
        compress_state_dict=quantized_state,
        master_weight_bits=master_weight_bits,
        check_numerics=False,
    )
    opt.step()
    opt.zero_grad()

    # copy state dict into a new instance
    state_dict = opt.state_dict()
    opt_new = opt_config.factory(
        params,
        compress_state_dict=quantized_state,
        master_weight_bits=master_weight_bits,
        check_numerics=False,
    )
    opt_new.load_state_dict(state_dict)
    for p in params:
        d_orig = opt.state[p]
        d_new = opt_new.state[p]
        assert sorted(d_orig.keys()) == sorted(d_new.keys())
        for key in opt_config.state_var_names:
            state_orig = d_orig[key]
            state_new = d_new[key]
            if quantized_state:
                # Optimizer load_state_dict insists on converting scales to
                # dtype of param, which is lossy for bf16 params.
                # Ideally we'd require == for everything but it's less complexity
                # to just relax the bf16 test
                assert torch.all(state_orig.quantized == state_new.quantized)
                if dtype == torch.bfloat16:
                    torch.testing.assert_close(
                        state_orig.scales, state_new.scales, atol=1e-3, rtol=1e-2
                    )
                else:
                    assert torch.all(state_orig.scales == state_new.scales)

            torch.testing.assert_close(
                state_orig.materialize(),
                state_new.materialize(),
                atol=1.0 / (2 * 127),
                rtol=1e-2,
            )

        err_bytes = _BITS_TO_BYTES[master_weight_bits] - _DTYPE_WIDTHS[dtype]
        if err_bytes == 1:
            assert d_new["error_bits"].dtype == torch.int8
            torch.testing.assert_close(
                d_orig["error_bits"].view(dtype=torch.int8),
                d_new["error_bits"].view(dtype=torch.int8),
            )
        elif err_bytes == 2:
            assert d_new["error_bits"].dtype == torch.int16
            torch.testing.assert_close(
                d_orig["error_bits"].view(dtype=torch.int16),
                d_new["error_bits"].view(dtype=torch.int16),
            )

    # The loaded optimizer must produce finite, reasonable params when stepped.
    # Params are initialized in [0, 1] and stepped with small lr, so anything
    # above 10 is clearly wrong.
    for p in params:
        p.grad = torch.rand(p.shape, device=device, dtype=dtype, generator=gen)
    opt_new.step()
    for p in params:
        assert p.isfinite().all()
        assert p.abs().max() < 10


# ---------------------------------------------------------------------------
# Checkpoint disk round-trip
# ---------------------------------------------------------------------------

_CKPT_SEEDS = [0, 1]


@pytest.mark.parametrize("seed", _CKPT_SEEDS, ids=seed_id)
@pytest.mark.parametrize("ckpt_config", _CKPT_CONFIGS, ids=ckpt_id)
@pytest.mark.parametrize(
    "compress_state_dict", [False, True], ids=["uncompressed", "compressed"]
)
def test_state_dict_disk_roundtrip(
    opt_config: OptimizerTestConfig,
    seed: int,
    ckpt_config: tuple[bool, int],
    compress_state_dict: bool,
) -> None:
    """Optimizer state_dict must survive a torch.save -> torch.load round-trip."""
    quantize, master_weight_bits = ckpt_config
    if compress_state_dict and not quantize:
        pytest.skip("compress_state_dict=True requires quantize=True")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    d_in, d_out = 10, 5
    model_dtype = torch.bfloat16 if master_weight_bits in (24, 32) else torch.float32
    dataset = ToyDataset(n=128, d_in=d_in, d_out=d_out, seed=seed)
    batches = _prepare_batches(dataset, 5, model_dtype=model_dtype)

    model = _create_simple_model(d_in, d_out).to("cuda", dtype=model_dtype)
    opt = opt_config.factory(
        model.parameters(),
        lr=0.01,
        quantize=quantize,
        master_weight_bits=master_weight_bits,
        compress_state_dict=compress_state_dict,
        check_numerics=False,
    )
    _train_steps(model, opt, batches, 0, 3)

    # Save to disk and load back
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/opt_state.pt"
        torch.save(opt.state_dict(), path)
        loaded_sd = torch.load(path, weights_only=False)

    # Load into fresh optimizer
    model2 = _create_simple_model(d_in, d_out).to("cuda", dtype=model_dtype)
    model2.load_state_dict(model.state_dict())
    opt2 = opt_config.factory(
        model2.parameters(),
        lr=0.01,
        quantize=quantize,
        master_weight_bits=master_weight_bits,
        compress_state_dict=compress_state_dict,
        check_numerics=False,
    )
    opt2.load_state_dict(loaded_sd)

    # Verify state matches
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        d_orig = opt.state[p1]
        d_new = opt2.state[p2]
        assert sorted(d_orig.keys()) == sorted(d_new.keys())

        for key in opt_config.state_var_names:
            torch.testing.assert_close(
                d_orig[key].materialize(),
                d_new[key].materialize(),
                atol=1.0 / (2 * 127),
                rtol=1e-2,
            )

        if "error_bits" in d_orig:
            torch.testing.assert_close(
                d_orig["error_bits"].view(dtype=torch.int8),
                d_new["error_bits"].view(dtype=torch.int8),
            )

    # Loaded optimizer must produce finite, reasonable params when stepped.
    _train_steps(model2, opt2, batches, 3, 5)
    for p in model2.parameters():
        assert p.isfinite().all()
        assert p.abs().max() < 100


@pytest.mark.parametrize("seed", SEEDS, ids=seed_id)
@pytest.mark.parametrize(
    "N,D", [(32, 32), (256, 256), (1024, 1024), (4096, 4096), (16384, 16384)]
)
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES, ids=dtype_id)
def test_fused_as_fast_as_unfused(
    opt_config: OptimizerTestConfig,
    seed: int,
    N: int,
    D: int,
    dtype: torch.dtype,
    min_elems_traversed: int = 1000000,
):
    """Test that fused implementations are at least as fast as unfused implementations."""
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def _time_kernels(N: int, D: int, min_elems_traversed: int):
        W = torch.randn(
            (N, D), dtype=dtype, device="cuda", requires_grad=True, generator=gen
        )
        with torch.no_grad():  # avoid check_numerics raising
            W.clip_(-1.75, 1.75)
        W.grad = torch.randn(
            (N, D), dtype=dtype, device="cuda", requires_grad=False, generator=gen
        )

        num_iters = int(np.ceil(min_elems_traversed / W.grad.numel()))
        num_iters = min(100, num_iters)  # limit duration when overhead-bound

        times = {}
        combos = [
            (False, False, None),
            (True, False, None),
            (True, True, None),
            (True, True, 24),
            (True, True, 32),
        ]
        for key in combos:
            quantize, fused, master_weight_bits = key

            opt = opt_config.factory(
                [W],
                quantize=quantize,
                fused=fused,
                master_weight_bits=master_weight_bits,
                weight_decay=0.01,
                check_numerics=False,
            )
            for _ in range(3):
                opt.step()  # warmup iters
            torch.cuda.synchronize()
            t_start = time.time()
            for _ in range(num_iters):
                opt.step()
            torch.cuda.synchronize()
            t_end = time.time()
            dur = (t_end - t_start) / num_iters
            times[key] = dur
        return times

    times = _time_kernels(N, D, min_elems_traversed)

    atol = 2e-4  # should always be faster, but atol helps avoid flakiness
    it = 0
    while True:
        try:
            t_noquant = times[(False, False, None)]
            t_nofuse = times[(True, False, None)]
            t_ecc_0B = times[(True, True, None)]
            t_ecc_3B = times[(True, True, 24)]
            t_ecc_4B = times[(True, True, 32)]
            assert t_ecc_0B < t_nofuse + atol
            assert t_ecc_0B < t_noquant + atol
            assert t_ecc_3B < t_nofuse + atol
            assert t_ecc_3B < t_noquant + atol
            assert t_ecc_4B < t_noquant + atol
            print("")
            print("time fused (ms):       ", t_ecc_0B * 1e3)
            print("time fused+24b (ms):   ", t_ecc_3B * 1e3)
            print("time fused+32b (ms):   ", t_ecc_4B * 1e3)
            print("time unfused (ms):     ", t_nofuse * 1e3)
            print("time unquantized (ms): ", t_noquant * 1e3)
            break
        except AssertionError as e:
            if it >= 2:  # allow 3 retries to avoid flakiness
                raise e
        times = _time_kernels(N, D, min_elems_traversed)
        it += 1


@pytest.mark.parametrize("seed", SEEDS, ids=seed_id)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=dtype_id)
@pytest.mark.parametrize("init_mode", ["boundary", "random"])
def test_ecc_increases_precision(
    opt_config: OptimizerTestConfig, seed: int, dtype: torch.dtype, init_mode: str
):
    """Test that error correction codes improve precision, with higher byte widths yielding better accuracy."""
    D_in, D_out = 4096, 4096
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)
    numpy_rng = np.random.default_rng(seed)

    models = []
    for _ in range(3):  # 0-byte, 3-byte, 4-byte
        model = torch.nn.Linear(D_in, D_out, dtype=dtype, device=device, bias=False)
        models.append(model)

    model_ref = torch.nn.Linear(
        D_in, D_out, dtype=torch.float32, device=device, bias=False
    )

    with torch.no_grad():
        W = model_ref.weight

        if init_mode == "boundary":
            # Initialize with values that have specific mantissa patterns to stress ECC
            float_candidates = [
                1.0,  # 0x3f800000 - all-zero mantissa
                1.9999998807907104,  # 0x3fffffff - all-one mantissa
                2.0,  # 0x40000000 - all-zero mantissa
                3.9999997615814209,  # 0x407fffff - all-one mantissa
                0.5,  # 0x3f000000 - all-zero mantissa
                0.9999999403953552,  # 0x3f7fffff - all-one mantissa
                -1.0,  # 0xbf800000 - all-zero mantissa
                -1.9999998807907104,  # 0xbfffffff - all-one mantissa
            ]
            # Randomly choose from candidates for entire tensor
            chosen_indices = numpy_rng.choice(len(float_candidates), size=W.shape)
            W_np = np.array(
                [
                    [float_candidates[chosen_indices[i, j]] for j in range(W.shape[1])]
                    for i in range(W.shape[0])
                ]
            )
            W.copy_(torch.from_numpy(W_np))

        else:  # init_mode == 'random'
            W.copy_(torch.randn(W.shape, device=W.device, dtype=W.dtype, generator=gen))

        initial_weights = []
        for m in models + [model_ref]:
            m.weight.copy_(W.to(m.weight.dtype))
            # store initial weights to measure changes from initialization
            initial_weights.append(m.weight.detach().clone())

    # Create optimizers with different master weight precisions
    lr = 0.01 if opt_config is LION_CONFIG else 1.0  # lion step size = lr
    opt_kwargs = {
        "lr": lr,
        "weight_decay": 0,
        "quantize": True,
        "check_numerics": False,
    }
    opts = [
        opt_config.factory(
            models[0].parameters(),
            **opt_kwargs,
            master_weight_bits=None,
        ),  # no ECC
        opt_config.factory(
            models[1].parameters(),
            **opt_kwargs,
            master_weight_bits=24,
        ),  # 24-bit
        opt_config.factory(
            models[2].parameters(),
            **opt_kwargs,
            master_weight_bits=32,
        ),  # 32-bit
    ]
    opt_ref = opt_config.factory(model_ref.parameters(), **opt_kwargs)  # f32 reference

    models_and_opts = list(zip(models, opts))

    # Generate a single update tensor to use across all models and steps
    # This ensures truncation differences compound predictably
    with torch.no_grad():
        reference_param = model_ref.weight
        base_updates = (
            abs(reference_param)
            * torch.randn(
                reference_param.shape,
                device=reference_param.device,
                dtype=reference_param.dtype,
                generator=gen,
            )
            * 0.05
        )

    # Apply the same artificial gradients across all models and steps
    for _ in range(100):
        for m, opt in models_and_opts + [(model_ref, opt_ref)]:
            opt.zero_grad()
            with torch.no_grad():
                param = m.weight
                if param.grad is None:
                    param.grad = torch.empty_like(param)
                param.grad.copy_(base_updates.to(param.dtype))
            opt.step()

    total_params = sum([p.numel() for p in model_ref.parameters()])
    mean_abs_errs = {}
    rel_errs = {}

    # Ground truth param changes
    ref_initial = initial_weights[-1].to(dtype=torch.float64)
    ref_final = model_ref.weight.to(dtype=torch.float64)
    ref_changes = ref_final - ref_initial

    for idx, (ecc_bytes, model) in enumerate(
        [(0, models[0]), (3, models[1]), (4, models[2])]
    ):
        error_sum = torch.tensor([0.0], device=device, dtype=torch.float64)
        ref_change_norm = torch.tensor([0.0], device=device, dtype=torch.float64)

        # Convert to float32 for precise error computation
        model_initial = initial_weights[idx].to(dtype=torch.float64)
        model_final = model.weight.to(dtype=torch.float64)
        model_changes = model_final - model_initial

        # Error is difference between model changes and reference changes
        diffs = model_changes - ref_changes
        error_sum += diffs.abs().sum()
        ref_change_norm += ref_changes.abs().sum()

        error_sum_float = error_sum.item()
        mean_abs_errs[ecc_bytes] = error_sum_float / total_params
        rel_errs[ecc_bytes] = error_sum_float / ref_change_norm.item()

    print("Mean absolute errors in parameter changes: ", mean_abs_errs)
    print("Relative errors in parameter changes:     ", rel_errs)

    # Check proper ordering: more ECC bytes should give lower error
    assert mean_abs_errs[3] < mean_abs_errs[0], (
        f"3-byte ECC should beat no ECC: {mean_abs_errs[3]} vs {mean_abs_errs[0]} -- diff {mean_abs_errs[0] - mean_abs_errs[3]:.5g}"
    )
    assert mean_abs_errs[4] < mean_abs_errs[0], (
        f"4-byte ECC should beat no ECC: {mean_abs_errs[4]} vs {mean_abs_errs[0]} -- diff {mean_abs_errs[0] - mean_abs_errs[4]:.5g}"
    )

    # TODO FIXME: currently there are some outliers where 4-byte ECC doesn't beat 3-byte
    assert mean_abs_errs[4] < mean_abs_errs[3] + 1e-5, (
        f"4-byte ECC should beat 3-byte ECC: {mean_abs_errs[4]} vs {mean_abs_errs[3]} -- diff {mean_abs_errs[3] - mean_abs_errs[4]:.5g}"
    )


@pytest.mark.parametrize("seed", SEEDS, ids=seed_id)
def test_gradient_release_matches_manual_step(
    opt_config: OptimizerTestConfig, seed: int
):
    """Test that enable_gradient_release produces identical results to explicit step/zero_grad."""
    from flashoptim import enable_gradient_release

    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)
    dtype = torch.float32
    N, D = 32, 5

    X = torch.randn(
        (N, D), device=device, requires_grad=False, dtype=dtype, generator=gen
    )

    # Create two identical modules
    W_init = torch.randn((D, D), device=device, dtype=dtype, generator=gen)

    model_release = nn.Linear(D, D, bias=False, device=device, dtype=dtype)
    model_manual = nn.Linear(D, D, bias=False, device=device, dtype=dtype)
    with torch.no_grad():
        model_release.weight.copy_(W_init)
        model_manual.weight.copy_(W_init)

    kwargs = {
        "lr": 1e-2,
        "quantize": False,
        "check_numerics": False,
    }
    opt_release = opt_config.factory(model_release.parameters(), **kwargs)
    opt_manual = opt_config.factory(model_manual.parameters(), **kwargs)

    handle = enable_gradient_release(model_release, opt_release)

    prev_loss_release = np.inf
    prev_loss_manual = np.inf
    for _ in range(3):
        # Gradient release: backward triggers hook → step + free grad
        Y_release = X @ model_release.weight.T
        loss_release = (Y_release * Y_release).mean()
        loss_release.backward()
        # step/zero_grad are no-ops in gradient release mode

        # Manual: explicit step/zero_grad
        Y_manual = X @ model_manual.weight.T
        loss_manual = (Y_manual * Y_manual).mean()
        loss_manual.backward()
        opt_manual.step()
        opt_manual.zero_grad()

        # Check that both optimizers achieve descent
        assert loss_release.item() < prev_loss_release
        assert loss_manual.item() < prev_loss_manual
        prev_loss_release = loss_release.item()
        prev_loss_manual = loss_manual.item()

        # Check that parameters are close between release and manual
        torch.testing.assert_close(model_release.weight, model_manual.weight)

        # Check that optimizer states are identical
        for key in opt_config.state_var_names:
            state_release = opt_release.state[model_release.weight][key].materialize()
            state_manual = opt_manual.state[model_manual.weight][key].materialize()
            torch.testing.assert_close(state_release, state_manual)

    handle.remove()


@pytest.mark.parametrize("seed", SEEDS, ids=seed_id)
def test_gradient_release_shared_params(opt_config: OptimizerTestConfig, seed: int):
    """Tied/shared params are stepped exactly once per backward in gradient-release mode."""
    from flashoptim import enable_gradient_release

    device = "cuda"
    V, D = 32, 16
    embedding = nn.Embedding(V, D, device=device)
    linear = nn.Linear(D, V, bias=False, device=device)
    linear.weight = embedding.weight
    model = nn.Sequential(embedding, linear)
    assert len(list(model.parameters())) == 1

    opt = opt_config.factory(
        model.parameters(), lr=1e-2, quantize=False, check_numerics=False
    )
    step_counts: list[int] = []
    handle = enable_gradient_release(
        model, opt, pre_step=lambda p, g: (step_counts.append(1) or True)
    )

    gen = torch.Generator(device=device).manual_seed(seed)
    for _ in range(3):
        step_counts.clear()
        x = torch.randint(0, V, (4, 8), device=device, generator=gen)
        model(x).sum().backward()
        assert len(step_counts) == 1, f"shared param stepped {len(step_counts)}x"

    handle.remove()


@pytest.mark.parametrize("seed", SEEDS, ids=seed_id)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=dtype_id)
@pytest.mark.parametrize("master_weight_bits", [24, 32], ids=master_weight_bits_id)
@pytest.mark.parametrize(
    "opt_config", [ADAMW_CONFIG, SGDM_CONFIG, LION_CONFIG], ids=lambda cfg: cfg.name
)
def test_fp32_state_dict_roundtrip(opt_config, seed, dtype, master_weight_bits):
    """Test that set_fp32_model_state_dict → get_fp32_model_state_dict roundtrips correctly."""
    gen = torch.Generator(device="cuda").manual_seed(seed)

    # Create model with narrow dtype
    params = [
        torch.randn(
            32, 32, dtype=dtype, device="cuda", requires_grad=True, generator=gen
        )
        for _ in range(2)
    ]

    # Create optimizer with ECC enabled
    opt = opt_config.factory(params, lr=0.01, master_weight_bits=master_weight_bits)

    # Take a few optimizer steps to populate state (including error_bits)
    for _ in range(3):
        loss = sum((p * p).sum() for p in params)
        loss.backward()
        opt.step()
        opt.zero_grad()

    # Create a fake model for testing (just wraps params)
    class FakeModel(torch.nn.Module):
        def __init__(self, params):
            super().__init__()
            for i, p in enumerate(params):
                self.register_parameter(f"param_{i}", torch.nn.Parameter(p))

    model = FakeModel(params)

    # Get original fp32 state
    orig_fp32_state = opt.get_fp32_model_state_dict(model)

    # Modify model weights to ensure set_fp32_model_state_dict actually changes things
    for p in params:
        p.data.fill_(0.123)

    # Set fp32 state back
    opt.set_fp32_model_state_dict(model, orig_fp32_state)

    # Get reconstructed fp32 state
    new_fp32_state = opt.get_fp32_model_state_dict(model)

    # Compare original vs reconstructed
    for name in orig_fp32_state.keys():
        orig = orig_fp32_state[name]
        new = new_fp32_state[name]

        # Compute metrics
        mse = ((orig - new) ** 2).mean().item()
        var = orig.var().item()
        nmse = mse / (var + 1e-12)  # Normalized MSE

        max_abs_diff = (orig - new).abs().max().item()
        max_rel_error = max_abs_diff / (orig.abs().max().item() + 1e-12)

        # Assert thresholds
        # NMSE should be very small (< 1e-4 for good ECC)
        assert nmse < 1e-6, f"NMSE too high: {nmse:.6f} for {name}"

        # Max relative error should be small
        assert max_rel_error < 1 / 128, (
            f"Max relative error too high: {max_rel_error:.6f} for {name}"
        )


def test_cast_model():
    # --- Basic downcast: all params become bf16 ---
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 10))
    cast_model(model)
    for param in model.parameters():
        assert param.dtype == torch.bfloat16
    out = model(torch.randn(2, 16, dtype=torch.bfloat16))
    assert out.shape == (2, 10)

    # --- Selective keeps norm layers with running stats fp32 ---
    model = nn.Sequential(
        nn.Linear(16, 32), nn.LayerNorm(32), nn.BatchNorm1d(32), nn.Linear(32, 10)
    )
    cast_model(model, selective=True)
    for param in model[1].parameters():  # LayerNorm → now bf16
        assert param.dtype == torch.bfloat16
    for param in model[2].parameters():  # BatchNorm → still fp32
        assert param.dtype == torch.float32
    for param in model[0].parameters():  # Linear
        assert param.dtype == torch.bfloat16
    out = model(torch.randn(2, 16, dtype=torch.bfloat16))
    assert out.shape == (2, 10)

    # --- selective=False casts everything ---
    model = nn.Sequential(nn.Linear(16, 32), nn.LayerNorm(32), nn.Linear(32, 10))
    cast_model(model, selective=False)
    for param in model.parameters():
        assert param.dtype == torch.bfloat16
    out = model(torch.randn(2, 16, dtype=torch.bfloat16))
    assert out.shape == (2, 10)

    # --- full_precision_keywords keeps matching modules fp32 + patches forward ---
    class HeadModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(16, 32)
            self.head = nn.Linear(32, 10)

        def forward(self, x):
            return self.head(self.backbone(x))

    model = HeadModel()
    cast_model(model, full_precision_keywords=["head"])
    for param in model.head.parameters():
        assert param.dtype == torch.float32
    for param in model.backbone.parameters():
        assert param.dtype == torch.bfloat16
    assert getattr(model.head, "_has_fp32_input_hook", False)
    assert len(model.head._forward_pre_hooks) > 0

    # --- Patched forward upcasts bf16 input to fp32 output ---
    x = torch.randn(2, 32, dtype=torch.bfloat16)
    out = model.head(x)
    assert out.dtype == torch.float32

    # --- Hook casts ALL floating-point tensor args, not just the first ---
    class MultiInputBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)

        def forward(self, x, mask):
            return self.linear(x) + mask

    class MultiInputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = MultiInputBlock()

        def forward(self, x, mask):
            return self.block(x, mask)

    model = MultiInputModel()
    cast_model(model, full_precision_keywords=["block"])
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    mask = torch.randn(2, 8, dtype=torch.bfloat16)
    out = model(x, mask)
    assert out.dtype == torch.float32

    # --- Nested modules are hooked via named_modules() substring matching ---
    class NestedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(10, 10)
            self.decoder = nn.ModuleDict(
                {"head": nn.Linear(10, 5), "proj": nn.Linear(10, 3)}
            )

        def forward(self, x):
            h = self.encoder(x)
            return self.decoder["head"](h)

    model = NestedModel()
    cast_model(model, full_precision_keywords=["head"])
    assert getattr(model.decoder["head"], "_has_fp32_input_hook", False)
    assert model.decoder["head"].weight.dtype == torch.float32
    assert not getattr(model.decoder["proj"], "_has_fp32_input_hook", False)
    x = torch.randn(2, 10, dtype=torch.bfloat16)
    assert model(x).dtype == torch.float32

    # --- Segment keyword matching: "head" matches exact segments, not substrings ---
    class SegmentModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.multi_head_attn = nn.Linear(10, 10)
            self.head = nn.Linear(10, 5)
            self.backbone = nn.Linear(10, 10)

        def forward(self, x):
            return self.head(self.multi_head_attn(x))

    model = SegmentModel()
    cast_model(model, full_precision_keywords=["head"])
    assert getattr(model.head, "_has_fp32_input_hook", False)
    assert not getattr(model.multi_head_attn, "_has_fp32_input_hook", False)
    assert not getattr(model.backbone, "_has_fp32_input_hook", False)

    # --- Hook skips non-floating-point first args (e.g. integer indices) ---
    class EmbedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(50, 10)

        def forward(self, indices):
            return self.embed(indices)

    model = EmbedModel()
    cast_model(model, full_precision_keywords=["embed"])
    assert getattr(model.embed, "_has_fp32_input_hook", False)
    indices = torch.tensor([0, 1, 2], dtype=torch.long)
    out = model(indices)
    assert out.dtype == torch.float32  # embed weight is fp32, so output is fp32

    # --- Buffers: selective=False casts buffers, selective=True preserves norm buffers ---
    model = nn.Sequential(nn.BatchNorm2d(3), nn.Flatten(), nn.Linear(3, 10))
    cast_model(model, selective=False)
    for buf in model[0].buffers():
        assert buf.dtype == torch.bfloat16

    model2 = nn.Sequential(nn.BatchNorm2d(3), nn.Flatten(), nn.Linear(3, 10))
    cast_model(model2, selective=True)
    for buf in model2[0].buffers():
        if buf.is_floating_point():
            assert buf.dtype == torch.float32
    out = model2(torch.randn(2, 3, 1, 1, dtype=torch.bfloat16))
    assert out.shape == (2, 10)

    # --- Tied/shared parameters remain tied after downcast ---
    class TiedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32, 16)
            self.output = nn.Linear(16, 32, bias=False)
            self.output.weight = self.embed.weight  # tie weights

        def forward(self, x):
            return self.output(self.embed(x))

    model = TiedModel()
    assert model.embed.weight is model.output.weight
    cast_model(model)
    assert model.embed.weight.dtype == torch.bfloat16
    assert model.embed.weight is model.output.weight
    assert model.embed.weight.data_ptr() == model.output.weight.data_ptr()


@pytest.mark.parametrize(
    "norm, shape, expect_fp32",
    [
        (nn.BatchNorm1d(64), (4, 64), True),
        (nn.InstanceNorm1d(64, affine=True), (4, 64, 16), True),
        (nn.InstanceNorm2d(64, affine=True), (4, 64, 16, 16), True),
        (nn.GroupNorm(4, 64), (4, 64), False),
        (nn.LayerNorm(64), (4, 64), False),
        (nn.RMSNorm(64), (4, 64), False),
    ],
    ids=[
        "BatchNorm1d",
        "InstanceNorm1d",
        "InstanceNorm2d",
        "GroupNorm",
        "LayerNorm",
        "RMSNorm",
    ],
)
def test_cast_model_selective_all_norm_types(norm, shape, expect_fp32):
    """cast_model(selective=True) preserves dtypes correctly and forward pass works."""
    D = 64
    if isinstance(norm, (nn.InstanceNorm1d, nn.InstanceNorm2d)):
        conv = nn.Conv1d if isinstance(norm, nn.InstanceNorm1d) else nn.Conv2d
        model = nn.Sequential(conv(D, D, 1), norm, conv(D, D, 1))
    else:
        model = nn.Sequential(nn.Linear(D, D), norm, nn.Linear(D, D))
    model = model.to("cuda")

    cast_model(model, dtype=torch.bfloat16, selective=True)

    expected_norm_dtype = torch.float32 if expect_fp32 else torch.bfloat16
    for p in model[1].parameters():
        assert p.dtype == expected_norm_dtype
    for buf in model[1].buffers():
        if buf.is_floating_point():
            assert buf.dtype == expected_norm_dtype
    # Non-norm layers should always be bf16
    for idx in (0, 2):
        for p in model[idx].parameters():
            assert p.dtype == torch.bfloat16

    # Forward pass should not crash
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    out = model(x)
    assert out.is_floating_point()


# ========== decouple_lr baseline correctness ==========


@pytest.mark.parametrize("lr", [1e-4, 1e-3, 1e-2], ids=lr_id)
@pytest.mark.parametrize("weight_decay", [0.0, 0.01, 0.1], ids=weight_decay_id)
@pytest.mark.parametrize("shape", [(10, 10), (64, 32)], ids=shape_id)
def test_reference_adamw_decouple_lr_matches_torch_adamw(lr, weight_decay, shape):
    """ReferenceAdamW(decouple_lr=True) with wd_ref=wd*lr must match TorchAdamW.

    LR-decoupled WD: wd_ref = wd_torch * lr, because wd_torch * lr * (lr_t/lr) = wd_torch * lr_t.
    """
    from reference import ReferenceAdamW
    from torch.optim.adamw import AdamW as TorchAdamW

    torch.manual_seed(42)
    p_ref = torch.randn(shape, device="cuda", requires_grad=True)
    p_torch = p_ref.detach().clone().requires_grad_(True)

    betas = (0.9, 0.999)
    eps = 1e-8
    opt_ref = ReferenceAdamW(
        [p_ref],
        lr=lr,
        weight_decay=weight_decay * lr,
        betas=betas,
        eps=eps,
        decouple_lr=True,
    )
    opt_torch = TorchAdamW(
        [p_torch], lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
    )

    for step in range(10):
        torch.manual_seed(42 + step)
        grad = torch.randn_like(p_ref)
        p_ref.grad = grad.clone()
        p_torch.grad = grad.clone()
        opt_ref.step()
        opt_torch.step()

    torch.testing.assert_close(p_ref, p_torch, rtol=1e-5, atol=1e-7)
