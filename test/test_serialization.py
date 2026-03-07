# Copyright 2026 Databricks AI Research authors

"""Optimizer state_dict serialization tests.

Tests that optimizer state survives save/load roundtrips — in-memory,
to disk, and interop with vanilla PyTorch optimizers.
"""

import tempfile
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Optional, Union

import pytest
import torch
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
    _DTYPE_WIDTHS,
    _FLOAT_DTYPES,
    _MANY_PARAM_SHAPES,
    _MASTER_WEIGHT_BITS,
    ADAMW_CONFIG,
    DTYPE_ECC_QUANT_CONFIGS,
    LION_CONFIG,
    SGDM_CONFIG,
    OptimizerTestConfig,
    compress_state_dict_id,
    dtype_ecc_quant_id,
    dtype_id,
    master_weight_bits_id,
    quantized_state_id,
)
from torch.optim.optimizer import Optimizer

from flashoptim.optimizers import _BITS_TO_BYTES

SEEDS = list(range(3))
_OPT_CONFIGS = [LION_CONFIG, SGDM_CONFIG, ADAMW_CONFIG]


def seed_id(seed: int) -> str:
    return f"seed{seed}"


def make_params_with_grads(
    device: Union[str, torch.device], dtype: torch.dtype, generator: torch.Generator
) -> list[torch.Tensor]:
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
    master_weight_bits: Optional[int],
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
    master_weight_bits: Optional[int],
):
    """Test that optimizer state can be saved and loaded correctly, preserving quantized and error correction data."""
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)
    params = make_params_with_grads(device=device, dtype=dtype, generator=gen)

    opt = opt_config.factory(
        params,
        compress_state_dict=quantized_state,
        master_weight_bits=master_weight_bits,
        check_numerics=False,
    )
    opt.step()
    opt.zero_grad()

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
    for p in params:
        p.grad = torch.rand(p.shape, device=device, dtype=dtype, generator=gen)
    opt_new.step()
    for p in params:
        assert p.isfinite().all()
        assert p.abs().max() < 10


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

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/opt_state.pt"
        torch.save(opt.state_dict(), path)
        loaded_sd = torch.load(path, weights_only=False)

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

    _train_steps(model2, opt2, batches, 3, 5)
    for p in model2.parameters():
        assert p.isfinite().all()
        assert p.abs().max() < 100
