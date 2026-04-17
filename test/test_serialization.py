# Copyright 2026 Databricks AI Research authors

"""Optimizer state_dict serialization tests.

Tests that optimizer state survives save/load roundtrips - in-memory,
to disk, and interop with vanilla PyTorch optimizers.
"""

import gc
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
    ADAMW_DECOUPLE_LR_CONFIG,
    DTYPE_ECC_QUANT_CONFIGS,
    LION_CONFIG,
    LION_DECOUPLE_LR_CONFIG,
    SGDM_CONFIG,
    SGDM_DECOUPLE_LR_CONFIG,
    OptimizerTestConfig,
    Tolerances,
    check_tensor_similarity,
    compress_state_dict_id,
    dtype_ecc_quant_id,
    dtype_id,
    master_weight_bits_id,
    quantized_state_id,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
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
                assert torch.all(state_orig.quantized == state_new.quantized)
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

    # Memory leak check: one extra roundtrip must not grow GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/opt_state.pt"
        torch.save(opt2.state_dict(), path)
        sd = torch.load(path, weights_only=False)
        opt2.load_state_dict(sd)
        del sd

    gc.collect()
    torch.cuda.empty_cache()
    mem_after = torch.cuda.memory_allocated()
    assert mem_after <= mem_before + 2**20, (
        f"Memory leak detected: before={mem_before}, after={mem_after}, "
        f"delta={mem_after - mem_before}"
    )


@pytest.mark.parametrize(
    "compress_state_dict", [False, True], ids=compress_state_dict_id
)
def test_frozen_params_state_dict_roundtrip(
    opt_config: OptimizerTestConfig,
    compress_state_dict: bool,
):
    """State dict roundtrip with mixed frozen/trainable params.

    Frozen params should have no state entries; training continues correctly
    after load.
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    d_in, d_out = 10, 5
    model = _create_simple_model(d_in, d_out).to("cuda")

    # Freeze weight matrices, keep biases trainable
    model[0].weight.requires_grad = False
    model[2].weight.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    frozen_params = [p for p in model.parameters() if not p.requires_grad]

    # compress_state_dict requires quantize=True
    opt = opt_config.factory(
        model.parameters(),
        lr=0.01,
        quantize=compress_state_dict,
        compress_state_dict=compress_state_dict,
        check_numerics=False,
    )
    dataset = ToyDataset(n=128, d_in=d_in, d_out=d_out, seed=0)
    batches = _prepare_batches(dataset, 10)

    # Train a few steps
    _train_steps(model, opt, batches, 0, 3)

    # Verify frozen params have no state
    for p in frozen_params:
        assert p not in opt.state, "Frozen param should not have optimizer state"

    # Save/load roundtrip
    state_dict = opt.state_dict()

    # Verify state dict has no entries for frozen params
    n_state_entries = len(state_dict["state"])
    assert n_state_entries == len(trainable_params), (
        f"Expected {len(trainable_params)} state entries, got {n_state_entries}"
    )

    model2 = _create_simple_model(d_in, d_out).to("cuda")
    model2[0].weight.requires_grad = False
    model2[2].weight.requires_grad = False
    model2.load_state_dict(model.state_dict())

    opt2 = opt_config.factory(
        model2.parameters(),
        lr=0.01,
        quantize=compress_state_dict,
        compress_state_dict=compress_state_dict,
        check_numerics=False,
    )
    opt2.load_state_dict(state_dict)

    # Continue training after load
    losses = _train_steps(model2, opt2, batches, 3, 10)

    # Verify training works: loss should be finite
    for loss_val in losses:
        assert torch.isfinite(torch.tensor(loss_val)), f"Non-finite loss: {loss_val}"

    for p in model2.parameters():
        assert p.isfinite().all()
        assert p.abs().max() < 100


# ============================================================================
# ECC bits bit-exact roundtrip
# ============================================================================

_ECC_CONFIGS = [
    # (master_weight_bits, compress_state_dict)
    (24, False),
    (24, True),
    (32, False),
    (32, True),
]


def _ecc_config_id(cfg: tuple[int, bool]) -> str:
    mwb, compress = cfg
    return f"ecc{mwb}b_{'compressed' if compress else 'uncompressed'}"


@pytest.mark.parametrize("seed", [0, 1], ids=seed_id)
@pytest.mark.parametrize("ecc_config", _ECC_CONFIGS, ids=_ecc_config_id)
def test_ecc_bits_bitexact_roundtrip(
    opt_config: OptimizerTestConfig,
    ecc_config: tuple[int, bool],
    seed: int,
):
    """ECC error_bits must survive a save/load roundtrip bit-identically."""
    master_weight_bits, compress = ecc_config

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    d_in, d_out = 10, 5
    model_dtype = torch.bfloat16
    dataset = ToyDataset(n=128, d_in=d_in, d_out=d_out, seed=seed)
    batches = _prepare_batches(dataset, 5, model_dtype=model_dtype)

    model = _create_simple_model(d_in, d_out).to("cuda", dtype=model_dtype)
    opt = opt_config.factory(
        model.parameters(),
        lr=0.01,
        quantize=True,
        master_weight_bits=master_weight_bits,
        compress_state_dict=compress,
        check_numerics=False,
    )
    _train_steps(model, opt, batches, 0, 3)

    state_dict = opt.state_dict()

    model2 = _create_simple_model(d_in, d_out).to("cuda", dtype=model_dtype)
    model2.load_state_dict(model.state_dict())
    opt2 = opt_config.factory(
        model2.parameters(),
        lr=0.01,
        quantize=True,
        master_weight_bits=master_weight_bits,
        compress_state_dict=compress,
        check_numerics=False,
    )
    opt2.load_state_dict(state_dict)

    for p1, p2 in zip(model.parameters(), model2.parameters()):
        d_orig = opt.state[p1]
        d_new = opt2.state[p2]

        for key in opt_config.state_var_names:
            orig = d_orig[key]
            new = d_new[key]
            assert torch.equal(orig.quantized, new.quantized), (
                f"{key} quantized values not bit-identical after roundtrip"
            )
            assert torch.equal(orig.scales, new.scales), (
                f"{key} scales not bit-identical after roundtrip"
            )

        assert "error_bits" in d_orig, "Expected error_bits in state"
        assert torch.equal(
            d_orig["error_bits"].view(dtype=torch.int8),
            d_new["error_bits"].view(dtype=torch.int8),
        ), "error_bits not bit-identical after roundtrip"


# ============================================================================
# Compressed vs uncompressed precision equivalence
# ============================================================================


@pytest.mark.parametrize("seed", [0, 1], ids=seed_id)
@pytest.mark.parametrize("master_weight_bits", [24, 32], ids=master_weight_bits_id)
def test_compress_vs_uncompress_precision(
    opt_config: OptimizerTestConfig,
    master_weight_bits: int,
    seed: int,
):
    """Both compress modes should produce identical materialized optimizer states."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    d_in, d_out = 10, 5
    model_dtype = torch.bfloat16
    dataset = ToyDataset(n=128, d_in=d_in, d_out=d_out, seed=seed)
    batches = _prepare_batches(dataset, 5, model_dtype=model_dtype)

    def _save_load(compress: bool) -> list[dict]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        model = _create_simple_model(d_in, d_out).to("cuda", dtype=model_dtype)
        opt = opt_config.factory(
            model.parameters(),
            lr=0.01,
            quantize=True,
            master_weight_bits=master_weight_bits,
            compress_state_dict=compress,
            check_numerics=False,
        )
        _train_steps(model, opt, batches, 0, 3)
        sd = opt.state_dict()

        # Load into fresh optimizer
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        model2 = _create_simple_model(d_in, d_out).to("cuda", dtype=model_dtype)
        model2.load_state_dict(model.state_dict())
        opt2 = opt_config.factory(
            model2.parameters(),
            lr=0.01,
            quantize=True,
            master_weight_bits=master_weight_bits,
            compress_state_dict=compress,
            check_numerics=False,
        )
        opt2.load_state_dict(sd)

        states = []
        for p in model2.parameters():
            d = {}
            for key in opt_config.state_var_names:
                d[key] = opt2.state[p][key].materialize().clone()
            if "error_bits" in opt2.state[p]:
                d["error_bits"] = opt2.state[p]["error_bits"].clone()
            states.append(d)
        return states

    states_compressed = _save_load(compress=True)
    states_uncompressed = _save_load(compress=False)

    for sc, su in zip(states_compressed, states_uncompressed):
        for key in opt_config.state_var_names:
            torch.testing.assert_close(
                sc[key],
                su[key],
                atol=1.0 / (2 * 127),
                rtol=1e-2,
                msg=f"Materialized state '{key}' differs between compress modes",
            )
        if "error_bits" in sc:
            assert torch.equal(
                sc["error_bits"].view(dtype=torch.int8),
                su["error_bits"].view(dtype=torch.int8),
            ), "error_bits differ between compress modes"


# ============================================================================
# Training continuation: both compress modes match continuous baseline
# ============================================================================


@pytest.mark.parametrize("seed", [0], ids=seed_id)
@pytest.mark.parametrize("master_weight_bits", [24, 32], ids=master_weight_bits_id)
def test_training_continuation_both_compress_modes(
    opt_config: OptimizerTestConfig,
    master_weight_bits: int,
    seed: int,
):
    """Training resumed from checkpoint (both modes) should match continuous baseline."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    d_in, d_out = 10, 5
    model_dtype = torch.bfloat16
    dataset = ToyDataset(n=256, d_in=d_in, d_out=d_out, seed=seed)
    total_steps = 10
    ckpt_step = 5
    batches = _prepare_batches(dataset, total_steps, model_dtype=model_dtype)

    def _make_model_opt():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        m = _create_simple_model(d_in, d_out).to("cuda", dtype=model_dtype)
        o = opt_config.factory(
            m.parameters(),
            lr=0.01,
            quantize=True,
            master_weight_bits=master_weight_bits,
            compress_state_dict=False,
            check_numerics=False,
        )
        return m, o

    # Continuous baseline: train all steps
    model_base, opt_base = _make_model_opt()
    _train_steps(model_base, opt_base, batches, 0, total_steps)

    # Checkpoint + resume for each compress mode
    for compress in [True, False]:
        model_a, opt_a = _make_model_opt()
        _train_steps(model_a, opt_a, batches, 0, ckpt_step)

        # Save with specific compress mode
        opt_a._compress_state_dict = compress
        sd = opt_a.state_dict()

        model_b, opt_b = _make_model_opt()
        # Copy model weights at checkpoint
        model_b.load_state_dict(model_a.state_dict())
        opt_b._compress_state_dict = compress
        opt_b.load_state_dict(sd)

        _train_steps(model_b, opt_b, batches, ckpt_step, total_steps)

        for p_base, p_resumed in zip(model_base.parameters(), model_b.parameters()):
            if compress:
                # Compressed roundtrip is bit-exact (int8+scales preserved losslessly)
                assert torch.equal(p_base, p_resumed), (
                    "Training continuation diverged (compress=True); "
                    "expected bit-exact match"
                )
            else:
                # Uncompressed roundtrip re-quantizes (materialize → quantize),
                # introducing one quantization step of error that compounds
                # over subsequent training steps.
                torch.testing.assert_close(
                    p_base,
                    p_resumed,
                    atol=5e-2,
                    rtol=5e-2,
                    msg="Training continuation diverged (compress=False)",
                )


# ---------------------------------------------------------------------------
# test_lr_scheduler_roundtrip
# ---------------------------------------------------------------------------


def _train_steps_with_scheduler(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    start: int,
    end: int,
) -> list[float]:
    model.train()
    loss_fn = torch.nn.MSELoss()
    losses: list[float] = []
    for i in range(start, end):
        xb, yb = batches[i]
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return losses


_LR_SCHED_CONFIGS: dict[str, OptimizerTestConfig] = {
    "lion": LION_CONFIG,
    "lion_decouple_lr": LION_DECOUPLE_LR_CONFIG,
    "sgdm": SGDM_CONFIG,
    "sgdm_decouple_lr": SGDM_DECOUPLE_LR_CONFIG,
    "adamw": ADAMW_CONFIG,
    "adamw_decouple_lr": ADAMW_DECOUPLE_LR_CONFIG,
}


@pytest.mark.parametrize(
    "cfg_name", _LR_SCHED_CONFIGS.keys(), ids=_LR_SCHED_CONFIGS.keys()
)
def test_lr_scheduler_roundtrip(cfg_name: str) -> None:
    """CosineAnnealingLR schedule: checkpoint at step 10, resume, match no-checkpoint run."""
    cfg = _LR_SCHED_CONFIGS[cfg_name]

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    d_in, d_out = 10, 5
    model_dtype = torch.float32
    total_steps = 20
    ckpt_step = 10
    lr = 0.01

    dataset = ToyDataset(n=128, d_in=d_in, d_out=d_out, seed=seed)
    batches = _prepare_batches(dataset, total_steps, model_dtype=model_dtype)

    # -- reference run (vanilla PyTorch optimizer, 20 steps) --
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model_ref = _create_simple_model(d_in, d_out).to("cuda", dtype=model_dtype)
    opt_ref = cfg.reference_factory(model_ref.parameters(), lr=lr)
    sched_ref = CosineAnnealingLR(opt_ref, T_max=total_steps)
    _train_steps_with_scheduler(model_ref, opt_ref, sched_ref, batches, 0, total_steps)

    # -- flash optimizer, no checkpoint, 20 steps --
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model_flash = _create_simple_model(d_in, d_out).to("cuda", dtype=model_dtype)
    opt_flash = cfg.factory(
        model_flash.parameters(),
        lr=lr,
        quantize=False,
        compress_state_dict=False,
        check_numerics=False,
    )
    sched_flash = CosineAnnealingLR(opt_flash, T_max=total_steps)
    _train_steps_with_scheduler(
        model_flash, opt_flash, sched_flash, batches, 0, total_steps
    )

    # Flash vs vanilla reference: different implementations (esp. decouple_lr) diverge
    # over 20 steps. Worst observed: nmse ~1e-2, max_diff ~3.5e-2.
    ref_tol = Tolerances(rtol=0.1, atol=0.05, min_cossim=0.999, max_nmse=0.05)
    ref_errors = []
    for (n, p_ref), (_, p_flash) in zip(
        model_ref.named_parameters(), model_flash.named_parameters()
    ):
        ref_errors.extend(check_tensor_similarity(p_ref, p_flash, n, tol=ref_tol))
    assert not ref_errors, "\n".join(ref_errors)

    # -- flash optimizer with checkpoint at step 10 --
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model_ckpt = _create_simple_model(d_in, d_out).to("cuda", dtype=model_dtype)
    opt_ckpt = cfg.factory(
        model_ckpt.parameters(),
        lr=lr,
        quantize=False,
        compress_state_dict=False,
        check_numerics=False,
    )
    sched_ckpt = CosineAnnealingLR(opt_ckpt, T_max=total_steps)
    _train_steps_with_scheduler(model_ckpt, opt_ckpt, sched_ckpt, batches, 0, ckpt_step)

    with tempfile.TemporaryDirectory() as tmpdir:
        torch.save(model_ckpt.state_dict(), f"{tmpdir}/model.pt")
        torch.save(opt_ckpt.state_dict(), f"{tmpdir}/opt.pt")
        torch.save(sched_ckpt.state_dict(), f"{tmpdir}/sched.pt")

        # load into fresh instances
        model_resumed = _create_simple_model(d_in, d_out).to("cuda", dtype=model_dtype)
        model_resumed.load_state_dict(
            torch.load(f"{tmpdir}/model.pt", weights_only=True)
        )
        opt_resumed = cfg.factory(
            model_resumed.parameters(),
            lr=lr,
            quantize=False,
            compress_state_dict=False,
            check_numerics=False,
        )
        opt_resumed.load_state_dict(torch.load(f"{tmpdir}/opt.pt", weights_only=False))
        sched_resumed = CosineAnnealingLR(opt_resumed, T_max=total_steps)
        sched_resumed.load_state_dict(
            torch.load(f"{tmpdir}/sched.pt", weights_only=False)
        )

    _train_steps_with_scheduler(
        model_resumed, opt_resumed, sched_resumed, batches, ckpt_step, total_steps
    )

    # Checkpoint-resumed run must be bit-identical to the no-checkpoint run.
    for (name, p_flash), (_, p_resumed) in zip(
        model_flash.named_parameters(), model_resumed.named_parameters()
    ):
        assert torch.equal(p_flash, p_resumed), (
            f"{name}: params diverged after checkpoint roundtrip"
        )
