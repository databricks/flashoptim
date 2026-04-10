# SPDX-FileCopyrightText: Copyright 2026 Databricks, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# test/test_cuda_adam.py
#
# Correctness tests for the CUDA Adam kernel.
#


import math

import pytest
import torch


CUDA_AVAILABLE = torch.cuda.is_available()
CUDA_EXT_AVAILABLE = False
if CUDA_AVAILABLE:
    try:
        import flashoptim._cuda_adam  # noqa: F401
        CUDA_EXT_AVAILABLE = True
    except ImportError:
        pass

requires_cuda_ext = pytest.mark.skipif(
    not (CUDA_AVAILABLE and CUDA_EXT_AVAILABLE),
    reason="CUDA GPU + compiled flashoptim._cuda_adam extension required",
)
requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA GPU required",
)


def _make_state(N, device, quantize, dtype=torch.bfloat16, seed=42):
    """Return (param, grad, mom, mom_scales, var, var_scales)."""
    rng = torch.Generator(device=device).manual_seed(seed)
    param = torch.randn(N, device=device, dtype=dtype, generator=rng)
    grad  = torch.randn(N, device=device, dtype=dtype, generator=rng) * 0.01
    if quantize:
        G = (N + 31) // 32
        mom        = torch.randint(-10, 10, (N,), device=device, dtype=torch.int8)
        mom_scales = (torch.rand(G, device=device, generator=rng, dtype=torch.float32) * 0.1 + 0.01).half()
        var        = torch.randint(10, 50,  (N,), device=device, dtype=torch.uint8)
        var_scales = (torch.rand(G, device=device, generator=rng, dtype=torch.float32) * 0.01 + 1e-4).half()
    else:
        mom        = torch.zeros(N, device=device, dtype=dtype)
        mom_scales = torch.empty(0, device=device, dtype=torch.float16)
        var        = torch.zeros(N, device=device, dtype=dtype)
        var_scales = torch.empty(0, device=device, dtype=torch.float16)
    return param, grad, mom, mom_scales, var, var_scales


def _fp32_adam_step(param_fp32, grad_fp32, mom_fp32, var_fp32,
                    lr, beta1, beta2, eps, wd, step, decoupled):
    """Reference Adam step entirely in fp32 (no quantization)."""
    if decoupled:
        param_fp32.mul_(1.0 - wd)
    else:
        grad_fp32 = grad_fp32 + param_fp32 * wd

    mom_fp32.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
    var_fp32.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    m_hat = mom_fp32 / bc1
    v_hat = var_fp32 / bc2
    param_fp32.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)


def _run_cuda(param, grad, mom, mom_scales, var, var_scales,
              lr, beta1, beta2, eps, wd, step, quantize, decoupled):
    import flashoptim._cuda_adam as ext
    ext.adam_step(
        mom, mom_scales, var, var_scales,
        param, grad, None,
        lr, beta1, beta2, eps, wd, step,
        quantize, decoupled, 32,
    )


def _run_triton(param, grad, mom, mom_scales, var, var_scales,
                lr, beta1, beta2, eps, wd, step, quantize, decoupled):
    import flashoptim.optimizers as opt_mod
    opt_mod._try_load_cuda_adam_ext()
    orig, opt_mod._cuda_adam_ext = opt_mod._cuda_adam_ext, None
    try:
        opt_mod._fused_adam_step(
            mom, mom_scales, var, var_scales,
            param, grad, None,
            lr, beta1, beta2, eps, wd, decoupled, step,
            quantize_optim_states=quantize,
        )
    finally:
        opt_mod._cuda_adam_ext = orig


# ---------------------------------------------------------------------------
# 1. CUDA vs fp32 ground truth – param value
# ---------------------------------------------------------------------------

@requires_cuda_ext
@pytest.mark.parametrize("N", [32, 128, 1024, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("decoupled", [True, False])
def test_cuda_param_vs_fp32_no_quant(N, dtype, decoupled):
    """Without quantization, CUDA param update matches fp32 reference closely."""
    device = "cuda"
    lr, beta1, beta2, eps, wd, step = 1e-3, 0.9, 0.999, 1e-8, 0.01, 1

    param_c, grad_c, mom_c, ms_c, var_c, vs_c = _make_state(N, device, False, dtype)
    param_fp32 = param_c.float().clone()
    grad_fp32  = grad_c.float().clone()
    mom_fp32   = mom_c.float().clone()
    var_fp32   = var_c.float().clone()

    _run_cuda(param_c, grad_c, mom_c, ms_c, var_c, vs_c,
              lr, beta1, beta2, eps, wd, step, False, decoupled)
    _fp32_adam_step(param_fp32, grad_fp32, mom_fp32, var_fp32,
                    lr, beta1, beta2, eps, wd, step, decoupled)

    # bf16/fp16 have ~3e-3 relative precision; allow a few ulps of rounding
    atol = 5e-3
    torch.testing.assert_close(
        param_c.float(), param_fp32,
        atol=atol, rtol=1e-2,
        msg=f"param vs fp32 ref: N={N}, dtype={dtype}, decoupled={decoupled}",
    )


# ---------------------------------------------------------------------------
# 2. CUDA optimizer states vs fp32 ground truth (no quantization)
# ---------------------------------------------------------------------------

@requires_cuda_ext
@pytest.mark.parametrize("N", [128, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("decoupled", [True, False])
def test_cuda_states_vs_fp32_no_quant(N, dtype, decoupled):
    """Without quantization, CUDA mom/var states match fp32 reference."""
    device = "cuda"
    lr, beta1, beta2, eps, wd, step = 1e-3, 0.9, 0.999, 1e-8, 0.01, 1

    param_c, grad_c, mom_c, ms_c, var_c, vs_c = _make_state(N, device, False, dtype)
    param_fp32 = param_c.float().clone()
    grad_fp32  = grad_c.float().clone()
    mom_fp32   = mom_c.float().clone()
    var_fp32   = var_c.float().clone()

    _run_cuda(param_c, grad_c, mom_c, ms_c, var_c, vs_c,
              lr, beta1, beta2, eps, wd, step, False, decoupled)
    _fp32_adam_step(param_fp32, grad_fp32, mom_fp32, var_fp32,
                    lr, beta1, beta2, eps, wd, step, decoupled)

    # mom/var stored at param dtype; allow rounding
    atol = 1e-3
    torch.testing.assert_close(
        mom_c.float(), mom_fp32, atol=atol, rtol=1e-2,
        msg=f"mom vs fp32: N={N}, dtype={dtype}, decoupled={decoupled}",
    )
    torch.testing.assert_close(
        var_c.float(), var_fp32, atol=atol, rtol=1e-2,
        msg=f"var vs fp32: N={N}, dtype={dtype}, decoupled={decoupled}",
    )


# ---------------------------------------------------------------------------
# 3. CUDA vs Triton agreement (regression guard for all flag combos)
# ---------------------------------------------------------------------------

@requires_cuda_ext
@pytest.mark.parametrize("N", [32, 128, 1024, 4096, 10001])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("quantize", [True, False])
@pytest.mark.parametrize("decoupled", [True, False])
def test_cuda_vs_triton_param(N, dtype, quantize, decoupled):
    """CUDA and Triton produce the same updated param (within tolerance)."""
    device = "cuda"
    lr, beta1, beta2, eps, wd, step = 1e-3, 0.9, 0.999, 1e-8, 0.01, 1

    param_t, grad_t, mom_t, ms_t, var_t, vs_t = _make_state(N, device, quantize, dtype)
    param_c = param_t.clone(); grad_c = grad_t.clone()
    mom_c   = mom_t.clone();   ms_c   = ms_t.clone()
    var_c   = var_t.clone();   vs_c   = vs_t.clone()

    _run_triton(param_t, grad_t, mom_t, ms_t, var_t, vs_t,
                lr, beta1, beta2, eps, wd, step, quantize, decoupled)
    _run_cuda(param_c, grad_c, mom_c, ms_c, var_c, vs_c,
              lr, beta1, beta2, eps, wd, step, quantize, decoupled)

    atol = 1e-2 if quantize else 1e-4
    rtol = 1e-2 if quantize else 1e-3
    torch.testing.assert_close(
        param_c.float(), param_t.float(), atol=atol, rtol=rtol,
        msg=f"param CUDA vs Triton: N={N}, dtype={dtype}, q={quantize}, d={decoupled}",
    )


@requires_cuda_ext
@pytest.mark.parametrize("N", [128, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("decoupled", [True, False])
def test_cuda_vs_triton_quant_states(N, dtype, decoupled):
    """With quantization: CUDA and Triton int8/uint8 states are within ±1 LSB,
    and scales are close."""
    device = "cuda"
    lr, beta1, beta2, eps, wd, step = 1e-3, 0.9, 0.999, 1e-8, 0.01, 1

    param_t, grad_t, mom_t, ms_t, var_t, vs_t = _make_state(N, device, True, dtype)
    param_c = param_t.clone(); grad_c = grad_t.clone()
    mom_c   = mom_t.clone();   ms_c   = ms_t.clone()
    var_c   = var_t.clone();   vs_c   = vs_t.clone()

    _run_triton(param_t, grad_t, mom_t, ms_t, var_t, vs_t,
                lr, beta1, beta2, eps, wd, step, True, decoupled)
    _run_cuda(param_c, grad_c, mom_c, ms_c, var_c, vs_c,
              lr, beta1, beta2, eps, wd, step, True, decoupled)

    diff_mom = (mom_c.int() - mom_t.int()).abs().max().item()
    diff_var = (var_c.int() - var_t.int()).abs().max().item()
    assert diff_mom <= 1, f"mom int8 mismatch > 1 LSB: max_diff={diff_mom}, dtype={dtype}, d={decoupled}"
    assert diff_var <= 1, f"var uint8 mismatch > 1 LSB: max_diff={diff_var}, dtype={dtype}, d={decoupled}"

    torch.testing.assert_close(ms_c, ms_t, atol=1e-3, rtol=1e-2, msg="mom_scales mismatch")
    torch.testing.assert_close(vs_c, vs_t, atol=1e-3, rtol=1e-2, msg="var_scales mismatch")


# ---------------------------------------------------------------------------
# 4. Tail / boundary sizes
# ---------------------------------------------------------------------------

@requires_cuda_ext
@pytest.mark.parametrize("N", [1, 3, 31, 33, 63, 65, 127, 129, 255, 257, 10001])
@pytest.mark.parametrize("quantize", [True, False])
def test_cuda_vs_triton_boundary_sizes(N, quantize):
    """CUDA handles non-aligned tensor sizes correctly (tail elements)."""
    device = "cuda"
    dtype = torch.bfloat16
    lr, beta1, beta2, eps, wd, step = 1e-3, 0.9, 0.999, 1e-8, 0.01, 1

    param_t, grad_t, mom_t, ms_t, var_t, vs_t = _make_state(N, device, quantize, dtype)
    param_c = param_t.clone(); grad_c = grad_t.clone()
    mom_c   = mom_t.clone();   ms_c   = ms_t.clone()
    var_c   = var_t.clone();   vs_c   = vs_t.clone()

    _run_triton(param_t, grad_t, mom_t, ms_t, var_t, vs_t,
                lr, beta1, beta2, eps, wd, step, quantize, True)
    _run_cuda(param_c, grad_c, mom_c, ms_c, var_c, vs_c,
              lr, beta1, beta2, eps, wd, step, quantize, True)

    atol = 1e-2 if quantize else 1e-4
    torch.testing.assert_close(
        param_c.float(), param_t.float(), atol=atol, rtol=1e-2,
        msg=f"boundary N={N}, quantize={quantize}",
    )


# ---------------------------------------------------------------------------
# 5. Multi-step numerical stability and drift
# ---------------------------------------------------------------------------

@requires_cuda_ext
@pytest.mark.parametrize("quantize", [True, False])
@pytest.mark.parametrize("decoupled", [True, False])
def test_cuda_multi_step_vs_triton(quantize, decoupled):
    """Over 20 steps, CUDA and Triton produce identical params (within ±1 quant LSB per step)."""
    N, device, dtype = 4096, "cuda", torch.bfloat16
    lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01

    param_t, grad_t, mom_t, ms_t, var_t, vs_t = _make_state(N, device, quantize, dtype)
    param_c = param_t.clone(); mom_c = mom_t.clone(); ms_c = ms_t.clone()
    var_c   = var_t.clone();   vs_c  = vs_t.clone()

    for step in range(1, 21):
        grad_new = torch.randn(N, device=device, dtype=dtype,
                               generator=torch.Generator(device).manual_seed(step)) * 0.01
        _run_triton(param_t, grad_new, mom_t, ms_t, var_t, vs_t,
                    lr, beta1, beta2, eps, wd, step, quantize, decoupled)
        _run_cuda(param_c, grad_new, mom_c, ms_c, var_c, vs_c,
                  lr, beta1, beta2, eps, wd, step, quantize, decoupled)

    assert not param_c.isnan().any(), "NaN in CUDA param after 20 steps"
    assert not param_c.isinf().any(), "Inf in CUDA param after 20 steps"

    # CUDA and Triton should agree closely; quantization adds ~1 LSB per step
    atol = 1e-2 if quantize else 1e-4
    torch.testing.assert_close(
        param_c.float(), param_t.float(), atol=atol, rtol=1e-2,
        msg=f"20-step CUDA vs Triton: quantize={quantize}, decoupled={decoupled}",
    )


# ---------------------------------------------------------------------------
# 6. Integration: FlashAdamW dispatches to CUDA ext
# ---------------------------------------------------------------------------

@requires_cuda_ext
def test_flash_adamw_uses_cuda_ext(monkeypatch):
    """FlashAdamW dispatches to the CUDA extension when available."""
    import flashoptim
    import flashoptim.optimizers as opt_mod

    opt_mod._try_load_cuda_adam_ext()

    calls = []
    orig_step = opt_mod._cuda_adam_ext.adam_step

    def _spy(*args, **kwargs):
        calls.append(1)
        return orig_step(*args, **kwargs)

    monkeypatch.setattr(opt_mod._cuda_adam_ext, "adam_step", _spy)
    monkeypatch.setattr(opt_mod, "_cuda_adam_load_attempted", True)

    model = torch.nn.Linear(64, 64, bias=False).cuda().bfloat16()
    optimizer = flashoptim.FlashAdamW(model.parameters(), lr=1e-3)
    x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    model(x).sum().backward()
    optimizer.step()

    assert len(calls) > 0, "CUDA extension was not called during optimizer.step()"


# ---------------------------------------------------------------------------
# 7. Smoke test: extension imports cleanly
# ---------------------------------------------------------------------------

@requires_cuda
def test_cuda_ext_importable():
    if not CUDA_EXT_AVAILABLE:
        pytest.skip("flashoptim._cuda_adam not compiled")
    import flashoptim._cuda_adam as ext  # noqa: F401
    assert hasattr(ext, "adam_step")
