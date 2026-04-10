# SPDX-FileCopyrightText: Copyright 2026 Databricks, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# bench/bench_cuda_adam.py
#
# Performance benchmark: CUDA Adam kernel vs Triton reference.
#
# Usage:
#   python bench/bench_cuda_adam.py                  # all configs, default warmup/iters
#   python bench/bench_cuda_adam.py --iters 200 --warmup 50
#   python bench/bench_cuda_adam.py --csv results.csv

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _make_state(N: int, device: str, quantize: bool, dtype: torch.dtype):
    param = torch.randn(N, device=device, dtype=dtype)
    grad  = torch.randn(N, device=device, dtype=dtype) * 0.01
    if quantize:
        G = (N + 31) // 32
        mom        = torch.zeros(N, device=device, dtype=torch.int8)
        mom_scales = torch.ones(G, device=device, dtype=torch.float16) * 0.01
        var        = torch.zeros(N, device=device, dtype=torch.uint8)
        var_scales = torch.ones(G, device=device, dtype=torch.float16) * 1e-4
    else:
        mom        = torch.zeros(N, device=device, dtype=dtype)
        mom_scales = torch.empty(0, device=device, dtype=torch.float16)
        var        = torch.zeros(N, device=device, dtype=dtype)
        var_scales = torch.empty(0, device=device, dtype=torch.float16)
    return param, grad, mom, mom_scales, var, var_scales


def _run_triton(param, grad, mom, mom_scales, var, var_scales,
                quantize, decoupled, step):
    import flashoptim.optimizers as opt_mod
    opt_mod._try_load_cuda_adam_ext()
    orig, opt_mod._cuda_adam_ext = opt_mod._cuda_adam_ext, None
    try:
        opt_mod._fused_adam_step(
            mom, mom_scales, var, var_scales, param, grad, None,
            1e-3, 0.9, 0.999, 1e-8, 0.01, decoupled, step,
            quantize_optim_states=quantize,
        )
    finally:
        opt_mod._cuda_adam_ext = orig


def _run_cuda(param, grad, mom, mom_scales, var, var_scales,
              quantize, decoupled, step):
    import flashoptim._cuda_adam as ext
    ext.adam_step(
        mom, mom_scales, var, var_scales, param, grad, None,
        1e-3, 0.9, 0.999, 1e-8, 0.01, step,
        quantize, decoupled, 32,
    )


def _bench(fn, warmup: int, iters: int) -> float:
    """Return median wall-time per call in milliseconds (GPU-synchronised)."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    times.sort()
    n = len(times)
    return times[n // 2]  # median


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    N: int
    dtype: str          # "bf16" | "fp16" | "fp32"
    quantize: bool
    decoupled: bool


# All combinations to benchmark
CONFIGS = [
    BenchConfig(N=n, dtype=dt, quantize=q, decoupled=d)
    for n in [4_096, 65_536, 1_048_576, 16_777_216]   # 4K → 16M elements
    for dt in ["bf16", "fp16"]
    for q  in [True, False]
    for d  in [True, False]
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FlashAdam CUDA vs Triton benchmark")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup iterations")
    parser.add_argument("--iters",  type=int, default=100, help="Timed iterations")
    parser.add_argument("--csv",    type=str, default=None, help="Save results to CSV")
    parser.add_argument("--dtype",  type=str, default=None,
                        help="Filter dtype (bf16|fp16|fp32)")
    parser.add_argument("--n",      type=int, default=None,
                        help="Filter N (exact match)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available.", file=sys.stderr)
        sys.exit(1)

    # Check extension
    try:
        import flashoptim._cuda_adam  # noqa: F401
    except ImportError:
        print("ERROR: flashoptim._cuda_adam not compiled. Run:\n"
              "  FLASHOPTIM_BUILD_CUDA=1 python setup.py build_ext --inplace",
              file=sys.stderr)
        sys.exit(1)

    import flashoptim.optimizers as opt_mod
    opt_mod._try_load_cuda_adam_ext()

    # Pre-warm Triton JIT for all (dtype, quantize, decoupled) combos
    # Use N=65536 to ensure the exact same Triton kernel (tiled for that size) is compiled.
    device = "cuda"
    print("Pre-warming Triton JIT...", end=" ", flush=True)
    seen = set()
    for cfg in CONFIGS:
        key = (cfg.dtype, cfg.quantize, cfg.decoupled)
        if key in seen:
            continue
        seen.add(key)
        dtype = _DTYPE_MAP[cfg.dtype]
        p, g, m, ms, v, vs = _make_state(65536, device, cfg.quantize, dtype)
        for _ in range(3):
            _run_triton(p.clone(), g, m.clone(), ms.clone(), v.clone(), vs.clone(),
                        cfg.quantize, cfg.decoupled, 1)
        torch.cuda.synchronize()
    print("done")
    gpu_name = torch.cuda.get_device_name(0)
    p = torch.cuda.get_device_properties(0)
    print(f"\n{'='*72}")
    print(f"  GPU : {gpu_name}  (SM {p.major}.{p.minor})")
    print(f"  Warmup: {args.warmup}  Timed: {args.iters}")
    print(f"{'='*72}")
    print(f"{'N':>12}  {'dtype':>5}  {'quant':>5}  {'decoup':>6}  "
          f"{'Triton(ms)':>10}  {'CUDA(ms)':>10}  {'Speedup':>7}  "
          f"{'BW Triton':>10}  {'BW CUDA':>10}")
    print(f"{'-'*92}")

    configs = CONFIGS
    if args.dtype:
        configs = [c for c in configs if c.dtype == args.dtype]
    if args.n:
        configs = [c for c in configs if c.N == args.n]

    rows = []
    for cfg in configs:
        dtype = _DTYPE_MAP[cfg.dtype]
        param, grad, mom, ms, var, vs = _make_state(cfg.N, device, cfg.quantize, dtype)

        # Clones for each backend so state doesn't accumulate differences
        def triton_step(step=[1]):
            _run_triton(param.clone(), grad, mom.clone(), ms.clone(),
                        var.clone(), vs.clone(), cfg.quantize, cfg.decoupled, step[0])
            step[0] += 1

        def cuda_step(step=[1]):
            _run_cuda(param.clone(), grad, mom.clone(), ms.clone(),
                      var.clone(), vs.clone(), cfg.quantize, cfg.decoupled, step[0])
            step[0] += 1

        t_triton = _bench(triton_step, args.warmup, args.iters)
        t_cuda   = _bench(cuda_step,   args.warmup, args.iters)
        speedup  = t_triton / t_cuda if t_cuda > 0 else float("inf")

        # Approximate memory bandwidth (bytes read + written per step)
        elem_bytes = 2 if cfg.dtype in ("bf16", "fp16") else 4
        if cfg.quantize:
            # mom(i8) + var(u8) + mom_scales(f16) + var_scales(f16) + param + grad
            G = (cfg.N + 31) // 32
            bw_elems = 2 * cfg.N + 2 * G * 2 + 2 * cfg.N * elem_bytes
        else:
            bw_elems = 4 * cfg.N * elem_bytes  # mom + var + param + grad
        bw_gb_triton = bw_elems / (t_triton * 1e-3) / 1e9
        bw_gb_cuda   = bw_elems / (t_cuda   * 1e-3) / 1e9

        print(f"{cfg.N:>12,}  {cfg.dtype:>5}  {str(cfg.quantize):>5}  "
              f"{str(cfg.decoupled):>6}  "
              f"{t_triton:>10.3f}  {t_cuda:>10.3f}  {speedup:>7.2f}x  "
              f"{bw_gb_triton:>9.1f}G  {bw_gb_cuda:>9.1f}G")

        rows.append({
            "gpu": gpu_name,
            "N": cfg.N,
            "dtype": cfg.dtype,
            "quantize": cfg.quantize,
            "decoupled": cfg.decoupled,
            "triton_ms": round(t_triton, 4),
            "cuda_ms":   round(t_cuda,   4),
            "speedup":   round(speedup,  4),
            "bw_triton_GBs": round(bw_gb_triton, 2),
            "bw_cuda_GBs":   round(bw_gb_cuda,   2),
        })

    print(f"{'='*92}\n")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results saved to {args.csv}")


if __name__ == "__main__":
    main()
