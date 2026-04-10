# SPDX-FileCopyrightText: Copyright 2026 Databricks, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# setup.py
#
# Builds the optional CUDA extension `flashoptim._cuda_adam` when NVCC is
# available.  The pure-Python / Triton path remains fully functional without it.
#
# Build:
#   pip install -e .                 # Python + Triton only (no nvcc needed)
#   FLASHOPTIM_BUILD_CUDA=1 pip install -e .   # force CUDA ext even if nvcc
#                                               # detection would skip it
#   python setup.py build_ext --inplace        # build in-place for dev

import os
import shutil
import sys

from setuptools import setup


def _nvcc_available() -> bool:
    if os.environ.get("FLASHOPTIM_BUILD_CUDA", "0") == "1":
        return True
    return shutil.which("nvcc") is not None


def _get_gencode_flags():
    """
    Build -gencode flags.  We always include a broad baseline set and then
    attempt to add the SM of whatever GPU is currently installed so that
    the native cubin is embedded (faster JIT + guarantees no 'no kernel image'
    errors at runtime).
    """
    sm_set = {80, 86, 89, 90, 100}

    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                sm = p.major * 10 + p.minor
                sm_set.add(sm)
    except Exception:
        pass

    flags = []
    for sm in sorted(sm_set):
        flags.append(f"-gencode=arch=compute_{sm},code=sm_{sm}")
    # Also add a PTX target for forward-compatibility with future GPUs
    max_sm = max(sm_set)
    flags.append(f"-gencode=arch=compute_{max_sm},code=compute_{max_sm}")
    return flags


ext_modules = []

if _nvcc_available():
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        cuda_ext = CUDAExtension(
            name="flashoptim._cuda_adam",
            sources=[
                "flashoptim/csrc/flash_adam_cuda.cu",
                "flashoptim/csrc/flash_adam_cuda_ext.cpp",
            ],
            include_dirs=["flashoptim/csrc"],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "--ftz=true",
                    "--prec-div=false",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "--extended-lambda",
                    "-lineinfo",
                ] + _get_gencode_flags(),
                "cxx": ["-O3", "-std=c++17"],
            },
        )
        ext_modules.append(cuda_ext)

        setup(
            ext_modules=ext_modules,
            cmdclass={"build_ext": BuildExtension},
        )

    except Exception as exc:
        print(
            f"[flashoptim] WARNING: Could not configure CUDA extension ({exc}). "
            "Falling back to Triton-only install.",
            file=sys.stderr,
        )
        setup()
else:
    setup()
