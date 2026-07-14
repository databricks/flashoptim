// SPDX-FileCopyrightText: Copyright 2026 Databricks, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// flash_adam_cuda_ext.cpp
// pybind11 / PyTorch C++ extension binding for the fused Adam CUDA kernel.
//


#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>  // at::cuda::getCurrentCUDAStream
#include <cuda_fp16.h>
#include "flash_adam_cuda.cuh"

// Map a torch dtype to our internal integer code:
//   0 = bfloat16, 1 = float16, 2 = float32
static int dtype_code(const torch::Tensor& t) {
    if (t.dtype() == torch::kBFloat16) return 0;
    if (t.dtype() == torch::kFloat16)  return 1;
    if (t.dtype() == torch::kFloat32)  return 2;
    TORCH_CHECK(false, "flash_adam: unsupported param dtype ", t.dtype());
}

/// Main entry point called from Python.
///
/// Arguments match `_fused_adam_step` in optimizers.py exactly so that the
/// Python dispatcher can call this with zero overhead.
///
/// mom        – int8  tensor (quantized) or ParamT tensor (full-precision)
/// mom_scales – float16 tensor (only used when quantize=True)
/// var        – uint8 tensor (quantized) or ParamT tensor (full-precision)
/// var_scales – float16 tensor (only used when quantize=True)
/// param      – bf16/fp16/fp32 parameter tensor (in-place updated)
/// grad       – gradient tensor (same dtype as param)
/// ecc        – optional int8/int16 tensor, or empty tensor (USE_ECC=false)
void adam_step(
    torch::Tensor mom,
    torch::Tensor mom_scales,
    torch::Tensor var,
    torch::Tensor var_scales,
    torch::Tensor param,
    torch::Tensor grad,
    torch::optional<torch::Tensor> ecc,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    int64_t step,
    bool quantize,
    bool decoupled,
    int64_t group_size
) {
    TORCH_CHECK(param.is_cuda(),  "flash_adam: param must be a CUDA tensor");
    TORCH_CHECK(param.is_contiguous(), "flash_adam: param must be contiguous");
    TORCH_CHECK(grad.is_contiguous(),  "flash_adam: grad must be contiguous");
    TORCH_CHECK(mom.is_contiguous(),   "flash_adam: mom must be contiguous");
    TORCH_CHECK(var.is_contiguous(),   "flash_adam: var must be contiguous");
    TORCH_CHECK(group_size == 32, "flash_adam: only group_size=32 is supported");

    const int N = (int)param.numel();
    const int pdtype = dtype_code(param);

    bool use_ecc = false;
    int  ecc_bits = 8;
    void* ecc_ptr = nullptr;

    if (ecc.has_value() && ecc->defined() && ecc->numel() > 0) {
        use_ecc = true;
        TORCH_CHECK(ecc->is_contiguous(), "flash_adam: ecc must be contiguous");
        if (ecc->dtype() == torch::kInt8) {
            ecc_bits = 8;
        } else if (ecc->dtype() == torch::kInt16) {
            ecc_bits = 16;
        } else {
            TORCH_CHECK(false, "flash_adam: ecc must be int8 or int16");
        }
        ecc_ptr = ecc->data_ptr();
    }

    // Resolve the current CUDA stream so the kernel is enqueued correctly
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(param.device().index());

    launch_flash_adam(
        quantize ? mom.data_ptr<int8_t>()   : reinterpret_cast<int8_t*>(mom.data_ptr()),
        quantize ? var.data_ptr<uint8_t>()  : reinterpret_cast<uint8_t*>(var.data_ptr()),
        quantize ? reinterpret_cast<__half*>(mom_scales.data_ptr<at::Half>()) : nullptr,
        quantize ? reinterpret_cast<__half*>(var_scales.data_ptr<at::Half>()) : nullptr,
        param.data_ptr(),
        grad.data_ptr(),
        ecc_ptr,
        pdtype,
        N,
        (float)lr,
        (float)beta1,
        (float)beta2,
        (float)eps,
        (float)weight_decay,
        (int)step,
        quantize,
        decoupled,
        use_ecc,
        ecc_bits,
        (int)group_size,
        stream
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashOptim fused Adam CUDA kernel";
    m.def(
        "adam_step",
        &adam_step,
        "Fused Adam/AdamW step (CUDA)",
        py::arg("mom"),
        py::arg("mom_scales"),
        py::arg("var"),
        py::arg("var_scales"),
        py::arg("param"),
        py::arg("grad"),
        py::arg("ecc"),
        py::arg("lr"),
        py::arg("beta1"),
        py::arg("beta2"),
        py::arg("eps"),
        py::arg("weight_decay"),
        py::arg("step"),
        py::arg("quantize")         = true,
        py::arg("decoupled")        = false,
        py::arg("group_size")       = 32
    );
}
