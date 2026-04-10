// SPDX-FileCopyrightText: Copyright 2026 Databricks, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// flash_adam_cuda.cu
// Fused Adam / AdamW CUDA kernel replacing the Triton implementation.
//

#include "flash_adam_cuda.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>


template <typename T>
__device__ __forceinline__ void vec_load(const T* ptr, int base, bool valid, float out[VEC]) {
    if (valid) {
        if constexpr (sizeof(T) == 4) {  
            float4 v = *reinterpret_cast<const float4*>(ptr + base);
            out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
        } else if constexpr (sizeof(T) == 2) { 
            uint64_t raw;
            *reinterpret_cast<uint64_t*>(&raw) =
                *reinterpret_cast<const uint64_t*>(ptr + base);
            const T* vals = reinterpret_cast<const T*>(&raw);
            out[0] = param_to_float(vals[0]);
            out[1] = param_to_float(vals[1]);
            out[2] = param_to_float(vals[2]);
            out[3] = param_to_float(vals[3]);
        }
    } else {
        out[0] = out[1] = out[2] = out[3] = 0.f;
    }
}

__device__ __forceinline__ void vec_load_i8(const int8_t* ptr, int base, bool valid, float out[VEC]) {
    if (valid) {
        int32_t raw = *reinterpret_cast<const int32_t*>(ptr + base);  // 32-bit = 4×int8
        out[0] = (float)((int8_t)(raw & 0xFF));
        out[1] = (float)((int8_t)((raw >> 8)  & 0xFF));
        out[2] = (float)((int8_t)((raw >> 16) & 0xFF));
        out[3] = (float)((int8_t)((raw >> 24) & 0xFF));
    } else {
        out[0] = out[1] = out[2] = out[3] = 0.f;
    }
}

/// Load VEC=4 uint8 values and convert to float.
__device__ __forceinline__ void vec_load_u8(const uint8_t* ptr, int base, bool valid, float out[VEC]) {
    if (valid) {
        uint32_t raw = *reinterpret_cast<const uint32_t*>(ptr + base);
        out[0] = (float)(raw & 0xFF);
        out[1] = (float)((raw >> 8)  & 0xFF);
        out[2] = (float)((raw >> 16) & 0xFF);
        out[3] = (float)((raw >> 24) & 0xFF);
    } else {
        out[0] = out[1] = out[2] = out[3] = 0.f;
    }
}

/// Store VEC=4 float values back to a typed array.
template <typename T>
__device__ __forceinline__ void vec_store(T* ptr, int base, bool valid, const float src[VEC]) {
    if (valid) {
        if constexpr (sizeof(T) == 4) {
            float4 v = {src[0], src[1], src[2], src[3]};
            *reinterpret_cast<float4*>(ptr + base) = v;
        } else if constexpr (sizeof(T) == 2) {
            T vals[VEC] = {float_to_param<T>(src[0]), float_to_param<T>(src[1]),
                           float_to_param<T>(src[2]), float_to_param<T>(src[3])};
            *reinterpret_cast<uint64_t*>(ptr + base) = *reinterpret_cast<const uint64_t*>(vals);
        }
    }
}

/// Store VEC=4 float values as int8 (round-to-nearest, clamp to [-127,127]).
__device__ __forceinline__ void vec_store_i8(int8_t* ptr, int base, bool valid, const float src[VEC]) {
    if (valid) {
        int32_t out = 0;
        for (int i = 0; i < VEC; i++) {
            int v = __float2int_rn(src[i]);
            v = max(-127, min(127, v));
            out |= ((uint8_t)(int8_t)v) << (i * 8);
        }
        *reinterpret_cast<int32_t*>(ptr + base) = out;
    }
}

/// Store VEC=4 float values as uint8 (round-to-nearest, clamp to [0,255]).
__device__ __forceinline__ void vec_store_u8(uint8_t* ptr, int base, bool valid, const float src[VEC]) {
    if (valid) {
        uint32_t out = 0;
        for (int i = 0; i < VEC; i++) {
            unsigned int v = __float2uint_rn(src[i]);
            v = min(v, 255u);
            out |= v << (i * 8);
        }
        *reinterpret_cast<uint32_t*>(ptr + base) = out;
    }
}


template <
    typename ParamT,   // __nv_bfloat16 | __half | float
    int  GROUP_SIZE,   // 32 (must equal warp size for warp-reduce trick)
    int  BLOCK_SIZE,   // threads per block (must be multiple of GROUP_SIZE)
    bool kQuantize,    // INT8 optimizer states
    bool kDecoupled,   // AdamW decoupled weight decay
    bool kUseECC,      // ECC error correction bits
    int  kECCBits      // 8 or 16
>
__global__ void flash_adam_kernel(
    int8_t*  __restrict__ mom_ptr,
    uint8_t* __restrict__ var_ptr,
    __half*  __restrict__ mom_scales_ptr,
    __half*  __restrict__ var_scales_ptr,
    ParamT*  __restrict__ param_ptr,
    const ParamT* __restrict__ grad_ptr,
    void*    __restrict__ ecc_ptr,
    int   N,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int   step
) {
    static_assert(GROUP_SIZE == 32, "GROUP_SIZE must equal warp size (32)");
    static_assert(BLOCK_SIZE % GROUP_SIZE == 0, "BLOCK_SIZE must be multiple of GROUP_SIZE");

    constexpr int ELEMS_PER_BLOCK = BLOCK_SIZE * VEC;

    const int tid  = threadIdx.x;
    const int lane = tid % 32;           

    const float bc1 = 1.f - __powf(beta1, (float)step);
    const float bc2 = 1.f - __powf(beta2, (float)step);

    for (int block_start = (int)blockIdx.x * ELEMS_PER_BLOCK;
         block_start < N;
         block_start += (int)gridDim.x * ELEMS_PER_BLOCK)
    {
        const int elem_base  = block_start + tid * VEC;
        const bool in_bounds = (elem_base + VEC - 1) < N;
        const bool any_valid = (elem_base < N);

        if (!any_valid) continue;  // entire thread is out of bounds

        float grad_f[VEC], param_f[VEC];
        vec_load<ParamT>(grad_ptr,  elem_base, in_bounds || any_valid, grad_f);
        vec_load<ParamT>(param_ptr, elem_base, in_bounds || any_valid, param_f);

        // Mask out-of-bounds elements for tail correctness
        if (!in_bounds && any_valid) {
            for (int i = 0; i < VEC; i++) {
                if (elem_base + i >= N) {
                    grad_f[i] = 0.f;
                    param_f[i] = 0.f;
                }
            }
        }
        if constexpr (kUseECC) {
            using EccT = typename std::conditional<kECCBits == 8, int8_t, int16_t>::type;
            const EccT* ecc_typed = reinterpret_cast<const EccT*>(ecc_ptr);
            for (int i = 0; i < VEC; i++) {
                if (elem_base + i < N) {
                    EccT ecc_val = ecc_typed[elem_base + i];
                    ParamT x_narrow = param_ptr[elem_base + i];
                    param_f[i] = decode_ecc<ParamT, kECCBits>(x_narrow, (int)ecc_val);
                }
            }
        }

        if constexpr (!kDecoupled) {
            for (int i = 0; i < VEC; i++)
                grad_f[i] += param_f[i] * weight_decay;
        }

        float mom_f[VEC], var_f[VEC];

        if constexpr (kQuantize) {
            // Number of threads that cover one GROUP_SIZE-element group:
            constexpr int THREADS_PER_GROUP = GROUP_SIZE / VEC;  // 32/4 = 8

            const int group_idx_in_block = tid / THREADS_PER_GROUP;
            const int group_idx_global = block_start / GROUP_SIZE + group_idx_in_block;
            const int num_total_groups = (N + GROUP_SIZE - 1) / GROUP_SIZE;
            const bool group_valid = (group_idx_global < num_total_groups);

            constexpr unsigned int reduce_mask = 0xffffffffu;
            const int group_lane       = lane % THREADS_PER_GROUP;
            const int group_start_lane = (lane / THREADS_PER_GROUP) * THREADS_PER_GROUP;

            // Load raw quantised values directly into mom_f/var_f to save registers
            vec_load_i8(mom_ptr, elem_base, in_bounds || any_valid, mom_f);
            vec_load_u8(var_ptr, elem_base, in_bounds || any_valid, var_f);

            // Load scales (only lane 0 of each group reads; broadcast via shuffle)
            float mom_scale = 0.f, var_scale = 0.f;
            if (group_lane == 0 && group_valid) {
                mom_scale = __half2float(mom_scales_ptr[group_idx_global]);
                var_scale = __half2float(var_scales_ptr[group_idx_global]);
            }
            mom_scale = __shfl_sync(reduce_mask, mom_scale, group_start_lane);
            var_scale = __shfl_sync(reduce_mask, var_scale, group_start_lane);

            // Dequantise in-place
            for (int i = 0; i < VEC; i++) {
                float m_t = mom_f[i] / 127.f;
                mom_f[i] = inv_softsign(m_t) * mom_scale;

                float v_t = var_f[i] / 255.f;
                float vs  = v_t * var_scale;
                var_f[i]  = vs * vs;  // undo sqrt
            }
        } else {
            // Full-precision states: load directly
            if constexpr (sizeof(ParamT) == 4) {
                vec_load<float>(reinterpret_cast<float*>(mom_ptr), elem_base, in_bounds || any_valid, mom_f);
                vec_load<float>(reinterpret_cast<float*>(var_ptr), elem_base, in_bounds || any_valid, var_f);
            } else {
                vec_load<ParamT>(reinterpret_cast<ParamT*>(mom_ptr), elem_base, in_bounds || any_valid, mom_f);
                vec_load<ParamT>(reinterpret_cast<ParamT*>(var_ptr), elem_base, in_bounds || any_valid, var_f);
            }
        }

        for (int i = 0; i < VEC; i++) {
            // Update first moment
            mom_f[i] = beta1 * mom_f[i] + (1.f - beta1) * grad_f[i];
            // Update second moment
            var_f[i] = beta2 * var_f[i] + (1.f - beta2) * grad_f[i] * grad_f[i];
        }

        // Decoupled weight decay: param *= (1 - wd)
        if constexpr (kDecoupled) {
            const float wd_scale = 1.f - weight_decay;
            for (int i = 0; i < VEC; i++)
                param_f[i] *= wd_scale;
        }

        // Bias-corrected param update
        for (int i = 0; i < VEC; i++) {
            float m_hat = mom_f[i] / bc1;
            float v_hat = var_f[i] / bc2;
            param_f[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
        }

        if (in_bounds) {
            vec_store<ParamT>(param_ptr, elem_base, true, param_f);
        } else {
            for (int i = 0; i < VEC; i++)
                if (elem_base + i < N)
                    param_ptr[elem_base + i] = float_to_param<ParamT>(param_f[i]);
        }

        if constexpr (kUseECC) {
            using EccT = typename std::conditional<kECCBits == 8, int8_t, int16_t>::type;
            EccT* ecc_typed = reinterpret_cast<EccT*>(ecc_ptr);
            for (int i = 0; i < VEC; i++) {
                if (elem_base + i < N) {
                    ParamT x_narrow = param_ptr[elem_base + i];
                    ecc_typed[elem_base + i] =
                        (EccT)encode_ecc<ParamT, kECCBits>(param_f[i], x_narrow);
                }
            }
        }

        if constexpr (kQuantize) {
            constexpr int THREADS_PER_GROUP = GROUP_SIZE / VEC;

            const int group_lane         = lane % THREADS_PER_GROUP;
            const int group_start_lane   = (lane / THREADS_PER_GROUP) * THREADS_PER_GROUP;
            const int group_idx_in_block = tid / THREADS_PER_GROUP;
            const int group_idx_global   = block_start / GROUP_SIZE + group_idx_in_block;
            const int num_total_groups   = (N + GROUP_SIZE - 1) / GROUP_SIZE;
            const bool group_valid       = (group_idx_global < num_total_groups);

            constexpr unsigned int reduce_mask = 0xffffffffu;

            // Compute sqrt(var) in-place: reuse var_f to avoid extra registers
            float mom_abs = 0.f, var_sqrt_abs = 0.f;
            for (int i = 0; i < VEC; i++) {
                mom_abs      = fmaxf(mom_abs,      fabsf(mom_f[i]));
                var_f[i]     = sqrtf(fmaxf(var_f[i], 0.f));  // var_f now holds sqrt(var)
                var_sqrt_abs = fmaxf(var_sqrt_abs, var_f[i]);
            }
#pragma unroll
            for (int off = THREADS_PER_GROUP >> 1; off > 0; off >>= 1) {
                mom_abs      = fmaxf(mom_abs,      __shfl_xor_sync(reduce_mask, mom_abs,      off));
                var_sqrt_abs = fmaxf(var_sqrt_abs, __shfl_xor_sync(reduce_mask, var_sqrt_abs, off));
            }
            mom_abs      = fmaxf(mom_abs,      1e-12f);
            var_sqrt_abs = fmaxf(var_sqrt_abs, 1e-12f);

            // Quantise in-place into mom_f/var_f, then store
            for (int i = 0; i < VEC; i++) {
                mom_f[i] = softsign(mom_f[i] / mom_abs) * 127.f;
                var_f[i] = (var_f[i] / var_sqrt_abs) * 255.f;
            }

            if (in_bounds) {
                vec_store_i8(mom_ptr, elem_base, true, mom_f);
                vec_store_u8(var_ptr, elem_base, true, var_f);
            } else {
                for (int i = 0; i < VEC; i++) {
                    if (elem_base + i < N) {
                        int mv = __float2int_rn(mom_f[i]);
                        mom_ptr[elem_base + i] = (int8_t)max(-127, min(127, mv));
                        unsigned int vv = __float2uint_rn(var_f[i]);
                        var_ptr[elem_base + i] = (uint8_t)min(vv, 255u);
                    }
                }
            }

            // Store scales (only the first thread of each group)
            if (group_lane == 0 && group_valid) {
                mom_scales_ptr[group_idx_global] = __float2half(mom_abs);
                var_scales_ptr[group_idx_global] = __float2half(var_sqrt_abs);
            }
        } else {
            // Store states at param precision
            if (in_bounds) {
                if constexpr (sizeof(ParamT) == 4) {
                    vec_store<float>(reinterpret_cast<float*>(mom_ptr), elem_base, true, mom_f);
                    vec_store<float>(reinterpret_cast<float*>(var_ptr), elem_base, true, var_f);
                } else {
                    vec_store<ParamT>(reinterpret_cast<ParamT*>(mom_ptr), elem_base, true, mom_f);
                    vec_store<ParamT>(reinterpret_cast<ParamT*>(var_ptr), elem_base, true, var_f);
                }
            } else {
                for (int i = 0; i < VEC; i++) {
                    if (elem_base + i < N) {
                        if constexpr (sizeof(ParamT) == 4) {
                            reinterpret_cast<float*>(mom_ptr)[elem_base + i] = mom_f[i];
                            reinterpret_cast<float*>(var_ptr)[elem_base + i] = var_f[i];
                        } else {
                            reinterpret_cast<ParamT*>(mom_ptr)[elem_base + i] = float_to_param<ParamT>(mom_f[i]);
                            reinterpret_cast<ParamT*>(var_ptr)[elem_base + i] = float_to_param<ParamT>(var_f[i]);
                        }
                    }
                }
            }
        }
    }  // end grid-stride loop
}


// Macro to launch a particular instantiation
#define LAUNCH_KERNEL(ParamT, kQ, kD, kE, kB)                                  \
    flash_adam_kernel<ParamT, 32, 256, kQ, kD, kE, kB>                         \
        <<<grid, 256, 0, stream>>>(                                             \
            mom_ptr, var_ptr, mom_scales_ptr, var_scales_ptr,                   \
            reinterpret_cast<ParamT*>(param_ptr),                               \
            reinterpret_cast<const ParamT*>(grad_ptr),                          \
            ecc_ptr, N, lr, beta1, beta2, eps, weight_decay, step)

// Helper: dispatch over (kDecoupled, kUseECC, kECCBits) for a fixed ParamT and kQuantize
#define DISPATCH_FLAGS(ParamT, kQ)                                              \
    do {                                                                        \
        if (use_ecc && ecc_bits == 8) {                                         \
            if (decoupled) { LAUNCH_KERNEL(ParamT, kQ, true,  true,  8); }     \
            else           { LAUNCH_KERNEL(ParamT, kQ, false, true,  8); }     \
        } else if (use_ecc && ecc_bits == 16) {                                 \
            if (decoupled) { LAUNCH_KERNEL(ParamT, kQ, true,  true,  16); }    \
            else           { LAUNCH_KERNEL(ParamT, kQ, false, true,  16); }    \
        } else {                                                                \
            if (decoupled) { LAUNCH_KERNEL(ParamT, kQ, true,  false, 8); }     \
            else           { LAUNCH_KERNEL(ParamT, kQ, false, false, 8); }     \
        }                                                                       \
    } while(0)

void launch_flash_adam(
    int8_t*  mom_ptr,
    uint8_t* var_ptr,
    __half*  mom_scales_ptr,
    __half*  var_scales_ptr,
    void*    param_ptr,
    const void* grad_ptr,
    void*    ecc_ptr,
    int      param_dtype,
    int      N,
    float    lr,
    float    beta1,
    float    beta2,
    float    eps,
    float    weight_decay,
    int      step,
    bool     quantize,
    bool     decoupled,
    bool     use_ecc,
    int      ecc_bits,
    int      group_size,
    cudaStream_t stream
) {
    if (N == 0) return;

    // Cache SM count per device to avoid repeated cudaGetDeviceProperties calls.
    static int cached_sm_count[64] = {};  // index by device id (up to 64 GPUs)
    int device;
    cudaGetDevice(&device);
    if (cached_sm_count[device] == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        cached_sm_count[device] = prop.multiProcessorCount;
    }
    const int sm_count = cached_sm_count[device];
    constexpr int BLOCK_SIZE  = 256;
    constexpr int ELEMS_PER_BLOCK = BLOCK_SIZE * VEC;  // 256 * 4 = 1024
    int total_blocks = (N + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
    int grid = min(2 * sm_count, total_blocks);

    // Dispatch on param_dtype × quantize
    if (param_dtype == 0) {  // bf16
        if (quantize) { DISPATCH_FLAGS(__nv_bfloat16, true);  }
        else          { DISPATCH_FLAGS(__nv_bfloat16, false); }
    } else if (param_dtype == 1) {  // fp16
        if (quantize) { DISPATCH_FLAGS(__half, true);  }
        else          { DISPATCH_FLAGS(__half, false); }
    } else {  // fp32
        if (quantize) { DISPATCH_FLAGS(float, true);  }
        else          { DISPATCH_FLAGS(float, false); }
    }
}
