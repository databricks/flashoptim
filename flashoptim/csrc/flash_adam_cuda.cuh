// SPDX-FileCopyrightText: Copyright 2026 Databricks, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// flash_adam_cuda.cuh
// Shared device helpers and kernel declarations for the fused Adam CUDA kernel.
//

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>  // for std::min (host-side use in encode_ecc)

// Number of elements loaded per thread per iteration (128-bit vector load).
// 4 × float32 = 16 bytes = one 128-bit transaction.
static constexpr int VEC = 4;

#ifdef __CUDACC__

template <typename T> struct ParamTypeTraits {};

template <> struct ParamTypeTraits<__nv_bfloat16> {
    static constexpr int mantissa_bits = 7;
    static constexpr int exponent_bits = 8;
    static constexpr int exponent_bias = 127;
    using uint_type = uint16_t;
    __device__ static uint16_t bitcast_to_uint(__nv_bfloat16 x) {
        return *reinterpret_cast<const uint16_t*>(&x);
    }
};

template <> struct ParamTypeTraits<__half> {
    static constexpr int mantissa_bits = 10;
    static constexpr int exponent_bits = 5;
    static constexpr int exponent_bias = 15;
    using uint_type = uint16_t;
    __device__ static uint16_t bitcast_to_uint(__half x) {
        return *reinterpret_cast<const uint16_t*>(&x);
    }
};

template <> struct ParamTypeTraits<float> {
    static constexpr int mantissa_bits = 23;
    static constexpr int exponent_bits = 8;
    static constexpr int exponent_bias = 127;
    using uint_type = uint32_t;
    __device__ static uint32_t bitcast_to_uint(float x) {
        uint32_t u;
        __builtin_memcpy(&u, &x, sizeof(u));  // avoids __float_as_uint which requires CUDA intrinsics
        return u;
    }
};

// ---------------------------------------------------------------------------
// Type-safe float ↔ ParamT conversion helpers
// ---------------------------------------------------------------------------

/// Convert float → ParamT without relying on implicit __half constructor.
template <typename ParamT>
__device__ __forceinline__ ParamT float_to_param(float x);

template <>
__device__ __forceinline__ __nv_bfloat16 float_to_param<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

template <>
__device__ __forceinline__ __half float_to_param<__half>(float x) {
    return __float2half(x);
}

template <>
__device__ __forceinline__ float float_to_param<float>(float x) {
    return x;
}

/// Convert ParamT → float without relying on implicit __half cast.
template <typename ParamT>
__device__ __forceinline__ float param_to_float(ParamT x);

template <>
__device__ __forceinline__ float param_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <>
__device__ __forceinline__ float param_to_float<__half>(__half x) {
    return __half2float(x);
}

template <>
__device__ __forceinline__ float param_to_float<float>(float x) {
    return x;
}

/// Warp-wide absolute maximum reduction.
__device__ __forceinline__ float warp_absmax(float val) {
    val = fabsf(val);
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, offset));
    return val;
}

/// Warp-wide maximum (for values that are always >= 0).
__device__ __forceinline__ float warp_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, offset));
    return val;
}

__device__ __forceinline__ float softsign(float x) {
    return 2.f * x / (1.f + fabsf(x));
}

__device__ __forceinline__ float inv_softsign(float y) {
    return y / (2.f - fabsf(y));
}

template <typename ParamT>
__device__ __forceinline__ int get_unbiased_exponent(ParamT x) {
    using Traits = ParamTypeTraits<ParamT>;
    // abs via negation to avoid __CUDA_NO_HALF_OPERATORS__ issues
    ParamT ax = (param_to_float(x) < 0.f) ? float_to_param<ParamT>(-param_to_float(x)) : x;
    auto bits = Traits::bitcast_to_uint(ax);
    int exp_bits = (int)(bits >> Traits::mantissa_bits);
    return (exp_bits == 0) ? (1 - Traits::exponent_bias)
                           : (exp_bits - Traits::exponent_bias);
}

template <typename ParamT>
__device__ __forceinline__ int log_ulp(ParamT x) {
    return get_unbiased_exponent(x) - ParamTypeTraits<ParamT>::mantissa_bits;
}

template <typename ParamT, int kECCBits>
__device__ __forceinline__ int encode_ecc(float x_f32, ParamT x_narrow) {
    constexpr int signed_max = (kECCBits == 8) ? 127 : 32767;
    float x_recon = param_to_float(x_narrow);
    float e = x_f32 - x_recon;

    int ls = log_ulp(x_narrow) - 1;
    float neg_ls = (float)(-ls);
    float h = floorf(neg_ls * 0.5f);
    float temp = e * exp2f(h);
    float e_norm = temp * exp2f(neg_ls - h);
    float e_clamped = fmaxf(-1.f, fminf(1.f, e_norm));
    float scaled = e_clamped * (float)signed_max;

    float sign = (scaled >= 0.f) ? 1.f : -1.f;
    int rounded = (int)(fabsf(scaled) + 0.5f);
    rounded = min(rounded, signed_max);
    return (int)(sign * (float)rounded);
}

template <typename ParamT, int kECCBits>
__device__ __forceinline__ float decode_ecc(ParamT x_narrow, int ecc_val) {
    constexpr int signed_max = (kECCBits == 8) ? 127 : 32767;
    float x_recon = param_to_float(x_narrow);
    int ls = log_ulp(x_narrow) - 1;
    float log_scale_f = (float)ls;
    float h = floorf(log_scale_f * 0.5f);
    float correction = ((float)ecc_val / (float)signed_max) * exp2f(h) * exp2f(log_scale_f - h);
    return x_recon + correction;
}

#endif  // __CUDACC__

template <
    typename ParamT,
    int  GROUP_SIZE,
    int  BLOCK_SIZE,
    bool kQuantize,
    bool kDecoupled,
    bool kUseECC,
    int  kECCBits
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
);

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
);
