#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace oprt::nvidia {

template <typename T, typename UnaryOp>
__global__ void unary_contiguous_kernel(T *out, const T *in, int64_t n, UnaryOp op) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        out[i] = op(in[i]);
    }
}

template <typename T, typename BinaryOp>
__global__ void binary_contiguous_kernel(T *out, const T *a, const T *b, int64_t n, BinaryOp op) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        out[i] = op(a[i], b[i]);
    }
}

struct CopyOp {
    template <typename T>
    __device__ T operator()(T value) const {
        return value;
    }
};

struct AddOp {
    __device__ float operator()(float a, float b) const {
        return a + b;
    }

    __device__ half operator()(half a, half b) const {
        return __hadd(a, b);
    }
};

} // namespace oprt::nvidia

