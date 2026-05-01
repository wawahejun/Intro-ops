#pragma once

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

namespace oprt::softmax::nvidia {

__global__ void softmax_rowwise_kernel(float *out, const float *in, int64_t rows, int64_t cols) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    float local_max = -FLT_MAX;
    for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
        float value = in[int64_t(row) * cols + col];
        local_max = fmaxf(local_max, value);
    }
    smem[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float row_max = smem[0];

    float local_sum = 0.0f;
    for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
        float value = expf(in[int64_t(row) * cols + col] - row_max);
        out[int64_t(row) * cols + col] = value;
        local_sum += value;
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float row_sum = smem[0];

    for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
        out[int64_t(row) * cols + col] /= row_sum;
    }
}

} // namespace oprt::softmax::nvidia
