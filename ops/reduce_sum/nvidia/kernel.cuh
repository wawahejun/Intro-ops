#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace oprt::reduce_sum::nvidia {

__global__ void reduce_sum_rowwise_kernel(float *out, const float *in, int64_t rows, int64_t cols) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    float sum = 0.0f;
    for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
        sum += in[int64_t(row) * cols + col];
    }
    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[row] = smem[0];
    }
}

} // namespace oprt::reduce_sum::nvidia
