#pragma once

#include <stdint.h>

namespace oprt::copy::nvidia {

template <typename T>
__global__ void copy_contiguous_kernel(T *dst, const T *src, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

} // namespace oprt::copy::nvidia
