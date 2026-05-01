#pragma once

#include <stdint.h>

namespace oprt::nvidia {

__device__ inline int64_t contiguous_offset(int64_t linear) {
    return linear;
}

} // namespace oprt::nvidia

