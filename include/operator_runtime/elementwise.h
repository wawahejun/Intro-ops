#pragma once

#include "operator_runtime/tensor_view.h"

#ifdef __cplusplus

namespace oprt {

inline bool elementwise_fast_path(const oprt_tensor_view_t &out,
                                  const oprt_tensor_view_t &a) {
    return same_shape(out, a) && is_contiguous(out) && is_contiguous(a);
}

inline bool elementwise_fast_path(const oprt_tensor_view_t &out,
                                  const oprt_tensor_view_t &a,
                                  const oprt_tensor_view_t &b) {
    return same_shape(out, a) && same_shape(out, b) &&
           is_contiguous(out) && is_contiguous(a) && is_contiguous(b);
}

} // namespace oprt

#endif

