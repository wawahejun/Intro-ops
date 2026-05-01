#include "ops/copy/nvidia/copy_cuda.h"

#include "operator_runtime/descriptor.h"
#include "operator_runtime/elementwise.h"
#include "operator_runtime/tensor_checks.h"
#include "operator_runtime/cuda_helpers.h"
#include "ops/common/nvidia/elementwise.cuh"

#include <cuda_fp16.h>

namespace {

struct CopyDescriptor final : oprt_operator_descriptor {
    oprt_tensor_view_t dst_view;
    oprt_tensor_view_t src_view;
    int64_t elements = 0;
    bool fast_path = false;

    const char *op_name() const override {
        return "copy";
    }
};

template <typename T>
oprt_status_t launch_copy(const CopyDescriptor *desc, void *dst, const void *src, oprt_stream_t stream) {
    constexpr int threads = 256;
    int blocks = oprt::blocks_for(desc->elements, threads);
    oprt::nvidia::unary_contiguous_kernel<T><<<blocks, threads, 0, oprt::as_cuda_stream(stream)>>>(
        static_cast<T *>(dst), static_cast<const T *>(src), desc->elements, oprt::nvidia::CopyOp{});
    OPRT_CUDA_RETURN_IF_ERROR(cudaGetLastError());
    return OPRT_SUCCESS;
}

} // namespace

extern "C" OPRT_EXPORT oprt_status_t oprt_create_copy_descriptor_nvidia(
    oprt_operator_descriptor_t *desc,
    const oprt_tensor_view_t *dst,
    const oprt_tensor_view_t *src) {
    if (desc == nullptr) {
        return OPRT_ERR_INVALID_ARG;
    }
    *desc = nullptr;
    auto status = oprt::check_tensor(dst);
    if (status != OPRT_SUCCESS) {
        return status;
    }
    status = oprt::check_tensor(src);
    if (status != OPRT_SUCCESS) {
        return status;
    }
    status = oprt::check_same_dtype(*dst, *src);
    if (status != OPRT_SUCCESS) {
        return status;
    }
    status = oprt::check_same_shape(*dst, *src);
    if (status != OPRT_SUCCESS) {
        return status;
    }
    if (!oprt::elementwise_fast_path(*dst, *src)) {
        return OPRT_ERR_NOT_SUPPORTED;
    }

    auto *typed = new CopyDescriptor();
    typed->dst_view = *dst;
    typed->src_view = *src;
    typed->elements = oprt::numel(*dst);
    typed->fast_path = true;
    typed->workspace_size = 0;
    *desc = typed;
    return OPRT_SUCCESS;
}

extern "C" OPRT_EXPORT oprt_status_t oprt_get_copy_workspace_size_nvidia(
    oprt_operator_descriptor_t desc,
    size_t *size) {
    if (desc == nullptr || size == nullptr) {
        return OPRT_ERR_INVALID_ARG;
    }
    *size = desc->workspace_size;
    return OPRT_SUCCESS;
}

extern "C" OPRT_EXPORT oprt_status_t oprt_execute_copy_nvidia(
    oprt_operator_descriptor_t desc,
    void *,
    size_t workspace_size,
    void *dst,
    const void *src,
    oprt_stream_t stream) {
    if (desc == nullptr || dst == nullptr || src == nullptr) {
        return OPRT_ERR_INVALID_ARG;
    }
    if (workspace_size < desc->workspace_size) {
        return OPRT_ERR_INSUFFICIENT_WORKSPACE;
    }
    auto *typed = static_cast<const CopyDescriptor *>(desc);
    switch (typed->dst_view.dtype) {
    case OPRT_DTYPE_F16:
        return launch_copy<half>(typed, dst, src, stream);
    case OPRT_DTYPE_F32:
        return launch_copy<float>(typed, dst, src, stream);
    default:
        return OPRT_ERR_UNSUPPORTED_DTYPE;
    }
}

extern "C" OPRT_EXPORT oprt_status_t oprt_destroy_copy_descriptor_nvidia(
    oprt_operator_descriptor_t desc) {
    delete desc;
    return OPRT_SUCCESS;
}
