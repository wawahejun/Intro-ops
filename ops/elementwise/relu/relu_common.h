#pragma once

#include "operator_runtime/descriptor.h"
#include "operator_runtime/detail/elementwise.h"

namespace oprt::elementwise {

struct ReluDescriptor final : oprt_operator_descriptor {
    oprt_tensor_view_t out_view{};
    oprt_tensor_view_t in_view{};
    oprt::ElementwiseInfo info;
    float negative_slope = 0.0f;

    const char *op_name() const override {
        return "relu";
    }
};

inline oprt_status_t create_relu_descriptor_common(
    oprt_operator_descriptor_t *desc,
    const oprt_tensor_view_t *out,
    const oprt_tensor_view_t *in,
    float negative_slope) {
    if (desc == nullptr) {
        return OPRT_ERR_INVALID_ARG;
    }
    *desc = nullptr;

    if (out == nullptr || in == nullptr) {
        return OPRT_ERR_INVALID_ARG;
    }

    oprt::ElementwiseInfo info;
    auto status = oprt::create_elementwise_info(out, {in}, &info);
    if (status != OPRT_SUCCESS) {
        return status;
    }

    auto *typed = new ReluDescriptor();
    typed->out_view = *out;
    typed->in_view = *in;
    typed->info = std::move(info);
    typed->negative_slope = negative_slope;
    typed->workspace_size = typed->info.workspace_bytes();
    *desc = typed;
    return OPRT_SUCCESS;
}

inline oprt_status_t get_relu_workspace_size_common(oprt_operator_descriptor_t desc,
                                                    size_t *size) {
    if (desc == nullptr || size == nullptr) {
        return OPRT_ERR_INVALID_ARG;
    }
    *size = desc->workspace_size;
    return OPRT_SUCCESS;
}

inline oprt_status_t validate_relu_execute_args(oprt_operator_descriptor_t desc,
                                                size_t workspace_size,
                                                void *out,
                                                const void *in) {
    if (desc == nullptr || out == nullptr || in == nullptr) {
        return OPRT_ERR_INVALID_ARG;
    }
    if (workspace_size < desc->workspace_size) {
        return OPRT_ERR_INSUFFICIENT_WORKSPACE;
    }
    return OPRT_SUCCESS;
}

inline oprt_status_t destroy_relu_descriptor_common(oprt_operator_descriptor_t desc) {
    delete desc;
    return OPRT_SUCCESS;
}

} // namespace oprt::elementwise
