#include "operator_runtime/api.h"

extern "C" OPRT_EXPORT const char *oprt_status_string(oprt_status_t status) {
    switch (status) {
    case OPRT_SUCCESS:
        return "success";
    case OPRT_ERR_INVALID_ARG:
        return "invalid argument";
    case OPRT_ERR_UNSUPPORTED_DTYPE:
        return "unsupported dtype";
    case OPRT_ERR_RUNTIME:
        return "runtime error";
    case OPRT_ERR_INSUFFICIENT_WORKSPACE:
        return "insufficient workspace";
    case OPRT_ERR_NOT_SUPPORTED:
        return "not supported";
    default:
        return "unknown status";
    }
}

