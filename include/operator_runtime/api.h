#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
#define OPRT_EXPORT __declspec(dllexport)
#else
#define OPRT_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OPRT_SUCCESS = 0,
    OPRT_ERR_INVALID_ARG = 1,
    OPRT_ERR_UNSUPPORTED_DTYPE = 2,
    OPRT_ERR_RUNTIME = 3,
    OPRT_ERR_INSUFFICIENT_WORKSPACE = 4,
    OPRT_ERR_NOT_SUPPORTED = 5
} oprt_status_t;

typedef enum {
    OPRT_DTYPE_F16 = 0,
    OPRT_DTYPE_F32 = 1
} oprt_dtype_t;

#define OPRT_MAX_DIMS 8

typedef void *oprt_stream_t;
typedef struct oprt_operator_descriptor *oprt_operator_descriptor_t;

typedef struct {
    void *data;
    oprt_dtype_t dtype;
    int32_t ndim;
    int64_t shape[OPRT_MAX_DIMS];
    int64_t strides[OPRT_MAX_DIMS];
} oprt_tensor_view_t;

OPRT_EXPORT const char *oprt_status_string(oprt_status_t status);

#ifdef __cplusplus
}
#endif

