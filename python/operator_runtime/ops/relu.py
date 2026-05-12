from __future__ import annotations

import ctypes

import torch

from operator_runtime.backend import Backend, normalize_backend
from operator_runtime._internal import PreparedOp, bind_relu, tensor_view
from operator_runtime.ops._common import build_prepared_op


def _broadcastable_to(src_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> bool:
    if len(src_shape) > len(out_shape):
        return False
    offset = len(out_shape) - len(src_shape)
    for out_dim, out_extent in enumerate(out_shape):
        src_dim = out_dim - offset
        if src_dim < 0:
            continue
        src_extent = src_shape[src_dim]
        if src_extent != out_extent and src_extent != 1:
            return False
    return True


def _has_broadcast_dim(tensor: torch.Tensor) -> bool:
    return any(size > 1 and stride == 0 for size, stride in zip(tensor.shape, tensor.stride()))


def _check(out: torch.Tensor, src: torch.Tensor) -> None:
    if not out.is_cuda or not src.is_cuda:
        raise ValueError("relu expects CUDA tensors")
    if _has_broadcast_dim(out):
        raise ValueError("relu expects a writable output tensor")
    if not _broadcastable_to(tuple(src.shape), tuple(out.shape)):
        raise ValueError("relu expects src to be broadcastable to out")
    if out.dtype != src.dtype:
        raise TypeError("relu expects matching dtypes")


def prepare_relu(
    out: torch.Tensor,
    src: torch.Tensor,
    negative_slope: float = 0.0,
    backend: str | Backend = Backend.NVIDIA,
) -> PreparedOp:
    backend = normalize_backend(backend)
    _check(out, src)
    if backend not in (Backend.NVIDIA, Backend.METAX):
        raise NotImplementedError(f"backend {backend.value} is not runnable")

    funcs = bind_relu(backend)
    out_view = tensor_view(out)
    src_view = tensor_view(src)
    create_args = (
        ctypes.byref(out_view),
        ctypes.byref(src_view),
        ctypes.c_float(float(negative_slope)),
    )
    return build_prepared_op(funcs, create_args, (out, src), out)


def relu_(
    out: torch.Tensor,
    src: torch.Tensor,
    negative_slope: float = 0.0,
    backend: str | Backend = Backend.NVIDIA,
) -> torch.Tensor:
    with prepare_relu(out, src, negative_slope, backend) as prepared:
        prepared.run()
    return out


def relu(
    src: torch.Tensor,
    negative_slope: float = 0.0,
    backend: str | Backend = Backend.NVIDIA,
) -> torch.Tensor:
    out = torch.empty_like(src)
    return relu_(out, src, negative_slope, backend)
