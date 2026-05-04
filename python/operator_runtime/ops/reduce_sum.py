from __future__ import annotations

import ctypes

import torch

from operator_runtime.backend import Backend, normalize_backend
from operator_runtime._runtime import Descriptor, bind_reduce_like, check_status, PreparedOp, tensor_view


def _check(out: torch.Tensor, src: torch.Tensor, dim: int) -> None:
    if not out.is_cuda or not src.is_cuda:
        raise ValueError("reduce_sum expects CUDA tensors")
    if src.dtype is not torch.float32 or out.dtype is not torch.float32:
        raise TypeError("reduce_sum v1 supports float32 only")
    if src.ndim != 2 or out.ndim != 1 or dim != 1:
        raise ValueError("reduce_sum v1 supports 2D row-wise reduction over dim=1")
    if out.shape[0] != src.shape[0]:
        raise ValueError("reduce_sum output shape must be [src.shape[0]]")
    if not out.is_contiguous() or not src.is_contiguous():
        raise ValueError("reduce_sum v1 supports contiguous tensors only")


def prepare_reduce_sum(
    out: torch.Tensor,
    src: torch.Tensor,
    dim: int = 1,
    backend: str | Backend = Backend.NVIDIA,
) -> PreparedOp:
    backend = normalize_backend(backend)
    _check(out, src, dim)
    if backend is Backend.TILELANG:
        from ops.reduce_sum.tilelang.reduce_sum_tl import prepare_reduce_sum_tl

        return prepare_reduce_sum_tl(out, src, dim=dim)
    if backend is not Backend.NVIDIA:
        raise NotImplementedError(f"backend {backend.value} is not runnable")

    funcs = bind_reduce_like("reduce_sum")
    desc = Descriptor()
    out_view = tensor_view(out)
    src_view = tensor_view(src)
    check_status(funcs.create(ctypes.byref(desc), ctypes.byref(out_view), ctypes.byref(src_view), ctypes.c_int64(dim)))
    workspace_size = ctypes.c_size_t()
    check_status(funcs.workspace(desc, ctypes.byref(workspace_size)))
    workspace = torch.empty(workspace_size.value, dtype=torch.uint8, device=out.device) if workspace_size.value else None
    args = (ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(src.data_ptr()))
    return PreparedOp(funcs, desc, workspace, args, out)


def reduce_sum_(out: torch.Tensor, src: torch.Tensor, dim: int = 1, backend: str | Backend = Backend.NVIDIA) -> torch.Tensor:
    with prepare_reduce_sum(out, src, dim, backend) as prepared:
        prepared.run()
    return out


def reduce_sum(src: torch.Tensor, dim: int = 1, backend: str | Backend = Backend.NVIDIA) -> torch.Tensor:
    if dim != 1 or src.ndim != 2:
        raise ValueError("reduce_sum v1 supports 2D row-wise reduction over dim=1")
    out = torch.empty((src.shape[0],), dtype=src.dtype, device=src.device)
    return reduce_sum_(out, src, dim, backend)

