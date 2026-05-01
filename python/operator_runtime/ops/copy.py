from __future__ import annotations

import ctypes

import torch

from operator_runtime.backend import Backend, normalize_backend
from operator_runtime.ctypes_bindings import Descriptor, bind_unary, check_status
from operator_runtime.prepared import PreparedOp
from operator_runtime.tensor_view import tensor_view


def _check(out: torch.Tensor, src: torch.Tensor) -> None:
    if not out.is_cuda or not src.is_cuda:
        raise ValueError("copy expects CUDA tensors")
    if out.shape != src.shape:
        raise ValueError("copy expects matching shapes")
    if out.dtype != src.dtype:
        raise TypeError("copy expects matching dtypes")
    if not out.is_contiguous() or not src.is_contiguous():
        raise ValueError("copy v1 supports contiguous tensors only")


def prepare_copy(out: torch.Tensor, src: torch.Tensor, backend: str | Backend = Backend.NVIDIA) -> PreparedOp:
    backend = normalize_backend(backend)
    _check(out, src)
    if backend is Backend.TILELANG:
        from ops.copy.tilelang.copy_tl import prepare_copy_tl

        return prepare_copy_tl(out, src)
    if backend is not Backend.NVIDIA:
        raise NotImplementedError(f"backend {backend.value} is not runnable")

    funcs = bind_unary("copy")
    desc = Descriptor()
    out_view = tensor_view(out)
    src_view = tensor_view(src)
    check_status(funcs.create(ctypes.byref(desc), ctypes.byref(out_view), ctypes.byref(src_view)))
    workspace_size = ctypes.c_size_t()
    check_status(funcs.workspace(desc, ctypes.byref(workspace_size)))
    workspace = torch.empty(workspace_size.value, dtype=torch.uint8, device=out.device) if workspace_size.value else None
    args = (ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(src.data_ptr()))
    return PreparedOp(funcs, desc, workspace, args, out)


def copy_(out: torch.Tensor, src: torch.Tensor, backend: str | Backend = Backend.NVIDIA) -> torch.Tensor:
    with prepare_copy(out, src, backend) as prepared:
        prepared.run()
    return out


def copy(src: torch.Tensor, backend: str | Backend = Backend.NVIDIA) -> torch.Tensor:
    out = torch.empty_like(src)
    return copy_(out, src, backend)

