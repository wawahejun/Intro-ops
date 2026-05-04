from __future__ import annotations

import ctypes

import torch

from operator_runtime.backend import Backend, normalize_backend
from operator_runtime._runtime import Descriptor, bind_binary, check_status, PreparedOp, tensor_view


def _check(out: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    if not out.is_cuda or not a.is_cuda or not b.is_cuda:
        raise ValueError("vector_add expects CUDA tensors")
    if out.shape != a.shape or out.shape != b.shape:
        raise ValueError("vector_add v1 expects matching shapes")
    if out.dtype != a.dtype or out.dtype != b.dtype:
        raise TypeError("vector_add expects matching dtypes")
    if not out.is_contiguous() or not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("vector_add v1 supports contiguous tensors only")


def prepare_vector_add(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    backend: str | Backend = Backend.NVIDIA,
) -> PreparedOp:
    backend = normalize_backend(backend)
    _check(out, a, b)
    if backend is Backend.TILELANG:
        from ops.vector_add.tilelang.vector_add_tl import prepare_vector_add_tl

        return prepare_vector_add_tl(out, a, b)
    if backend is not Backend.NVIDIA:
        raise NotImplementedError(f"backend {backend.value} is not runnable")

    funcs = bind_binary("vector_add")
    desc = Descriptor()
    out_view = tensor_view(out)
    a_view = tensor_view(a)
    b_view = tensor_view(b)
    check_status(funcs.create(ctypes.byref(desc), ctypes.byref(out_view), ctypes.byref(a_view), ctypes.byref(b_view)))
    workspace_size = ctypes.c_size_t()
    check_status(funcs.workspace(desc, ctypes.byref(workspace_size)))
    workspace = torch.empty(workspace_size.value, dtype=torch.uint8, device=out.device) if workspace_size.value else None
    args = (ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(b.data_ptr()))
    return PreparedOp(funcs, desc, workspace, args, out)


def vector_add_(out: torch.Tensor, a: torch.Tensor, b: torch.Tensor, backend: str | Backend = Backend.NVIDIA) -> torch.Tensor:
    with prepare_vector_add(out, a, b, backend) as prepared:
        prepared.run()
    return out


def vector_add(a: torch.Tensor, b: torch.Tensor, backend: str | Backend = Backend.NVIDIA) -> torch.Tensor:
    out = torch.empty_like(a)
    return vector_add_(out, a, b, backend)

