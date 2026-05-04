from __future__ import annotations

import ctypes

import torch

from operator_runtime._internal import CFunctions, Descriptor, PreparedOp, check_status


def build_prepared_op(
    funcs: CFunctions,
    create_args: tuple[object, ...],
    tensors: tuple[torch.Tensor, ...],
    stream_tensor: torch.Tensor,
) -> PreparedOp:
    desc = Descriptor()
    check_status(funcs.create(ctypes.byref(desc), *create_args))
    workspace_size = ctypes.c_size_t()
    check_status(funcs.workspace(desc, ctypes.byref(workspace_size)))
    workspace = None
    if workspace_size.value:
        workspace = torch.empty(workspace_size.value, dtype=torch.uint8, device=stream_tensor.device)
    runner_args = tuple(ctypes.c_void_p(tensor.data_ptr()) for tensor in tensors)
    return PreparedOp(funcs, desc, workspace, runner_args, stream_tensor)
