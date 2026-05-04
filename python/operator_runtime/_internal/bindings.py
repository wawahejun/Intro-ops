from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Callable

from .loader import load_library
from .tensor_view import TensorView

Status = ctypes.c_int
Descriptor = ctypes.c_void_p


class OperatorRuntimeError(RuntimeError):
    pass


def check_status(status: int) -> None:
    if status == 0:
        return
    lib = load_library()
    lib.oprt_status_string.argtypes = [ctypes.c_int]
    lib.oprt_status_string.restype = ctypes.c_char_p
    message = lib.oprt_status_string(status).decode("utf-8")
    raise OperatorRuntimeError(message)


@dataclass(frozen=True)
class CFunctions:
    create: Callable
    workspace: Callable
    execute: Callable
    destroy: Callable


def bind_unary(name: str) -> CFunctions:
    lib = load_library()
    create = getattr(lib, f"oprt_create_{name}_descriptor_nvidia")
    create.argtypes = [ctypes.POINTER(Descriptor), ctypes.POINTER(TensorView), ctypes.POINTER(TensorView)]
    create.restype = Status

    workspace = getattr(lib, f"oprt_get_{name}_workspace_size_nvidia")
    workspace.argtypes = [Descriptor, ctypes.POINTER(ctypes.c_size_t)]
    workspace.restype = Status

    execute = getattr(lib, f"oprt_execute_{name}_nvidia")
    execute.argtypes = [
        Descriptor,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    execute.restype = Status

    destroy = getattr(lib, f"oprt_destroy_{name}_descriptor_nvidia")
    destroy.argtypes = [Descriptor]
    destroy.restype = Status
    return CFunctions(create, workspace, execute, destroy)


def bind_binary(name: str) -> CFunctions:
    lib = load_library()
    create = getattr(lib, f"oprt_create_{name}_descriptor_nvidia")
    create.argtypes = [
        ctypes.POINTER(Descriptor),
        ctypes.POINTER(TensorView),
        ctypes.POINTER(TensorView),
        ctypes.POINTER(TensorView),
    ]
    create.restype = Status

    workspace = getattr(lib, f"oprt_get_{name}_workspace_size_nvidia")
    workspace.argtypes = [Descriptor, ctypes.POINTER(ctypes.c_size_t)]
    workspace.restype = Status

    execute = getattr(lib, f"oprt_execute_{name}_nvidia")
    execute.argtypes = [
        Descriptor,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    execute.restype = Status

    destroy = getattr(lib, f"oprt_destroy_{name}_descriptor_nvidia")
    destroy.argtypes = [Descriptor]
    destroy.restype = Status
    return CFunctions(create, workspace, execute, destroy)


def bind_reduce_like(name: str) -> CFunctions:
    lib = load_library()
    create = getattr(lib, f"oprt_create_{name}_descriptor_nvidia")
    create.argtypes = [
        ctypes.POINTER(Descriptor),
        ctypes.POINTER(TensorView),
        ctypes.POINTER(TensorView),
        ctypes.c_int64,
    ]
    create.restype = Status

    workspace = getattr(lib, f"oprt_get_{name}_workspace_size_nvidia")
    workspace.argtypes = [Descriptor, ctypes.POINTER(ctypes.c_size_t)]
    workspace.restype = Status

    execute = getattr(lib, f"oprt_execute_{name}_nvidia")
    execute.argtypes = [
        Descriptor,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    execute.restype = Status

    destroy = getattr(lib, f"oprt_destroy_{name}_descriptor_nvidia")
    destroy.argtypes = [Descriptor]
    destroy.restype = Status
    return CFunctions(create, workspace, execute, destroy)
