from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Callable

from operator_runtime.backend import Backend

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


def _missing_symbol_error(name: str, symbol: str, exc: AttributeError) -> OperatorRuntimeError:
    raise OperatorRuntimeError(
        f"missing symbol {symbol} for operator {name}"
    ) from exc


def bind_unary(name: str, backend: str | Backend = Backend.NVIDIA) -> CFunctions:
    lib = load_library()
    create_symbol = f"oprt_create_{name}_descriptor"
    try:
        create = getattr(lib, create_symbol)
    except AttributeError as exc:
        _missing_symbol_error(name, create_symbol, exc)
    create.argtypes = [ctypes.POINTER(Descriptor), ctypes.POINTER(TensorView), ctypes.POINTER(TensorView)]
    create.restype = Status

    workspace_symbol = f"oprt_get_{name}_workspace_size"
    try:
        workspace = getattr(lib, workspace_symbol)
    except AttributeError as exc:
        _missing_symbol_error(name, workspace_symbol, exc)
    workspace.argtypes = [Descriptor, ctypes.POINTER(ctypes.c_size_t)]
    workspace.restype = Status

    execute_symbol = f"oprt_execute_{name}"
    try:
        execute = getattr(lib, execute_symbol)
    except AttributeError as exc:
        _missing_symbol_error(name, execute_symbol, exc)
    execute.argtypes = [
        Descriptor,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    execute.restype = Status

    destroy_symbol = f"oprt_destroy_{name}_descriptor"
    try:
        destroy = getattr(lib, destroy_symbol)
    except AttributeError as exc:
        _missing_symbol_error(name, destroy_symbol, exc)
    destroy.argtypes = [Descriptor]
    destroy.restype = Status
    return CFunctions(create, workspace, execute, destroy)


def bind_relu(backend: str | Backend = Backend.NVIDIA) -> CFunctions:
    lib = load_library()
    name = "relu"
    create_symbol = "oprt_create_relu_descriptor"
    try:
        create = getattr(lib, create_symbol)
    except AttributeError as exc:
        _missing_symbol_error(name, create_symbol, exc)
    create.argtypes = [
        ctypes.POINTER(Descriptor),
        ctypes.POINTER(TensorView),
        ctypes.POINTER(TensorView),
        ctypes.c_float,
    ]
    create.restype = Status

    workspace_symbol = "oprt_get_relu_workspace_size"
    try:
        workspace = getattr(lib, workspace_symbol)
    except AttributeError as exc:
        _missing_symbol_error(name, workspace_symbol, exc)
    workspace.argtypes = [Descriptor, ctypes.POINTER(ctypes.c_size_t)]
    workspace.restype = Status

    execute_symbol = "oprt_execute_relu"
    try:
        execute = getattr(lib, execute_symbol)
    except AttributeError as exc:
        _missing_symbol_error(name, execute_symbol, exc)
    execute.argtypes = [
        Descriptor,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    execute.restype = Status

    destroy_symbol = "oprt_destroy_relu_descriptor"
    try:
        destroy = getattr(lib, destroy_symbol)
    except AttributeError as exc:
        _missing_symbol_error(name, destroy_symbol, exc)
    destroy.argtypes = [Descriptor]
    destroy.restype = Status
    return CFunctions(create, workspace, execute, destroy)


def bind_binary(name: str, backend: str | Backend = Backend.NVIDIA) -> CFunctions:
    lib = load_library()
    create_symbol = f"oprt_create_{name}_descriptor"
    try:
        create = getattr(lib, create_symbol)
    except AttributeError as exc:
        _missing_symbol_error(name, create_symbol, exc)
    create.argtypes = [
        ctypes.POINTER(Descriptor),
        ctypes.POINTER(TensorView),
        ctypes.POINTER(TensorView),
        ctypes.POINTER(TensorView),
    ]
    create.restype = Status

    workspace_symbol = f"oprt_get_{name}_workspace_size"
    try:
        workspace = getattr(lib, workspace_symbol)
    except AttributeError as exc:
        _missing_symbol_error(name, workspace_symbol, exc)
    workspace.argtypes = [Descriptor, ctypes.POINTER(ctypes.c_size_t)]
    workspace.restype = Status

    execute_symbol = f"oprt_execute_{name}"
    try:
        execute = getattr(lib, execute_symbol)
    except AttributeError as exc:
        _missing_symbol_error(name, execute_symbol, exc)
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

    destroy_symbol = f"oprt_destroy_{name}_descriptor"
    try:
        destroy = getattr(lib, destroy_symbol)
    except AttributeError as exc:
        _missing_symbol_error(name, destroy_symbol, exc)
    destroy.argtypes = [Descriptor]
    destroy.restype = Status
    return CFunctions(create, workspace, execute, destroy)


def bind_reduce_like(name: str, backend: str | Backend = Backend.NVIDIA) -> CFunctions:
    lib = load_library()
    create_symbol = f"oprt_create_{name}_descriptor"
    try:
        create = getattr(lib, create_symbol)
    except AttributeError as exc:
        _missing_symbol_error(name, create_symbol, exc)
    create.argtypes = [
        ctypes.POINTER(Descriptor),
        ctypes.POINTER(TensorView),
        ctypes.POINTER(TensorView),
        ctypes.c_int64,
    ]
    create.restype = Status

    workspace_symbol = f"oprt_get_{name}_workspace_size"
    try:
        workspace = getattr(lib, workspace_symbol)
    except AttributeError as exc:
        _missing_symbol_error(name, workspace_symbol, exc)
    workspace.argtypes = [Descriptor, ctypes.POINTER(ctypes.c_size_t)]
    workspace.restype = Status

    execute_symbol = f"oprt_execute_{name}"
    try:
        execute = getattr(lib, execute_symbol)
    except AttributeError as exc:
        _missing_symbol_error(name, execute_symbol, exc)
    execute.argtypes = [
        Descriptor,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    execute.restype = Status

    destroy_symbol = f"oprt_destroy_{name}_descriptor"
    try:
        destroy = getattr(lib, destroy_symbol)
    except AttributeError as exc:
        _missing_symbol_error(name, destroy_symbol, exc)
    destroy.argtypes = [Descriptor]
    destroy.restype = Status
    return CFunctions(create, workspace, execute, destroy)
