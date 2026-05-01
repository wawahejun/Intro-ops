from __future__ import annotations

import torch


def elem_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def estimate_copy(tensor: torch.Tensor) -> tuple[int, int]:
    return 2 * tensor.numel() * elem_bytes(tensor.dtype), 0


def estimate_vector_add(tensor: torch.Tensor) -> tuple[int, int]:
    return 3 * tensor.numel() * elem_bytes(tensor.dtype), tensor.numel()


def estimate_reduce_sum(tensor: torch.Tensor) -> tuple[int, int]:
    rows = tensor.shape[0]
    return (tensor.numel() + rows) * elem_bytes(tensor.dtype), tensor.numel()


def estimate_softmax(tensor: torch.Tensor) -> tuple[int, int]:
    return 5 * tensor.numel() * elem_bytes(tensor.dtype), 4 * tensor.numel()

