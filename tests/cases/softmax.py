from __future__ import annotations

import torch


def correctness_cases():
    return [
        {"name": "rowwise_32x128", "shape": (32, 128), "dtype": torch.float32, "atol": 1e-5, "rtol": 1e-5},
    ]


def api_error_cases():
    return [
        {"name": "wrong_dim", "shape": (16, 16), "dtype": torch.float32, "dim": 0},
        {"name": "wrong_dtype", "shape": (16, 16), "dtype": torch.float16, "dim": 1},
    ]


def benchmark_cases():
    return [
        {"name": "rowwise_1024x1024", "shape": (1024, 1024), "dtype": torch.float32},
    ]

