from __future__ import annotations

import torch


def correctness_cases():
    return [
        {"name": "rowwise_1x128", "shape": (1, 128), "dtype": torch.float32, "atol": 1e-5, "rtol": 1e-5},
        {"name": "rowwise_16x128", "shape": (16, 128), "dtype": torch.float32, "atol": 1e-5, "rtol": 1e-5},
        {"name": "rowwise_16x256", "shape": (16, 256), "dtype": torch.float32, "atol": 1e-5, "rtol": 1e-5},
        {"name": "rowwise_32x128", "shape": (32, 128), "dtype": torch.float32, "atol": 1e-5, "rtol": 1e-5},
    ]


def api_error_cases():
    return [
        {"name": "wrong_dim", "shape": (16, 16), "dtype": torch.float32, "dim": 0},
        {"name": "wrong_dtype", "shape": (16, 16), "dtype": torch.float16, "dim": 1},
        {"name": "wrong_output_shape", "shape": (16, 16), "dtype": torch.float32, "out_shape": (15,), "dim": 1},
        {"name": "non_contiguous", "shape": (16, 16), "dtype": torch.float32, "dim": 1},
        {"name": "wrong_rank", "shape": (16,), "dtype": torch.float32, "dim": 1},
    ]


def benchmark_cases():
    return [
        {"name": "rowwise_128x1024", "shape": (128, 1024), "dtype": torch.float32},
        {"name": "rowwise_1024x1024", "shape": (1024, 1024), "dtype": torch.float32},
        {"name": "rowwise_1024x4096", "shape": (1024, 4096), "dtype": torch.float32},
        {"name": "rowwise_4096x1024", "shape": (4096, 1024), "dtype": torch.float32},
    ]
