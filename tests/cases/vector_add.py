from __future__ import annotations

import torch


def correctness_cases():
    return [
        {"name": "contiguous_1k_fp32", "shape": (1024,), "dtype": torch.float32, "atol": 1e-6, "rtol": 1e-6},
        {"name": "contiguous_1k_fp16", "shape": (1024,), "dtype": torch.float16, "atol": 1e-3, "rtol": 1e-3},
        {"name": "contiguous_1536_fp32", "shape": (1536,), "dtype": torch.float32, "atol": 1e-6, "rtol": 1e-6},
        {"name": "contiguous_1536_fp16", "shape": (1536,), "dtype": torch.float16, "atol": 1e-3, "rtol": 1e-3},
    ]


def api_error_cases():
    return [
        {"name": "shape_mismatch", "shape": (16,), "other_shape": (8,), "dtype": torch.float32},
        {"name": "dtype_mismatch", "shape": (16,), "dtype": torch.float32, "other_dtype": torch.float16},
        {"name": "non_contiguous", "shape": (4, 4), "dtype": torch.float32},
    ]


def benchmark_cases():
    return [
        {"name": "contiguous_1m_fp16", "shape": (1 << 20,), "dtype": torch.float16},
        {"name": "contiguous_4m_fp16", "shape": (4 << 20,), "dtype": torch.float16},
        {"name": "contiguous_16m_fp16", "shape": (16 << 20,), "dtype": torch.float16},
        {"name": "contiguous_64m_fp16", "shape": (64 << 20,), "dtype": torch.float16},
        {"name": "contiguous_1m_fp32", "shape": (1 << 20,), "dtype": torch.float32},
        {"name": "contiguous_4m_fp32", "shape": (4 << 20,), "dtype": torch.float32},
        {"name": "contiguous_16m_fp32", "shape": (16 << 20,), "dtype": torch.float32},
        {"name": "contiguous_64m_fp32", "shape": (64 << 20,), "dtype": torch.float32},
    ]
