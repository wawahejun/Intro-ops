from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from operator_runtime import reduce_sum
from operator_runtime.testing import cuda_time_ms, PerformanceResult
from tests.cases import reduce_sum as reduce_sum_cases


def _estimate_reduce_sum(tensor: torch.Tensor) -> tuple[int, int]:
    rows = tensor.shape[0]
    elem_bytes = tensor.element_size()
    return (tensor.numel() + rows) * elem_bytes, tensor.numel()


def bench_reduce_sum(backend: str) -> list[PerformanceResult]:
    rows: list[PerformanceResult] = []
    for case in reduce_sum_cases.benchmark_cases():
        src = torch.randn(case["shape"], dtype=case["dtype"], device="cuda")
        runtime = cuda_time_ms(lambda: reduce_sum(src, dim=1, backend=backend))
        torch_ms = cuda_time_ms(lambda: torch.sum(src, dim=1))
        bytes_, flops = _estimate_reduce_sum(src)
        rows.append(
            PerformanceResult(
                "reduce_sum",
                backend,
                str(tuple(src.shape)),
                str(src.dtype),
                bytes_,
                flops,
                runtime,
                torch_ms,
            )
        )
    return rows
