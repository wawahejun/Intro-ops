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

from operator_runtime import softmax
from operator_runtime_testing import cuda_time_ms, PerformanceResult
from tests.cases import softmax as softmax_cases


def _estimate_softmax(tensor: torch.Tensor) -> tuple[int, int]:
    elem_bytes = tensor.element_size()
    return 5 * tensor.numel() * elem_bytes, 4 * tensor.numel()


def bench_softmax(backend: str) -> list[PerformanceResult]:
    rows: list[PerformanceResult] = []
    for case in softmax_cases.benchmark_cases():
        src = torch.randn(case["shape"], dtype=case["dtype"], device="cuda")
        runtime = cuda_time_ms(lambda: softmax(src, dim=1, backend=backend))
        torch_ms = cuda_time_ms(lambda: torch.softmax(src, dim=1))
        bytes_, flops = _estimate_softmax(src)
        rows.append(
            PerformanceResult(
                "softmax",
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
