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

from operator_runtime import copy
from operator_runtime_testing import cuda_time_ms, PerformanceResult
from tests.cases import copy as copy_cases


def _estimate_copy(tensor: torch.Tensor) -> tuple[int, int]:
    elem_bytes = tensor.element_size()
    return 2 * tensor.numel() * elem_bytes, 0


def bench_copy(backend: str) -> list[PerformanceResult]:
    rows: list[PerformanceResult] = []
    for case in copy_cases.benchmark_cases():
        src = torch.randn(case["shape"], dtype=case["dtype"], device="cuda")
        out = torch.empty_like(src)
        runtime = cuda_time_ms(lambda: copy(src, backend=backend))
        torch_ms = cuda_time_ms(lambda: out.copy_(src))
        bytes_, flops = _estimate_copy(src)
        rows.append(
            PerformanceResult(
                "copy",
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
