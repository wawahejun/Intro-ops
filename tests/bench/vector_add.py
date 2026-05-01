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

from operator_runtime import vector_add
from operator_runtime.benchmark import cuda_time_ms
from operator_runtime.profiler import PerformanceResult
from tests.cases import vector_add as vector_add_cases


def _estimate_vector_add(tensor: torch.Tensor) -> tuple[int, int]:
    elem_bytes = tensor.element_size()
    return 3 * tensor.numel() * elem_bytes, tensor.numel()


def bench_vector_add(backend: str) -> list[PerformanceResult]:
    rows: list[PerformanceResult] = []
    for case in vector_add_cases.benchmark_cases():
        a = torch.randn(case["shape"], dtype=case["dtype"], device="cuda")
        b = torch.randn_like(a)
        runtime = cuda_time_ms(lambda: vector_add(a, b, backend=backend))
        torch_ms = cuda_time_ms(lambda: torch.add(a, b))
        bytes_, flops = _estimate_vector_add(a)
        rows.append(
            PerformanceResult(
                "vector_add",
                backend,
                str(tuple(a.shape)),
                str(a.dtype),
                bytes_,
                flops,
                runtime,
                torch_ms,
            )
        )
    return rows
