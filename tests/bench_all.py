from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from operator_runtime import copy, reduce_sum, softmax, vector_add
from operator_runtime.benchmark import cuda_time_ms
from operator_runtime.perf_model import (
    estimate_copy,
    estimate_reduce_sum,
    estimate_softmax,
    estimate_vector_add,
)
from operator_runtime.profiler import PerformanceResult


def bench_copy(backend: str) -> PerformanceResult:
    src = torch.randn((1 << 20,), dtype=torch.float16, device="cuda")
    out = torch.empty_like(src)
    runtime = cuda_time_ms(lambda: copy(src, backend=backend))
    torch_ms = cuda_time_ms(lambda: out.copy_(src))
    bytes_, flops = estimate_copy(src)
    return PerformanceResult("copy", backend, str(tuple(src.shape)), str(src.dtype), bytes_, flops, runtime, torch_ms)


def bench_vector_add(backend: str) -> PerformanceResult:
    a = torch.randn((1 << 20,), dtype=torch.float16, device="cuda")
    b = torch.randn_like(a)
    runtime = cuda_time_ms(lambda: vector_add(a, b, backend=backend))
    torch_ms = cuda_time_ms(lambda: torch.add(a, b))
    bytes_, flops = estimate_vector_add(a)
    return PerformanceResult("vector_add", backend, str(tuple(a.shape)), str(a.dtype), bytes_, flops, runtime, torch_ms)


def bench_reduce_sum(backend: str) -> PerformanceResult:
    src = torch.randn((1024, 1024), dtype=torch.float32, device="cuda")
    runtime = cuda_time_ms(lambda: reduce_sum(src, dim=1, backend=backend))
    torch_ms = cuda_time_ms(lambda: torch.sum(src, dim=1))
    bytes_, flops = estimate_reduce_sum(src)
    return PerformanceResult("reduce_sum", backend, str(tuple(src.shape)), str(src.dtype), bytes_, flops, runtime, torch_ms)


def bench_softmax(backend: str) -> PerformanceResult:
    src = torch.randn((1024, 1024), dtype=torch.float32, device="cuda")
    runtime = cuda_time_ms(lambda: softmax(src, dim=1, backend=backend))
    torch_ms = cuda_time_ms(lambda: torch.softmax(src, dim=1))
    bytes_, flops = estimate_softmax(src)
    return PerformanceResult("softmax", backend, str(tuple(src.shape)), str(src.dtype), bytes_, flops, runtime, torch_ms)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="nvidia")
    parser.add_argument("--profile", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for benchmark", file=sys.stderr)
        return 2

    rows = [
        bench_copy(args.backend),
        bench_vector_add(args.backend),
        bench_reduce_sum(args.backend),
        bench_softmax(args.backend),
    ]

    print("operator backend shape dtype runtime_ms torch_ms speedup GB/s GFLOP/s")
    for row in rows:
        speedup = 0.0 if row.speedup is None else row.speedup
        print(
            f"{row.operator} {row.backend} {row.shape} {row.dtype} "
            f"{row.runtime_ms:.4f} {row.torch_ms or 0:.4f} {speedup:.2f} "
            f"{row.gbytes_per_sec:.2f} {row.gflops_per_sec:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

