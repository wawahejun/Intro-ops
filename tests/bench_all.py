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

from operator_runtime.profiler import PerformanceResult
from tests.bench.copy import bench_copy
from tests.bench.reduce_sum import bench_reduce_sum
from tests.bench.softmax import bench_softmax
from tests.bench.vector_add import bench_vector_add


def _format_table(rows: list[PerformanceResult]) -> str:
    headers = [
        "operator",
        "backend",
        "shape",
        "dtype",
        "runtime_ms",
        "torch_ms",
        "speedup",
        "GB/s",
        "GFLOP/s",
    ]
    body = []
    for row in rows:
        speedup = "-" if row.speedup is None else f"{row.speedup:.2f}"
        torch_ms = "-" if row.torch_ms is None else f"{row.torch_ms:.4f}"
        body.append(
            [
                row.operator,
                row.backend,
                row.shape,
                row.dtype,
                f"{row.runtime_ms:.4f}",
                torch_ms,
                speedup,
                f"{row.gbytes_per_sec:.2f}",
                f"{row.gflops_per_sec:.2f}",
            ]
        )

    widths = []
    for idx, header in enumerate(headers):
        content_width = max((len(r[idx]) for r in body), default=0)
        widths.append(max(len(header), content_width))

    def fmt_row(cols: list[str]) -> str:
        return " | ".join(col.ljust(widths[idx]) for idx, col in enumerate(cols))

    separator = "-+-".join("-" * width for width in widths)
    lines = [fmt_row(headers), separator]
    lines.extend(fmt_row(cols) for cols in body)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="nvidia")
    parser.add_argument("--profile", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for benchmark", file=sys.stderr)
        return 2

    rows: list[PerformanceResult] = []
    rows.extend(bench_copy(args.backend))
    rows.extend(bench_vector_add(args.backend))
    rows.extend(bench_reduce_sum(args.backend))
    rows.extend(bench_softmax(args.backend))

    print(_format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
