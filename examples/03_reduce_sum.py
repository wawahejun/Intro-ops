from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import torch

from operator_runtime import reduce_sum


def main() -> None:
    src = torch.randn((32, 128), device="cuda", dtype=torch.float32)
    out = reduce_sum(src, dim=1, backend="nvidia")
    torch.testing.assert_close(out, torch.sum(src, dim=1), atol=1e-5, rtol=1e-5)
    print("reduce_sum ok")


if __name__ == "__main__":
    main()

