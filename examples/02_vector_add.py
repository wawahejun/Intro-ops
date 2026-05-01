from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import torch

from operator_runtime import vector_add


def main() -> None:
    a = torch.randn((1024,), device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)
    out = vector_add(a, b, backend="nvidia")
    torch.testing.assert_close(out, a + b)
    print("vector_add ok")


if __name__ == "__main__":
    main()

