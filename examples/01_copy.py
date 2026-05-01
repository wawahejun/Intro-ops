from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import torch

from operator_runtime import copy


def main() -> None:
    src = torch.randn((1024,), device="cuda", dtype=torch.float32)
    out = copy(src, backend="nvidia")
    torch.testing.assert_close(out, src)
    print("copy ok")


if __name__ == "__main__":
    main()

