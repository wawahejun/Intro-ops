from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path


def _candidate_library_paths() -> list[Path]:
    candidates: list[Path] = []
    build_dir = os.environ.get("CAMP_BUILD_DIR")
    if build_dir:
        root = Path(build_dir)
        candidates.extend([root / "libcamp_ops.so", root / "ops" / "libcamp_ops.so"])
    repo_root = Path(__file__).resolve().parents[3]
    candidates.extend(
        [
            repo_root / "build" / "libcamp_ops.so",
            repo_root / "build" / "ops" / "libcamp_ops.so",
        ]
    )
    return candidates


@lru_cache(maxsize=1)
def load_library() -> ctypes.CDLL:
    for path in _candidate_library_paths():
        if path.exists():
            return ctypes.CDLL(str(path))
    searched = ", ".join(str(path) for path in _candidate_library_paths())
    raise FileNotFoundError(f"libcamp_ops.so not found; searched: {searched}")
