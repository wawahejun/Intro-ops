from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_manifest_validator_passes():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "tools/validate_operator_manifest.py",
            "--ops-root",
            "ops",
            "--tests-root",
            "tests",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

