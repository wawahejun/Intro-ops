#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required to validate operator manifests") from exc


REQUIRED_FIELDS = {
    "name",
    "kind",
    "python_module",
    "torch_reference",
    "backends",
    "tolerances",
    "benchmark",
}

NVIDIA_SYMBOLS = {"create", "workspace", "execute", "destroy"}


def fail(message: str) -> None:
    raise ValueError(message)


def load_manifest(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    missing = REQUIRED_FIELDS - set(data)
    if missing:
        fail(f"{path}: missing required fields {sorted(missing)}")
    return data


def check_path(root: Path, manifest_path: Path, relative: str) -> None:
    path = root / relative
    if not path.exists():
        fail(f"{manifest_path}: referenced path does not exist: {relative}")


def check_backend_paths(root: Path, manifest_path: Path, manifest: dict[str, Any]) -> None:
    for backend_name, backend in manifest["backends"].items():
        status = backend.get("status")
        if status == "runnable":
            for key in ("sources", "headers"):
                for relative in backend.get(key, []):
                    check_path(root, manifest_path, relative)
        if backend_name == "nvidia" and status == "runnable":
            symbols = backend.get("symbols", {})
            missing = NVIDIA_SYMBOLS - set(symbols)
            if missing:
                fail(f"{manifest_path}: nvidia backend missing symbols {sorted(missing)}")


def check_python_module(repo_root: Path, manifest_path: Path, manifest: dict[str, Any]) -> None:
    module_name = manifest["python_module"]
    module_path = repo_root / "python" / Path(*module_name.split(".")).with_suffix(".py")
    if not module_path.exists():
        fail(f"{manifest_path}: python module does not exist: {module_name}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        fail(f"{manifest_path}: cannot create import spec for {module_name}")


def check_cases(tests_root: Path, manifest_path: Path, manifest: dict[str, Any]) -> None:
    case_path = tests_root / "cases" / f"{manifest['name']}.py"
    if not case_path.exists():
        fail(f"{manifest_path}: missing test case file {case_path}")
    text = case_path.read_text()
    for token in ("correctness_cases", "api_error_cases", "benchmark_cases"):
        if token not in text:
            fail(f"{manifest_path}: {case_path} missing {token}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops-root", type=Path, required=True)
    parser.add_argument("--tests-root", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path.cwd()
    manifests = sorted(args.ops_root.glob("*/operator.yaml"))
    if not manifests:
        raise SystemExit(f"no operator manifests found under {args.ops_root}")

    errors: list[str] = []
    for manifest_path in manifests:
        try:
            manifest = load_manifest(manifest_path)
            check_backend_paths(repo_root, manifest_path, manifest)
            check_python_module(repo_root, manifest_path, manifest)
            check_cases(args.tests_root, manifest_path, manifest)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"validated {len(manifests)} operator manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

