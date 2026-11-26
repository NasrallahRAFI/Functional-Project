"""Fail with actionable guidance when the local browser runtime is incomplete."""
from __future__ import annotations

import importlib
import importlib.metadata
import sys

REQUIRED = {
    "torch": "torch",
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "librosa": "librosa",
    "soxr": "soxr",
    "soundfile": "soundfile",
    "yaml": "PyYAML",
}
VERSIONS = {
    "torch": "2.15.0", "numpy": "2.5.5", "pandas": "3.1.1", "scipy": "1.18.1",
    "soundfile": "0.14.0", "soxr": "1.1.0", "librosa": "1.0.1",
    "PyYAML": "6.0.4", "numba": "0.68.1", "llvmlite": "0.50.0",
}


def main() -> int:
    failed = []
    for module, package in REQUIRED.items():
        try:
            importlib.import_module(module)
        except Exception as exc:
            failed.append(f"{package}: {exc}")
    for package, expected in VERSIONS.items():
        try:
            actual = importlib.metadata.version(package).split("+")[0]
            if actual != expected:
                failed.append(f"{package}: expected {expected}, found {actual}")
        except importlib.metadata.PackageNotFoundError:
            failed.append(f"{package}: not installed")
    if sys.version_info[:2] != (3, 12):
        failed.append(f"Python 3.12 is required; found {sys.version.split()[0]}")
    if failed:
        print("The live detector runtime is incomplete:", file=sys.stderr)
        for item in failed:
            print(f" - {item}", file=sys.stderr)
        print("Use Python 3.12 and install the pinned runtime requirements.", file=sys.stderr)
        return 1
    print("Live detector imports and Python version are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
