"""Execute a single authorized V5.1 P3 attempt from closed P1/P2 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from dronedet.cnn_v51.artifacts import read_csv  # noqa: E402
from dronedet.cnn_v51.phase_p3 import run_p3  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-ledger", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    attempts_root = STUDY_ROOT.parents[1] / "local" / "study-v5-1" / "attempts"
    resolved_output = args.output.resolve()
    try:
        resolved_output.relative_to(attempts_root.resolve())
    except ValueError as exc:
        raise SystemExit(f"P3 output must stay under the ignored attempt directory {attempts_root}") from exc
    if resolved_output.exists():
        raise SystemExit(f"refusing to overwrite P3 attempt: {resolved_output}")

    bindings = json.loads(args.bindings.read_text(encoding="utf-8"))
    result = run_p3(
        resolved_output,
        candidate_rows=read_csv(args.candidate_ledger, "mechanical_ledger.csv"),
        source_file_rows=read_csv(args.source_inventory, "inventory_v51.csv"),
        quarantine_rows=read_csv(args.quarantine, "quarantine_manifest.csv"),
        bindings=bindings,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
