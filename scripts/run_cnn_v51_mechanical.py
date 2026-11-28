"""Run the V5.1 one-pass mechanical P2 operation on attempt-local inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from dronedet.cnn_v51.artifacts import read_csv  # noqa: E402
from dronedet.cnn_v51.config import load_v51_config  # noqa: E402
from dronedet.cnn_v51.phase_p2 import run_p2, write_p2_ledger  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-specs", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--legacy-reference", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=STUDY_ROOT / "config" / "v51.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    attempts_root = STUDY_ROOT.parents[1] / "local" / "study-v5-1" / "attempts"
    resolved_output = args.output.resolve()
    try:
        resolved_output.relative_to(attempts_root.resolve())
    except ValueError as exc:
        raise SystemExit(f"P2 output must stay under the ignored attempt directory {attempts_root}") from exc
    if resolved_output.exists():
        raise SystemExit(f"refusing to overwrite P2 attempt: {resolved_output}")

    candidate_specs = json.loads(args.candidate_specs.read_text(encoding="utf-8"))
    legacy_reference = json.loads(args.legacy_reference.read_text(encoding="utf-8"))
    if not isinstance(candidate_specs, list) or not isinstance(legacy_reference, list):
        raise SystemExit("candidate specs and legacy reference must be JSON arrays")
    config = load_v51_config(args.config)
    result = run_p2(
        candidate_specs,
        read_csv(args.source_inventory, "inventory_v51.csv"),
        config=config.frontend,
        legacy_reference_rows=legacy_reference,
    )
    summary = write_p2_ledger(resolved_output, result)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
