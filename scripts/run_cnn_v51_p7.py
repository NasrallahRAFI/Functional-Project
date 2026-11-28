"""Write the current V5.1 phase gate ledger and hash-only handoff attempt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

STUDY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STUDY_ROOT.parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from dronedet.cnn_v51.phase_p7 import write_p7_handoff  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-membership", type=Path)
    args = parser.parse_args()
    attempts_root = PROJECT_ROOT / "local" / "study-v5-1" / "attempts"
    output = args.output.resolve()
    try:
        output.relative_to(attempts_root.resolve())
    except ValueError as exc:
        raise SystemExit(f"P7 output must stay under ignored attempt directory {attempts_root}") from exc
    registry = json.loads((PROJECT_ROOT / "repository" / "registry" / "studies" / "study-v5-1.json").read_text(encoding="utf-8"))
    phase_states = dict(registry["preparation_gates"])
    phase_states["P7"] = "NOT_RUN"
    implementation = sorted((STUDY_ROOT / "src" / "dronedet" / "cnn_v51").glob("*.py"))
    implementation += sorted(path for path in (STUDY_ROOT / "scripts").glob("run_cnn_v51_*.py"))
    implementation += [STUDY_ROOT / "config" / "v51.yaml", STUDY_ROOT / "requirements-v51.txt"]
    result = write_p7_handoff(
        output, phase_states=phase_states,
        plan_path=STUDY_ROOT / "docs" / "v5.1" / "DATA_PREPARATION_CNN_V5_1_FINAL.md",
        contract_path=STUDY_ROOT / "docs" / "v5.1" / "DATA_PREPARATION_CNN_V5_1_CONTRACT.json",
        implementation_files=implementation,
        report_path=STUDY_ROOT / "reports" / "v5.1" / "DATA_PREPARATION_CNN_V5_1_REPORT.md",
        frozen_membership_path=args.frozen_membership,
        evidence={"P1": {"stop_reason": "required ITU ARIS and MUSAN snapshots are absent"},
                  "P3": {"state": "NOT_RUN", "reason": "no complete P2 ledger; no freeze"},
                  "P5": {"state": "NOT_RUN", "reason": "requires frozen memberships and P4 weights"},
                  "P6": {"state": "NOT_RUN", "reason": "requires frozen validation streams"}},
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["state"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
