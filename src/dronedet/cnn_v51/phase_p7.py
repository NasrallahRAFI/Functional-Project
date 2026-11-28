"""Final preparation gate ledger and immutable handoff manifest writer."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .artifacts import canonical_json_bytes

_GATE_STATES = {"PASS", "FAIL", "NOT_RUN", "NOT_IMPLEMENTED"}
_PHASES = ("A0", "P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_gate_ledger(phase_states: dict[str, str], *, plan_sha256: str,
                      contract_sha256: str, evidence: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Create explicit phase gate rows, defaulting every missing gate to NOT_RUN."""
    if set(phase_states) - set(_PHASES):
        raise ValueError("phase state map contains an unknown phase")
    if any(state not in _GATE_STATES for state in phase_states.values()):
        raise ValueError("gate state must be one of the four contract states")
    for label, digest in (("plan", plan_sha256), ("contract", contract_sha256)):
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"{label} binding must be lowercase SHA-256")
    evidence = evidence or {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return [{"phase": phase, "gate": phase, "state": phase_states.get(phase, "NOT_RUN"),
             "plan_sha256": plan_sha256, "contract_sha256": contract_sha256,
             "recorded_utc": now, "evidence": evidence.get(phase, {})}
            for phase in _PHASES]


def write_p7_handoff(output_directory: str | Path, *, phase_states: dict[str, str],
                     plan_path: str | Path, contract_path: str | Path,
                     implementation_files: Iterable[str | Path], report_path: str | Path,
                     frozen_membership_path: str | Path | None = None,
                     evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write a hash-only handoff manifest, withholding release state on open gates."""
    target = Path(output_directory)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite P7 handoff attempt: {target}")
    plan, contract, report = Path(plan_path), Path(contract_path), Path(report_path)
    if not plan.is_file() or not contract.is_file() or not report.is_file():
        raise FileNotFoundError("P7 requires the approved plan, contract and preparation report")
    plan_hash, contract_hash = _sha256(plan), _sha256(contract)
    artifacts = []
    for raw in sorted({Path(item).resolve() for item in implementation_files} | {report.resolve(), plan.resolve(), contract.resolve()}, key=str):
        if not raw.is_file():
            raise FileNotFoundError(f"P7 handoff file is missing: {raw}")
        artifacts.append({"name": raw.name, "sha256": _sha256(raw), "bytes": raw.stat().st_size,
                          "kind": "approved_plan" if raw == plan.resolve() else
                                  "contract" if raw == contract.resolve() else
                                  "preparation_report" if raw == report.resolve() else "implementation_source"})
    frozen_hash = None
    if frozen_membership_path is not None:
        frozen = Path(frozen_membership_path)
        if frozen.is_file():
            frozen_hash = _sha256(frozen)
    ledger = build_gate_ledger(phase_states, plan_sha256=plan_hash, contract_sha256=contract_hash,
                               evidence=evidence)
    blocking = [row["phase"] for row in ledger if row["phase"] != "P7" and row["state"] != "PASS"]
    state = "PASS" if not blocking and frozen_hash is not None else "NOT_RUN"
    handoff = {"study_id": "study-v5-1", "phase": "P7", "state": state,
               "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
               "plan_sha256": plan_hash, "contract_sha256": contract_hash,
               "frozen_membership_sha256": frozen_hash,
               "blocking_phases": blocking if blocking else ([] if frozen_hash is not None else ["P3"]),
               "future_training_threshold_activation_and_test_evaluation": "NOT_RUN",
               "artifacts": artifacts}
    target.mkdir(parents=True, exist_ok=False)
    ledger_path = target / "phase_gate_ledger.jsonl"
    ledger_path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in ledger))
    handoff_path = target / "handoff_manifest.json"
    handoff_bytes = canonical_json_bytes(handoff) + b"\n"
    handoff_path.write_bytes(handoff_bytes)
    return {"state": state, "ledger_sha256": _sha256(ledger_path),
            "handoff_sha256": hashlib.sha256(handoff_bytes).hexdigest(),
            "blocking_phases": handoff["blocking_phases"], "artifacts": len(artifacts)}
