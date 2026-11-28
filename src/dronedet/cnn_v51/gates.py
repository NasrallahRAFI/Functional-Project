"""V5.1 gate state vocabulary and append-only phase ledger."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any


class GateState(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


def append_gate(path: str | Path, *, attempt_id: str, phase: str, gate: str,
                state: GateState, evidence: Any = None) -> None:
    """Append one auditable result; never rewrite a prior gate record."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "attempt_id": attempt_id,
        "phase": phase,
        "gate": gate,
        "state": GateState(state).value,
        "evidence": evidence,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with target.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))
        stream.write("\n")
