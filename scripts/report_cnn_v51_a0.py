"""Summarize the existing V4 ambient score/RMS log without reading audio."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT = WORKSPACE.parents[1]
DEFAULT_LOG = PROJECT / "local/reports/study-v4/ambient-score-logs/ambient-v4-20261126T212021Z-c7cf5a35.jsonl"


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return ordered[left]
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ambient-log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()
    if not args.ambient_log.is_file():
        print("A0 unavailable: the existing V4 ambient score log is missing.")
        return 0

    records = [json.loads(line) for line in args.ambient_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    headers = [row for row in records if row.get("record_type") in {"header", "session_header"}]
    windows = [row for row in records if row.get("record_type") == "window" and row.get("status") == "score_only"]
    logits = [float(row["raw_logit"]) for row in windows if isinstance(row.get("raw_logit"), (float, int))]
    rms = [float(row["rms_dbfs"]) for row in windows if isinstance(row.get("rms_dbfs"), (float, int))]
    output = WORKSPACE / "reports/v5.1/A0_EXISTING_V4_LOGS.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not windows or not logits or not rms:
        output.write_text(
            "# A0 — existing V4 logs\n\n**Status: unavailable.** The existing log has no complete set of scored-window raw logits and RMS values. No waveform was opened and no calibration was attempted.\n",
            encoding="utf-8", newline="\n",
        )
        return 0

    header = headers[0] if headers else {}
    identity_fields = ("statistics_sha256", "frontend_hash")
    identity_missing = [field for field in identity_fields if not header.get(field)]
    logit_q = [quantile(logits, p) for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    rms_q = [quantile(rms, p) for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    duration = max((float(row.get("elapsed_seconds", 0)) for row in windows), default=0.0)
    lines = [
        "# A0 — existing V4 ambient score and RMS log",
        "",
        "**A0 gate: PASS.** Descriptive only; not identity-verified V5.1 calibration.",
        "",
        f"Input: `{args.ambient_log.name}` (existing V4 log; no waveform read).",
        f"Scored windows: {len(windows)}. Recorded span: {duration:.3f} seconds.",
        f"Raw logit min / p25 / median / p75 / max: {', '.join(f'{v:.6g}' for v in logit_q)}.",
        f"Recorded RMS dBFS min / p25 / median / p75 / max: {', '.join(f'{v:.6g}' for v in rms_q)}.",
        f"Missing identity header fields: {', '.join(identity_missing) if identity_missing else 'none'}.",
        "",
        "## Training-level comparison",
        "",
        "A verified V4 training-window RMS distribution is not available in the registered report artifacts. The V4 training report records the historical random-gain range, but that does not establish the RMS distribution of training inputs. Therefore this log cannot be compared quantitatively with recorded training levels. No levels were inferred from audio, and no score threshold or calibration claim was produced.",
        "",
        "This log is retained as owner-confirmed no-drone context in the V4 provisional-threshold report. Its missing frontend/statistics identity fields mean it is descriptive evidence only. The short exposure and absent training-level reference leave H1/H3/H4 unresolved; the score/RMS association is not causal evidence for H2.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({"a0": "PASS", "windows": len(windows), "training_level_comparison": "UNAVAILABLE", "report": str(output.relative_to(WORKSPACE))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
