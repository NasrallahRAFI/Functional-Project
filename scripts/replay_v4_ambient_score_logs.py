"""Replay V4 ambient-only JSONL score logs through the production VoteSmoother.

The script never reads audio. It refuses incomplete, drone-present, non-V4, or
threshold-selected sessions. A candidate is eligible only after the configured
minimum observed-window exposure has been collected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from demo.run_demo import VoteSmoother  # noqa: E402

EXPECTED_MODEL_SHA = "f4372da59644395c174514944aa6387901ed483f64c137fd1ae4e31c83092049"
EXPECTED_STATS_SHA = "4dc6820ca53a7965ae1e0264e4bc69e6197bd1f91ce0299e287ddeb71a26d6a4"
EXPECTED_FRONTEND_HASH = "028d682f79e0099f265bf9194e11ba3250d16fc8ef5377944e7d7671ab60fdd2"
CADENCE_SECONDS = 0.496
SMOOTHING_WINDOWS = 3


def poisson_upper_95(count: int, exposure_hours: float) -> float:
    return 0.5 * float(chi2.ppf(0.95, 2 * (count + 1))) / exposure_hours


def load_log(path: Path) -> tuple[dict, list[dict], dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) < 3 or records[0].get("record_type") != "session_start" or records[-1].get("record_type") != "session_end":
        raise ValueError(f"session log is incomplete: {path.name}")
    header, footer = records[0], records[-1]
    if (header.get("purpose") != "ambient_only_no_drone" or header.get("audio_saved") is not False
            or header.get("threshold_selected") is not False
            or header.get("model_id") != "CNN_V4.0"
            or header.get("checkpoint_sha256") != EXPECTED_MODEL_SHA
            or header.get("statistics_sha256") != EXPECTED_STATS_SHA
            or header.get("frontend_hash") != EXPECTED_FRONTEND_HASH
            or header.get("smoothing_windows_for_candidate_replay") != SMOOTHING_WINDOWS
            or float(header.get("cadence_seconds", -1)) != CADENCE_SECONDS
            or footer.get("audio_saved") is not False
            or footer.get("session_id") != header.get("session_id")):
        raise ValueError(f"session log does not match the V4 ambient-only contract: {path.name}")
    windows = records[1:-1]
    for row in windows:
        if row.get("record_type") != "window" or row.get("session_id") != header.get("session_id"):
            raise ValueError(f"invalid window record in {path.name}")
        score = row.get("raw_logit")
        rms = row.get("rms_dbfs")
        if score is not None and not math.isfinite(float(score)):
            raise ValueError(f"non-finite logit in {path.name}")
        if rms is not None and not math.isfinite(float(rms)):
            raise ValueError(f"non-finite RMS in {path.name}")
    if int(footer.get("window_records", -1)) != len(windows):
        raise ValueError(f"session footer count does not match window rows: {path.name}")
    return header, windows, footer


def replay(windows: list[dict], thresholds: list[float]) -> dict[float, int]:
    counts = {threshold: 0 for threshold in thresholds}
    for threshold in thresholds:
        smoother = VoteSmoother(SMOOTHING_WINDOWS)
        previous = None
        alarm_open = False
        for row in windows:
            sequence = int(row["chunk_sequence"])
            window_index = int(row.get("window_in_chunk", 0))
            source_frame = row.get("source_frame")
            rate = int(row["source_sample_rate_hz"])
            generation = row.get("generation")
            dropped = int(row.get("capture_dropped_frames_total", 0))
            gap = bool(row.get("recovered"))
            if previous is not None:
                gap = gap or generation != previous["generation"]
                gap = gap or rate != previous["rate"]
                gap = gap or sequence < previous["sequence"]
                if source_frame is not None and previous["source_frame"] is not None:
                    expected_frame = previous["source_frame"] + round(rate * 0.5) * max(0, sequence - previous["sequence"])
                    gap = gap or abs(int(source_frame) - expected_frame) > max(1, round(rate * 0.02))
                if sequence == previous["sequence"]:
                    gap = gap or window_index != previous["window_index"] + 1
                gap = gap or dropped > previous["dropped"]
            if gap:
                smoother.reset()
                alarm_open = False
            score = row.get("raw_logit")
            if score is None:
                smoother.reset()
                alarm_open = False
            else:
                decision = smoother.push(float(score), threshold)
                active = bool(decision)
                if active and not alarm_open:
                    counts[threshold] += 1
                alarm_open = active
            previous = {"sequence": sequence, "window_index": window_index, "source_frame": source_frame,
                        "rate": rate, "generation": generation, "dropped": dropped}
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=PROJECT / "local" / "reports" / "study-v4" / "ambient-score-logs")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT / "local" / "reports" / "study-v4" / "ambient-score-replay")
    parser.add_argument("--threshold-step", type=float, default=0.25)
    parser.add_argument("--false-alarm-target-per-hour", type=float, default=1.0)
    parser.add_argument("--minimum-hours", type=float, default=24.0)
    args = parser.parse_args()
    if (not math.isfinite(args.threshold_step) or args.threshold_step <= 0
            or not math.isfinite(args.false_alarm_target_per_hour) or args.false_alarm_target_per_hour <= 0
            or not math.isfinite(args.minimum_hours) or args.minimum_hours <= 0):
        raise SystemExit("grid step, false-alarm target, and minimum hours must be positive and finite")
    if not args.input_dir.is_dir():
        raise SystemExit(f"No ambient score log folder found: {args.input_dir}")
    files = sorted(args.input_dir.glob("ambient-v4-*.jsonl"))
    if not files:
        raise SystemExit(f"No V4 ambient JSONL logs found in {args.input_dir}")

    sessions = []
    session_data = []
    scores = []
    environment_hours: dict[str, float] = defaultdict(float)
    for path in files:
        header, windows, footer = load_log(path)
        usable = [row for row in windows if row.get("rms_dbfs") is not None]
        if not usable:
            raise SystemExit(f"No emitted audio windows in session log: {path.name}")
        exposure = len(usable) * CADENCE_SECONDS
        environment = str(header.get("environment_label") or "unspecified")
        environment_hours[environment] += exposure / 3600.0
        scores.extend(float(row["raw_logit"]) for row in usable if row.get("raw_logit") is not None)
        session_data.append(usable)
        sessions.append({"file": path.name, "session_id": header["session_id"],
            "environment_label": environment, "started_at_utc": header.get("started_at_utc"),
            "window_records": len(usable), "exposure_hours": exposure / 3600.0,
            "elapsed_seconds": footer.get("elapsed_seconds"), "end_reason": footer.get("end_reason")})

    total_hours = sum(item["exposure_hours"] for item in sessions)
    if not scores:
        raise SystemExit("No non-quiet raw logits exist in these sessions")
    low = math.floor(min(scores) / args.threshold_step) * args.threshold_step
    high = math.ceil(max(scores) / args.threshold_step) * args.threshold_step + args.threshold_step
    thresholds = [round(low + i * args.threshold_step, 6)
                  for i in range(int(round((high - low) / args.threshold_step)) + 1)]

    replay_rows = []
    qualifying = []
    for threshold in thresholds:
        alarms = 0
        for windows in session_data:
            alarms += replay(windows, [threshold])[threshold]
        upper = poisson_upper_95(alarms, total_hours)
        eligible = total_hours >= args.minimum_hours and upper <= args.false_alarm_target_per_hour
        row = {"threshold_logit": threshold, "smoothed_alarm_events": alarms,
               "observed_hours": total_hours, "alarms_per_hour": alarms / total_hours,
               "poisson_95pct_upper_per_hour": upper,
               "minimum_hours_met": total_hours >= args.minimum_hours,
               "meets_false_alarm_target": upper <= args.false_alarm_target_per_hour,
               "eligible_after_duration_gate": eligible}
        replay_rows.append(row)
        if eligible:
            qualifying.append(threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    replay_path = args.output_dir / "ambient_threshold_replay.csv"
    with replay_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(replay_rows[0]))
        writer.writeheader()
        writer.writerows(replay_rows)
    report = {"status": "CANDIDATE_READY" if qualifying else "MORE_AMBIENT_DATA_REQUIRED",
        "deployment_threshold_selected": False,
        "lowest_candidate_meeting_fixed_rule": min(qualifying) if qualifying else None,
        "false_alarm_target_per_hour": args.false_alarm_target_per_hour,
        "minimum_observed_hours": args.minimum_hours,
        "observed_exposure_hours": total_hours,
        "duration_gate_met": total_hours >= args.minimum_hours,
        "smoothing_windows": SMOOTHING_WINDOWS,
        "cadence_seconds": CADENCE_SECONDS,
        "threshold_grid_step_logit": args.threshold_step,
        "session_count": len(sessions), "sessions": sessions,
        "environment_exposure_hours": dict(sorted(environment_hours.items())),
        "test_access": False, "audio_read": False,
        "note": "A candidate is not automatically activated. Check validation recall against its frozen 80% floor, then freeze the operating point and evaluate test once.",
        "threshold_replay_csv": str(replay_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    report_path = args.output_dir / "ambient_threshold_replay_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"],
        "deployment_threshold_selected": False,
        "observed_exposure_hours": total_hours,
        "duration_gate_met": report["duration_gate_met"],
        "lowest_candidate_meeting_fixed_rule": report["lowest_candidate_meeting_fixed_rule"],
        "session_count": len(sessions), "environment_exposure_hours": report["environment_exposure_hours"],
        "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
