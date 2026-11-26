"""Score the local MUSAN noise subset as supplemental V4 negative audio.

This is not a deployment threshold selector. It reads only MUSAN/noise WAVs,
uses the verified V4 checkpoint and live window builder, writes score metadata
to local/ and never copies or persists waveform samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from demo.run_demo import LiveWindowBuilder, VoteSmoother  # noqa: E402
from dronedet.features import clip_flags, mel_power, transform  # noqa: E402
from dronedet.inference import load_detector  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def poisson_upper_95(count: int, exposure_hours: float) -> float:
    """Exact one-sided 95% Poisson upper confidence bound per hour."""
    if exposure_hours <= 0:
        raise ValueError("exposure must be positive")
    upper_count = 0.5 * float(chi2.ppf(0.95, 2 * (count + 1)))
    return upper_count / exposure_hours


def emit_windows(samples: np.ndarray, source_rate: int, detector) -> list[dict]:
    cfg = detector.config
    builder = LiveWindowBuilder(cfg, source_rate)
    chunk_size = max(1, round(source_rate * 0.5))
    windows: list[np.ndarray] = []
    for start in range(0, samples.size, chunk_size):
        windows.extend(builder.push(samples[start : start + chunk_size]))
    windows.extend(builder.finish())

    rows: list[dict] = []
    for index, window in enumerate(windows):
        rms = float(np.sqrt(np.mean(window.astype(np.float64) ** 2)))
        rms_dbfs = 20.0 * math.log10(max(rms, 1e-15))
        flags = clip_flags(window, cfg)
        if flags["silent"] or flags["near_silent"]:
            rows.append({"window_index": index, "time_seconds": index * cfg.hop *
                         round(cfg.decision_interval_s * cfg.sr / cfg.hop) / cfg.sr,
                         "status": "standby_quiet", "raw_logit": None,
                         "rms_dbfs": round(rms_dbfs, 3)})
            continue
        features = transform(mel_power(window, cfg), cfg, detector.stats)
        with torch.inference_mode():
            score = float(detector.model(torch.from_numpy(features).unsqueeze(0)).item())
        if not math.isfinite(score):
            raise ValueError("V4 model produced a non-finite logit")
        rows.append({"window_index": index, "time_seconds": index * cfg.hop *
                     round(cfg.decision_interval_s * cfg.sr / cfg.hop) / cfg.sr,
                     "status": "scored", "raw_logit": score,
                     "rms_dbfs": round(rms_dbfs, 3)})
    return rows


def replay_file(rows: list[dict], thresholds: list[float], smoother_windows: int) -> dict[float, int]:
    alarms = {threshold: 0 for threshold in thresholds}
    for threshold in thresholds:
        smoother = VoteSmoother(smoother_windows)
        alarm_open = False
        for row in rows:
            score = row["raw_logit"]
            if score is None:
                smoother.reset()
                alarm_open = False
                continue
            decision = smoother.push(float(score), threshold)
            active = bool(decision)
            if active and not alarm_open:
                alarms[threshold] += 1
            alarm_open = active
    return alarms


def threshold_grid(scores: list[float], step: float) -> list[float]:
    if not scores:
        return []
    low = math.floor(min(scores) / step) * step
    high = math.ceil(max(scores) / step) * step + step
    count = int(round((high - low) / step))
    return [round(low + index * step, 6) for index in range(count + 1)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musan-root", type=Path, default=Path(r"D:\musan"))
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT / "local" / "reports" / "study-v4" / "musan-noise-supplement")
    parser.add_argument("--threshold-step", type=float, default=0.25,
                        help="logit spacing for descriptive supplemental replay")
    parser.add_argument("--false-alarm-target-per-hour", type=float, default=1.0)
    args = parser.parse_args()

    noise_root = args.musan_root.resolve() / "musan" / "noise"
    if not noise_root.is_dir():
        raise SystemExit(f"MUSAN noise folder not found: {noise_root}")
    if not math.isfinite(args.threshold_step) or args.threshold_step <= 0:
        raise SystemExit("--threshold-step must be finite and positive")
    wavs = sorted(noise_root.rglob("*.wav"), key=lambda path: path.as_posix().casefold())
    if not wavs:
        raise SystemExit(f"No WAV files found under {noise_root}")

    detector = load_detector(ROOT)
    if detector.model_id != "CNN_V4.0" or detector.threshold is not None:
        raise SystemExit("This scorer requires the registered V4 raw-score-only selection")
    smoother_windows = int(detector.config.smoothing_windows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_csv = args.output_dir / "musan_noise_window_scores.csv"
    inventory_path = args.output_dir / "musan_noise_inventory.json"

    inventory = []
    scores: list[float] = []
    durations: dict[str, float] = {}
    audio_file_count: dict[str, int] = {}
    rows_by_file: dict[str, list[dict]] = {}
    skipped: list[dict] = []
    total_seconds = 0.0
    total_windows = 0
    scored_windows = 0

    with score_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["recording_id", "source_group",
            "duration_seconds", "window_index", "time_seconds", "status", "raw_logit",
            "rms_dbfs"])
        writer.writeheader()
        for number, path in enumerate(wavs, start=1):
            relative = path.relative_to(noise_root).as_posix()
            try:
                audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
                if audio.shape[1] <= detector.config.channel:
                    raise ValueError("configured channel is absent")
                samples = np.ascontiguousarray(audio[:, detector.config.channel])
                if not samples.size or not np.isfinite(samples).all():
                    raise ValueError("empty or non-finite waveform")
                duration = samples.size / float(sample_rate)
                rows = emit_windows(samples, int(sample_rate), detector)
                file_scores = [float(row["raw_logit"]) for row in rows if row["raw_logit"] is not None]
                scores.extend(file_scores)
                total_seconds += duration
                total_windows += len(rows)
                scored_windows += len(file_scores)
                source_group = path.parent.name
                durations[source_group] = durations.get(source_group, 0.0) + duration
                audio_file_count[source_group] = audio_file_count.get(source_group, 0) + 1
                inventory.append({"recording_id": relative, "bytes": path.stat().st_size,
                                  "sha256": sha256_file(path), "sample_rate_hz": int(sample_rate),
                                  "channels": int(audio.shape[1]), "duration_seconds": duration,
                                  "scored_windows": len(file_scores), "all_windows": len(rows)})
                rows_by_file[relative] = rows
                for row in rows:
                    writer.writerow({"recording_id": relative, "source_group": source_group,
                        "duration_seconds": f"{duration:.6f}", **row,
                        "raw_logit": "" if row["raw_logit"] is None else f"{row['raw_logit']:.9g}"})
            except Exception as exc:
                skipped.append({"recording_id": relative, "error": f"{type(exc).__name__}: {exc}"})
            if number % 50 == 0 or number == len(wavs):
                print(f"scored {number}/{len(wavs)} MUSAN noise files", flush=True)

    if not scores or total_seconds <= 0:
        raise SystemExit("No usable MUSAN noise audio was scored")

    thresholds = threshold_grid(scores, args.threshold_step)
    total_hours = total_seconds / 3600.0
    rows_out = []
    qualifying = []
    for threshold in thresholds:
        # File boundaries reset VoteSmoother, matching separate live sessions.
        count = sum(replay_file(file_rows, [threshold], smoother_windows)[threshold]
                    for file_rows in rows_by_file.values())
        upper = poisson_upper_95(count, total_hours)
        record = {"threshold_logit": threshold, "smoothed_alarm_events": count,
                  "exposure_hours": total_hours, "alarms_per_hour": count / total_hours,
                  "poisson_95pct_upper_per_hour": upper,
                  "meets_musan_only_target": upper <= args.false_alarm_target_per_hour}
        rows_out.append(record)
        if record["meets_musan_only_target"]:
            qualifying.append(threshold)

    with (args.output_dir / "musan_noise_threshold_replay.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)

    inventory_record = {"schema_version": 1, "dataset": "MUSAN noise subset",
        "source_path": str(noise_root), "recordings": inventory, "skipped": skipped}
    inventory_path.write_text(json.dumps(inventory_record, indent=2, allow_nan=False) + "\n",
                              encoding="utf-8")
    selection = json.loads((ROOT / "config/live_detector.json").read_text(encoding="utf-8"))
    report = {
        "status": "SUPPLEMENTAL_ONLY",
        "scope": "MUSAN/noise WAV files only; no music or speech; no V4 train/validation/test split used",
        "deployment_threshold_selected": False,
        "reason_threshold_not_selected": "MUSAN is supplemental downloaded noise, not 24-hour representative live ambient audio; engine/platform and operating environments remain unmeasured",
        "model_id": detector.model_id,
        "checkpoint_sha256": detector.artifact_sha256,
        "frontend_hash": selection["frontend_hash"],
        "smoothing_windows": smoother_windows,
        "decision_cadence_seconds": round(detector.config.hop * round(
            detector.config.decision_interval_s * detector.config.sr / detector.config.hop) /
            detector.config.sr, 6),
        "target_false_alarms_per_hour": args.false_alarm_target_per_hour,
        "recording_count_scored": len(inventory),
        "recording_count_skipped": len(skipped),
        "total_exposure_hours": total_hours,
        "all_windows": total_windows,
        "scored_windows": scored_windows,
        "quiet_windows": total_windows - scored_windows,
        "source_group_summary": {name: {"recordings": audio_file_count[name],
            "duration_hours": durations[name] / 3600.0} for name in sorted(durations)},
        "threshold_grid_step_logit": args.threshold_step,
        "lowest_musan_only_threshold_meeting_target": min(qualifying) if qualifying else None,
        "threshold_replay": rows_out,
        "score_csv_sha256": sha256_file(score_csv),
        "inventory_sha256": sha256_file(inventory_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audio_persisted": False,
        "outputs_local_only": True,
        "artifacts": {"window_scores": str(score_csv),
                      "threshold_replay": str(args.output_dir / "musan_noise_threshold_replay.csv"),
                      "inventory": str(inventory_path)},
    }
    (args.output_dir / "musan_noise_supplement_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "recording_count_scored",
        "recording_count_skipped", "total_exposure_hours", "all_windows", "scored_windows",
        "lowest_musan_only_threshold_meeting_target", "deployment_threshold_selected",
        "artifacts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
