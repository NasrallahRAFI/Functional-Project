"""Score frozen V4 validation clips at five fixed frame-aligned offsets.

Reads only split-002/val.csv and its referenced source files. The default 5.0
logit is a MUSAN-only candidate for analysis, not a selected live threshold.
All score outputs go to the ignored local/ tree; no audio is copied or written.
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
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dronedet.features import mel_power, standardize, transform  # noqa: E402
from dronedet.inference import load_detector  # noqa: E402

OFFSETS = (0, 8, 16, 24, 31)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grouped_bootstrap_recall(frame: pd.DataFrame, *, seed: int, replicates: int) -> dict:
    positives = frame[frame.label.astype(int) == 1]
    groups = sorted(positives.group.unique())
    if not groups:
        return {"groups": 0, "recall": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    by_group = {group: positives[positives.group == group] for group in groups}
    point = float(positives.detected.mean())
    values = []
    for _ in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        outcomes = [by_group[group].detected.to_numpy() for group in sampled]
        values.append(float(np.concatenate(outcomes).mean()))
    low, high = np.quantile(values, [0.025, 0.975])
    return {"groups": len(groups), "recall": point,
            "ci95_low": float(low), "ci95_high": float(high)}


def metrics(frame: pd.DataFrame, threshold: float, seed: int, replicates: int) -> dict:
    labels = frame.label.to_numpy(dtype=np.int8)
    scores = frame.max_logit.to_numpy(dtype=np.float64)
    detected = scores >= threshold
    positives = labels == 1
    negatives = labels == 0
    safe_roc = bool(positives.any() and negatives.any())
    return {
        "clips": int(len(frame)),
        "positive_clips": int(positives.sum()),
        "negative_clips": int(negatives.sum()),
        "recording_groups": int(frame.group.nunique()),
        "positive_groups": int(frame.loc[positives, "group"].nunique()),
        "roc_auc": float(roc_auc_score(labels, scores)) if safe_roc else None,
        "average_precision": float(average_precision_score(labels, scores)) if safe_roc else None,
        "predicted_positive_clips": int(detected.sum()),
        "recall_at_candidate_logit": float((detected & positives).sum() / max(1, positives.sum())) if positives.any() else None,
        "false_positive_rate_at_candidate_logit": float((detected & negatives).sum() / max(1, negatives.sum())) if negatives.any() else None,
        "group_bootstrap_recall_ci95": grouped_bootstrap_recall(
            frame.assign(detected=detected), seed=seed, replicates=replicates),
    }


def curve_rows(frame: pd.DataFrame, scope: str) -> list[dict]:
    labels = frame.label.to_numpy(dtype=np.int8)
    scores = frame.max_logit.to_numpy(dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return []
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    rows = [{"scope": scope, "curve": "roc", "threshold_logit": float(threshold),
             "x": float(x), "y": float(y)}
            for x, y, threshold in zip(fpr, tpr, roc_thresholds)]
    rows.extend({"scope": scope, "curve": "pr", "threshold_logit": float(pr_thresholds[index]) if index < len(pr_thresholds) else None,
                 "x": float(recall[index]), "y": float(precision[index])}
                for index in range(len(precision)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="MUSAN-only candidate logit to assess; does not activate V4")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT / "local" / "reports" / "study-v4" / "musan-noise-supplement")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or args.bootstrap_replicates < 100:
        raise SystemExit("threshold must be finite and bootstrap replicates at least 100")

    split_registry = json.loads((PROJECT / "repository/registry/splits/split-002.json").read_text(encoding="utf-8"))
    split_root = PROJECT / split_registry["root"]
    val_path = split_root / "val.csv"
    val_entry = next(item for item in split_registry["files"] if item["path"] == "val.csv")
    if sha256_file(val_path) != val_entry["sha256"]:
        raise SystemExit("validation manifest hash differs from frozen split registry")

    detector = load_detector(ROOT)
    if detector.model_id != "CNN_V4.0" or detector.threshold is not None:
        raise SystemExit("This scorer requires the registered V4 raw-score-only selection")
    manifest = pd.read_csv(val_path, keep_default_na=False)
    if len(manifest) != int(val_entry["rows"]) or set(manifest["split"]) != {"val"}:
        raise SystemExit("validation manifest row count or split label is invalid")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    wave_count = 0
    feature_batch: list[np.ndarray] = []
    window_metadata: list[dict] = []
    predictions: list[float] = []

    def flush_batch() -> None:
        if not feature_batch:
            return
        batch = torch.from_numpy(np.stack(feature_batch).astype(np.float32))
        with torch.inference_mode():
            predictions.extend(float(value) for value in detector.model(batch).cpu().numpy().reshape(-1))
        feature_batch.clear()

    for index, row in manifest.iterrows():
        path = Path(str(row["file"]))
        ae_allowed = (PROJECT / "storage/sources/al-emadi/al-emadi-r001/original/Binary_Drone_Audio").resolve()
        dd_allowed = (PROJECT / "storage/derived/corpus-002/dd-clips").resolve()
        dd_source_allowed = (PROJECT / "storage/sources/drone-detection-thesis/thesis-r001/original/Data/Audio").resolve()
        try:
            path.resolve().relative_to(ae_allowed)
            source = "ae"
        except ValueError:
            try:
                path.resolve().relative_to(dd_allowed)
                source = "dd"
            except ValueError:
                raise SystemExit(f"validation row points outside registered source roots: {row['clip_id']}")
        if not path.is_file():
            raise SystemExit(f"validation row points outside registered source roots or is missing: {row['clip_id']}")
        source_path = Path(str(row["source_file"]))
        if source == "ae":
            if sha256_file(path) != str(row["source_file_sha256"]):
                raise SystemExit(f"Al-Emadi validation source hash mismatch: {row['clip_id']}")
        else:
            try:
                source_path.resolve().relative_to(dd_source_allowed)
            except ValueError:
                raise SystemExit(f"DD original source escapes registered source root: {row['clip_id']}")
            if not source_path.is_file() or sha256_file(source_path) != str(row["source_file_sha256"]):
                raise SystemExit(f"DD validation original source hash mismatch: {row['clip_id']}")
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        if source == "dd":
            decoded_digest = hashlib.sha256(np.asarray(audio[:, 0], dtype="<f4").tobytes()).hexdigest()
            if (int(sample_rate) != 16000 or audio.shape[1] != 1
                    or len(audio) != int(row["n_samples"])
                    or decoded_digest != str(row["pcm_sha256"])):
                raise SystemExit(f"DD derived segment identity mismatch: {row['clip_id']}")
        waveform = standardize(audio, int(sample_rate), detector.config)
        wave_count += 1
        clip_windows = []
        clip_metadata = {"clip_id": str(row["clip_id"]), "label": int(row["label"]),
                         "source": str(row["source"]), "sub_class": str(row["sub_class"]),
                         "drone_type": str(row["drone_type"]), "family": str(row["family"]),
                         "group": str(row["group"])}
        for offset in OFFSETS:
            start = offset * detector.config.hop
            stop = start + detector.config.win_samples
            if stop > waveform.size:
                continue
            feature_batch.append(transform(mel_power(waveform[start:stop], detector.config),
                                           detector.config, detector.stats))
            window_metadata.append({**clip_metadata, "offset_frame": offset,
                                    "offset_seconds": start / detector.config.sr,
                                    "window_rms_dbfs": float(20 * np.log10(max(
                                        float(np.sqrt(np.mean(waveform[start:stop].astype(np.float64) ** 2))), 1e-15)))})
            clip_windows.append(len(window_metadata) - 1)
            if len(feature_batch) >= 64:
                flush_batch()
        if not clip_windows:
            raise SystemExit(f"validation clip has no legal requested crop: {row['clip_id']}")
        # Flush each clip so its maximum logit is available for its aggregate row.
        flush_batch()
        clip_scores = predictions[-len(clip_windows):]
        max_logit = max(clip_scores)
        for local_index, global_index in enumerate(clip_windows):
            window_metadata[global_index]["raw_logit"] = clip_scores[local_index]
            window_metadata[global_index]["candidate_vote"] = clip_scores[local_index] >= args.threshold
        rows.append({**clip_metadata, "max_logit": max_logit,
                     "windows_scored": len(clip_windows), "detected": max_logit >= args.threshold})
        predictions.clear()
        if (index + 1) % 100 == 0 or index + 1 == len(manifest):
            print(f"scored {index + 1}/{len(manifest)} validation clips", flush=True)

    clip_frame = pd.DataFrame(rows)
    window_path = out / "v4_validation_window_scores.csv"
    clip_path = out / "v4_validation_clip_scores.csv"
    curves_path = out / "v4_validation_roc_pr_curves.csv"
    pd.DataFrame(window_metadata).to_csv(window_path, index=False, lineterminator="\n")
    clip_frame.to_csv(clip_path, index=False, lineterminator="\n")

    curves = []
    summary = {"pooled": metrics(clip_frame, args.threshold, args.seed, args.bootstrap_replicates)}
    curves.extend(curve_rows(clip_frame, "clip_max:pooled"))
    window_frame = pd.DataFrame(window_metadata)
    window_frame = window_frame.merge(
        clip_frame[["clip_id", "label", "source", "sub_class", "drone_type", "family", "group"]],
        on=["clip_id", "label", "source", "sub_class", "drone_type", "family", "group"],
        how="left", validate="many_to_one")
    curves.extend(curve_rows(window_frame.rename(columns={"raw_logit": "max_logit"}),
                             "window:pooled"))
    for column in ("source", "sub_class", "drone_type", "family"):
        for value, subset in clip_frame.groupby(column, dropna=False):
            label = "(empty)" if value == "" else str(value)
            key = f"{column}={label}"
            summary[key] = metrics(subset, args.threshold, args.seed, args.bootstrap_replicates)
            curves.extend(curve_rows(subset, f"clip_max:{key}"))
            window_subset = window_frame[window_frame[column] == value].rename(
                columns={"raw_logit": "max_logit"})
            curves.extend(curve_rows(window_subset, f"window:{key}"))
    pd.DataFrame(curves).to_csv(curves_path, index=False, lineterminator="\n")
    report = {
        "status": "VALIDATION_ONLY_SUPPLEMENTAL_CANDIDATE",
        "threshold_selected": False,
        "candidate_logit_from_musan_only_grid": args.threshold,
        "recall_floor": 0.80,
        "model_id": detector.model_id,
        "checkpoint_sha256": detector.artifact_sha256,
        "validation_manifest_sha256": val_entry["sha256"],
        "validation_rows_read": len(manifest),
        "validation_audio_files_opened": wave_count,
        "test_manifest_or_audio_opened": False,
        "offset_frames": list(OFFSETS),
        "note": "Clip-level recall and clip-max ROC/PR treat a clip as detected if any requested crop crosses the candidate logit. Window-level ROC/PR are also exported, with correlated crops retained within recording groups. Overlapping crops are not a live VoteSmoother sequence. Intervals resample recording groups.",
        "bootstrap_seed": args.seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "results": summary,
        "artifacts": {"window_scores": str(window_path), "clip_scores": str(clip_path),
                      "roc_pr_curves": str(curves_path)},
        "audio_persisted": False,
        "outputs_local_only": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "v4_validation_candidate_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "threshold_candidate": args.threshold,
        "test_access": False, "pooled": summary["pooled"],
        "musan_heli_hard_negative": summary.get("sub_class=HELICOPTER"),
        "artifacts": report["artifacts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
