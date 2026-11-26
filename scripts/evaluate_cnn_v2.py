"""Score the frozen CNN V2 checkpoint once on the reused V1 test benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
import yaml

from dronedet.config import config_from_mapping
from dronedet.features import to_db
from dronedet.metrics import metrics_at_threshold
from dronedet.models.cnn import build_cnn


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def batch_tensor(power: np.ndarray, starts: np.ndarray, mean: np.ndarray, std: np.ndarray, fe) -> np.ndarray:
    crops = np.stack([power[:, int(s):int(s) + fe.n_win] for s in starts]).astype(np.float32)
    db = to_db(crops, fe)
    return np.ascontiguousarray(((db - mean[None, :, None]) / std[None, :, None])[:, None], dtype=np.float32)


def group_bootstrap(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, *, seed: int, accepted: int = 1000) -> dict:
    group_names = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in group_names}
    rng = np.random.default_rng(seed)
    values = {key: [] for key in ("f1", "accuracy", "recall", "false_alarm_rate", "miss_rate")}
    attempts = skipped = 0
    while len(values["f1"]) < accepted and attempts < accepted * 20:
        attempts += 1
        sample_groups = rng.choice(group_names, size=len(group_names), replace=True)
        ix = np.concatenate([members[g] for g in sample_groups])
        labels = y[ix]
        if np.unique(labels).size != 2:
            skipped += 1
            continue
        tp = int(np.sum((labels == 1) & (pred[ix] == 1)))
        fp = int(np.sum((labels == 0) & (pred[ix] == 1)))
        fn = int(np.sum((labels == 1) & (pred[ix] == 0)))
        tn = int(np.sum((labels == 0) & (pred[ix] == 0)))
        values["f1"].append(2 * tp / max(1, 2 * tp + fp + fn))
        values["accuracy"].append((tp + tn) / max(1, len(ix)))
        values["recall"].append(tp / max(1, tp + fn))
        values["false_alarm_rate"].append(fp / max(1, fp + tn))
        values["miss_rate"].append(fn / max(1, fn + tp))
    result = {key: {"low_2_5": float(np.percentile(vals, 2.5)),
                    "high_97_5": float(np.percentile(vals, 97.5)),
                    "accepted_draws": len(vals)} for key, vals in values.items()}
    return {"seed": seed, "target_accepted_draws": accepted, "attempts": attempts,
            "class_degenerate_skipped": skipped, "complete": all(len(v) == accepted for v in values.values()),
            "metrics": result}


def evaluate(config_path: Path, freeze_path: Path) -> None:
    cp, fp = config_path.resolve(), freeze_path.resolve()
    doc = yaml.safe_load(cp.read_text(encoding="utf-8"))
    fe = config_from_mapping(doc["frontend"])
    torch.set_num_threads(int(doc["threads"]))
    torch.use_deterministic_algorithms(True, warn_only=True)
    root = cp.parent.parent
    data = root / doc["output_root"] / "data"
    out = root / doc["output_root"] / "evaluation"
    freeze = json.loads(fp.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_TEST" or freeze.get("score_space") != "raw_logit":
        raise RuntimeError("test access requires an immutable raw-logit freeze record")
    ckpt = fp.parent / freeze["checkpoint"]
    if sha256(ckpt) != freeze["checkpoint_sha256"]:
        raise RuntimeError("frozen checkpoint digest mismatch")
    if sha256(data / "DATASET_READY.json") != freeze["dataset_ready_sha256"]:
        raise RuntimeError("frozen dataset readiness identity mismatch")
    if freeze.get("test_disclosure") != "V1 benchmark reused; not an independent unseen test":
        raise RuntimeError("required reused benchmark disclosure is absent")
    if out.exists() and (out / "scoring.lock.json").exists():
        raise FileExistsError("CNN V2 evaluation is already complete; refusing a second scoring pass")
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(data / "manifests/test.csv", keep_default_na=False)
    index = pd.read_csv(data / "cache/index_test.csv")
    v1_lock_path = root / "results/tables/test_scored.lock"
    v1_lock = json.loads(v1_lock_path.read_text(encoding="utf-8"))
    test_manifest_hash = sha256(data / "manifests/test.csv")
    v1_manifest = root / doc["reference_splits"] / "test.csv"
    if sha256(v1_manifest) != v1_lock.get("test_manifest_sha256"):
        raise RuntimeError("historical V1 test lock does not match its preserved manifest")
    v1_test = pd.read_csv(v1_manifest, keep_default_na=False)
    for field in ("clip_id", "label", "group"):
        if not v1_test[field].astype(str).sort_values().reset_index(drop=True).equals(
                manifest[field].astype(str).sort_values().reset_index(drop=True)):
            raise RuntimeError(f"V2 test {field} membership differs from the previously scored V1 benchmark")
    ready_record = json.loads((data / "DATASET_READY.json").read_text(encoding="utf-8"))
    if len(manifest) != 1506 or len(index) != len(manifest):
        raise RuntimeError("test membership differs from the frozen 1,506-clip contract")
    ready_hashes = {r["path"]: r["sha256"] for r in ready_record["files"]}
    for rel in ("manifests/test.csv", "cache/index_test.csv", "cache/mel_power_test.npy"):
        if sha256(data / rel) != ready_hashes.get(rel):
            raise RuntimeError(f"test artifact checksum does not match the ready package: {rel}")
    power = np.load(data / "cache/mel_power_test.npy", mmap_mode="r", allow_pickle=False)
    stats = np.load(data / "normalization_stats.npz", allow_pickle=False)
    starts = index.start_frame.to_numpy(dtype=np.int64) + index.center_start_frame.to_numpy(dtype=np.int64)
    y = manifest.label.to_numpy(dtype=np.int8)
    if power.shape != (64, 91494) or len(y) != 1506:
        raise RuntimeError("test cache dimensions do not match frozen test identity")
    model = build_cnn((1, 64, 28), channels=(16, 32, 64, 64), kernel_size=3,
                      pool=(2, 2), head="global_average", dropout=0.3)
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"]); model.eval()
    partial = out / "scores.partial.csv"
    state_path = out / "scoring_state.json"
    seen: set[str] = set()
    saved: list[dict] = []
    score_path = out / "scores.csv"
    existing_score_file = score_path if score_path.exists() else partial if partial.exists() else None
    if existing_score_file is not None:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("checkpoint_sha256") != freeze["checkpoint_sha256"] or state.get("test_manifest_sha256") != test_manifest_hash:
            raise RuntimeError("partial test-score identity mismatch; refusing unsafe recovery")
        saved = pd.read_csv(existing_score_file).to_dict("records")
        seen = {str(r["clip_id"]) for r in saved}
        if len(seen) != len(saved) or any(cid not in set(manifest.clip_id) for cid in seen):
            raise RuntimeError("partial score rows are duplicated or outside the frozen test set")
    remaining = [i for i, cid in enumerate(manifest.clip_id) if cid not in seen]
    started = time.perf_counter()
    for first in range(0, len(remaining), 64):
        ix = np.asarray(remaining[first:first + 64], dtype=np.int64)
        xb = torch.from_numpy(batch_tensor(power, starts[ix], stats["mean"], stats["std"], fe))
        with torch.no_grad():
            logits = model(xb).cpu().numpy().astype(np.float64)
        for j, row_id in enumerate(ix):
            row, idx = manifest.iloc[int(row_id)], index.iloc[int(row_id)]
            saved.append({"clip_id": row.clip_id, "label": int(row.label), "group": row.group,
                          "family": row.family, "drone_type": row.drone_type,
                          "silent": bool(idx.silent), "near_silent": bool(idx.near_silent),
                          "partial_zero": bool(row.partial_zero), "clipped": bool(row.clipped),
                          "crop_all_zero": bool(idx.crop_all_zero), "logit": float(logits[j]),
                          "prediction": int(logits[j] >= float(freeze["validation_threshold"]["threshold"]))})
        partial_tmp = partial.with_suffix(".csv.tmp")
        pd.DataFrame(saved).to_csv(partial_tmp, index=False, lineterminator="\n")
        os.replace(partial_tmp, partial)
        state_tmp = state_path.with_suffix(".json.tmp")
        state_tmp.write_text(json.dumps({"checkpoint_sha256": freeze["checkpoint_sha256"],
            "test_manifest_sha256": test_manifest_hash, "scored_rows": len(saved)}, indent=2) + "\n")
        os.replace(state_tmp, state_path)
        print(f"Scored {len(saved):,}/{len(manifest):,} test clips.", flush=True)
    scores = pd.DataFrame(saved)
    expected_ids = list(manifest.clip_id)
    scores = scores.set_index("clip_id").loc[expected_ids].reset_index()
    if partial.exists():
        os.replace(partial, score_path)
    y = scores.label.to_numpy(dtype=np.int8)
    logits = scores.logit.to_numpy(dtype=np.float64)
    pred = scores.prediction.to_numpy(dtype=np.int8)
    threshold = float(freeze["validation_threshold"]["threshold"])
    metrics = metrics_at_threshold(y, logits, threshold)
    metrics.update({"accuracy": float(accuracy_score(y, pred)),
                    "pr_auc": float(average_precision_score(y, logits))})
    boot = group_bootstrap(y, pred, scores.group.to_numpy(str), seed=20261126)
    stratified = {}
    for dimension in ("family", "drone_type"):
        stratified[dimension] = {}
        for name, subset in scores.groupby(dimension, dropna=False):
            yy, pp = subset.label.to_numpy(int), subset.prediction.to_numpy(int)
            pos = int((yy == 1).sum()); tp = int(((yy == 1) & (pp == 1)).sum())
            stratified[dimension][str(name)] = {
                "clips": len(subset), "groups": int(subset.group.nunique()),
                "positive_clips": pos, "positive_recall": tp / pos if pos else None,
                "interval_status": "descriptive_small_family" if subset.group.nunique() < 3 else "not_bootstrapped"}
    stratified["extra_membo"] = "not evaluated; train-only under the frozen split"
    sensitivity = {}
    for flag in ("silent", "near_silent", "partial_zero", "clipped", "crop_all_zero"):
        vals = scores[flag].astype(bool)
        cells = {}
        for label, sub in (("flagged", scores[vals]), ("unflagged", scores[~vals])):
            yy, pp = sub.label.to_numpy(int), sub.prediction.to_numpy(int)
            positives = int((yy == 1).sum())
            cells[label] = {"clips": len(sub), "groups": int(sub.group.nunique()),
                            "positive_clips": positives,
                            "positive_recall": int(((yy == 1) & (pp == 1)).sum()) / positives if positives else None}
        sensitivity[flag] = cells
    test_far_rates = {}
    for far, val_point in freeze["validation_far_operating_points"].items():
        op_threshold = float(val_point["threshold"])
        test_far_rates[far] = {"validation_threshold": op_threshold,
                               "test_metrics": metrics_at_threshold(y, logits, op_threshold)}
    lock = {"status": "COMPLETE", "model": "one CNN V2.0 depth_4 checkpoint",
            "checkpoint_sha256": freeze["checkpoint_sha256"], "freeze_sha256": sha256(fp),
            "dataset_ready_sha256": freeze["dataset_ready_sha256"],
            "test_manifest_sha256": test_manifest_hash,
            "reused_v1_test_lock_sha256": sha256(v1_lock_path),
            "scores_sha256": sha256(score_path), "score_rows": len(scores),
            "threshold": threshold, "score_space": "raw_logit", "metrics": metrics,
            "validation_far_operating_points_realized_on_test": test_far_rates,
            "group_bootstrap_95_percent": boot, "family_and_type_sensitivity": stratified,
            "quality_flag_sensitivity": sensitivity,
            "benchmark_disclosure": "reused V1 benchmark; not an independent unseen test",
            "test_inference_passes": 1, "scoring_seconds": time.perf_counter() - started}
    (out / "report.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    tmp = out / "scoring.lock.json.tmp"
    tmp.write_text(json.dumps({k: v for k, v in lock.items() if k in ("status", "model", "checkpoint_sha256", "freeze_sha256", "test_manifest_sha256", "reused_v1_test_lock_sha256", "scores_sha256", "score_rows", "test_inference_passes", "benchmark_disclosure")}, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, out / "scoring.lock.json")
    state_path.unlink(missing_ok=True)
    print(json.dumps({"status": "COMPLETE", "metrics": metrics, "bootstrap_complete": boot["complete"],
                      "scores": str(score_path), "report": str(out / "report.json")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.config, args.freeze)


if __name__ == "__main__":
    main()
