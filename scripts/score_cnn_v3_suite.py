"""Score the frozen V2 model on the fixed CNN V3 validation suite only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml

from dronedet.config import config_from_mapping
from dronedet.features import transform, to_db
from dronedet.cnn_v3.dataset import cache_tensor_for_offset
from dronedet.metrics import metrics_at_threshold


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _metrics(labels: np.ndarray, logits: np.ndarray, threshold: float) -> dict:
    return metrics_at_threshold(labels.astype(int), logits.astype(float), threshold)


def score(project: Path) -> dict:
    plan = yaml.safe_load((project / "config/cnn_v3.yaml").read_text(encoding="utf-8"))
    fe = config_from_mapping(plan["frontend"])
    suite = project / plan["output_root"] / "validation_suite"
    manifest_path = suite / "SUITE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN" or manifest.get("count_per_condition") != {k: 1510 for k in ("mild", "moderate", "severe")}:
        raise RuntimeError("fixed validation suite is missing or incomplete")
    if sha(manifest_path) != (suite / "SUITE_MANIFEST.sha256").read_text(encoding="ascii").split()[0]:
        raise RuntimeError("validation suite manifest hash mismatch")
    v2data = project / "results/cnn_v2/data"
    val = pd.read_csv(v2data / "manifests/val.csv", keep_default_na=False)
    index = pd.read_csv(v2data / "cache/index_val.csv")
    power = np.load(v2data / "cache/mel_power_val.npy", mmap_mode="r", allow_pickle=False)
    stats = np.load(v2data / "normalization_stats.npz", allow_pickle=False)
    mean, std = stats["mean"], stats["std"]
    model_path = project / "results/cnn_v2/export/CNN_V2.0/cnn_v2_torchscript.pt"
    model = torch.jit.load(str(model_path), map_location="cpu"); model.eval()
    threshold = 2.823960542678833
    rows = []
    for i, source in val.iterrows():
        meta = index.iloc[i]
        p = power[:, int(meta.start_frame):int(meta.start_frame) + int(meta.n_frames)]
        x = cache_tensor_for_offset(p, int(meta.center_start_frame), fe, mean, std)[None, ...]
        with torch.inference_mode():
            score_clean = float(model(torch.from_numpy(np.ascontiguousarray(x)))[0].item())
        rows.append({"condition": "clean", "clip_id": str(source.clip_id), "group": str(source.group),
                     "label": int(source.label), "logit": score_clean})
    for item in manifest["examples"]:
        wav = suite / item["path"]
        if sha(wav) != item["sha256"]:
            raise RuntimeError(f"validation suite member hash mismatch: {item['path']}")
        audio, sr = sf.read(wav, dtype="float32", always_2d=False)
        if sr != fe.sr or audio.ndim != 1 or not np.isfinite(audio).all():
            raise RuntimeError(f"invalid fixed validation view: {item['path']}")
        # Match the registered centered 0.496 s window policy.
        n = fe.win_samples
        if len(audio) < n:
            audio = np.pad(audio, (0, n-len(audio)), mode="wrap")
        offset = ((len(audio) - n) // 2 // fe.hop) * fe.hop
        window = audio[offset:offset+n]
        from dronedet.features import mel_power
        p = mel_power(window, fe)
        x = transform(p, fe, {"mean": mean, "std": std})[None, ...]
        with torch.inference_mode():
            logit = float(model(torch.from_numpy(np.ascontiguousarray(x)))[0].item())
        rows.append({"condition": item["severity"], "clip_id": item["clip_id"],
                     "group": item["group"], "label": int(item["label"]), "logit": logit})
    scores = pd.DataFrame(rows)
    clean = scores[scores.condition == "clean"]
    condition_metrics = {name: _metrics(g.label.to_numpy(), g.logit.to_numpy(), threshold)
                         for name, g in scores.groupby("condition", sort=False)}
    weights = {"clean": 0.5, "mild": 1/6, "moderate": 1/6, "severe": 1/6}
    sample_weights = scores.condition.map(weights).to_numpy(dtype=np.float64) / len(val)
    y = scores.label.to_numpy(dtype=np.float64)
    pred = (scores.logit.to_numpy() >= threshold).astype(np.float64)
    tp = float(np.sum(sample_weights * y * pred)); fp = float(np.sum(sample_weights * (1-y) * pred))
    fn = float(np.sum(sample_weights * y * (1-pred))); tn = float(np.sum(sample_weights * (1-y) * (1-pred)))
    pooled_precision = tp / max(tp + fp, 1e-15); pooled_recall = tp / max(tp + fn, 1e-15)
    pooled = {"threshold": threshold, "tp_weight": tp, "fp_weight": fp, "fn_weight": fn, "tn_weight": tn,
              "precision": pooled_precision, "recall": pooled_recall,
              "f1": 2*pooled_precision*pooled_recall/max(pooled_precision+pooled_recall, 1e-15)}
    output = project / "results/cnn_v3/validation_suite/V2_reference"
    output.mkdir(parents=True, exist_ok=False)
    scores_path = output / "scores.csv"
    scores.to_csv(scores_path, index=False, lineterminator="\n")
    record = {"status": "COMPLETE", "model": "CNN_V2.0", "model_sha256": sha(model_path),
              "suite_manifest_sha256": sha(manifest_path), "scores_sha256": sha(scores_path),
              "threshold": threshold, "score_space": "raw_logit", "clean_metrics": condition_metrics["clean"],
              "condition_metrics": condition_metrics, "weighted_pooled_metrics": pooled,
              "row_count": len(scores), "test_access": False}
    (output / "report.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    score(Path(__file__).resolve().parents[1])
