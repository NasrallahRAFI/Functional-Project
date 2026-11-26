"""Prepare hash-verified validation feature tensors for weighted-loss monitoring."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from dronedet.config import config_from_mapping
from dronedet.features import mel_power, transform


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def prepare(root: Path) -> dict:
    doc = yaml.safe_load((root / "config/cnn_v3.yaml").read_text(encoding="utf-8"))
    fe = config_from_mapping(doc["frontend"])
    data = root / "results/cnn_v2/data"
    stats = np.load(data / "normalization_stats.npz", allow_pickle=False)
    suite = root / "results/cnn_v3/validation_suite"
    manifest_path = suite / "SUITE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN":
        raise RuntimeError("validation suite is not frozen")
    val = pd.read_csv(data / "manifests/val.csv", keep_default_na=False)
    index = pd.read_csv(data / "cache/index_val.csv")
    cache = np.load(data / "cache/mel_power_val.npy", mmap_mode="r", allow_pickle=False)
    groups = {"clean": [], "mild": [], "moderate": [], "severe": []}
    # Build the clean view through the frozen V2 feature cache.
    for i, meta in index.iterrows():
        start = int(meta.start_frame) + int(meta.center_start_frame)
        p = cache[:, start:start+fe.n_win]
        groups["clean"].append(transform(p, fe, {"mean": stats["mean"], "std": stats["std"]}))
    labels = {"clean": val.label.to_numpy(dtype=np.int64)}
    for condition in ("mild", "moderate", "severe"):
        rows = [x for x in manifest["examples"] if x["severity"] == condition]
        if len(rows) != len(val):
            raise RuntimeError(f"{condition} suite does not contain every validation clip")
        if [str(x["clip_id"]) for x in rows] != val.clip_id.astype(str).tolist():
            raise RuntimeError(f"{condition} suite clip order differs from the frozen validation manifest")
        for row in rows:
            path = suite / row["path"]
            if sha(path) != row["sha256"]:
                raise RuntimeError(f"suite member hash mismatch: {row['path']}")
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
            if sr != fe.sr or audio.ndim != 1 or not np.isfinite(audio).all():
                raise RuntimeError(f"invalid suite audio: {path}")
            if len(audio) < fe.win_samples:
                audio = np.pad(audio, (0, fe.win_samples-len(audio)), mode="wrap")
            offset = ((len(audio)-fe.win_samples)//2//fe.hop)*fe.hop
            p = mel_power(audio[offset:offset+fe.win_samples], fe)
            groups[condition].append(transform(p, fe, {"mean": stats["mean"], "std": stats["std"]}))
        labels[condition] = np.asarray([int(x["label"]) for x in rows], dtype=np.int64)
    for name in groups:
        groups[name] = np.ascontiguousarray(np.stack(groups[name]), dtype=np.float32)
        if groups[name].shape != (len(val), 1, fe.n_mels, fe.n_win):
            raise RuntimeError(f"invalid tensor geometry for {name}: {groups[name].shape}")
        if not np.array_equal(labels[name], val.label.to_numpy(dtype=np.int64)):
            raise RuntimeError(f"label/order mismatch for {name}")
    out = suite / "tensors.npz"
    np.savez_compressed(out, **{f"x_{k}": v for k, v in groups.items()}, **{f"y_{k}": v for k, v in labels.items()})
    record = {"status": "PASS", "suite_manifest_sha256": sha(manifest_path),
              "tensors_sha256": sha(out), "rows_per_condition": len(val),
              "conditions": list(groups), "test_access": False}
    (suite / "TENSOR_CACHE_REPORT.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[1])
