"""Rebuild the V3 training-only mel-power cache directly from raw training WAVs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dronedet.config import config_from_mapping
from dronedet.features import file_to_power
from dronedet.splits import guard_test_access


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(root: Path) -> dict:
    doc = yaml.safe_load((root / "config/cnn_v3.yaml").read_text(encoding="utf-8"))
    fe = config_from_mapping(doc["frontend"])
    data = root / "results/cnn_v2/data"
    manifest_path = data / "manifests/train.csv"
    train_index_path = data / "cache/index_train.csv"
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    index = pd.read_csv(train_index_path, keep_default_na=False)
    if len(manifest) != 7607 or len(index) != 7607:
        raise RuntimeError("training manifest/index count differs from frozen split")
    if not np.array_equal(manifest.clip_id.astype(str).to_numpy(), index.clip_id.astype(str).to_numpy()):
        raise RuntimeError("training manifest and feature index clip order differ")
    guard_test_access("train", allow_test=False)
    raw_value = os.environ.get("DRONE_AUDIO_DATASET_ROOT", doc["raw_root"])
    raw_root = Path(raw_value)
    if not raw_root.is_absolute():
        raw_root = root / raw_root
    raw_root = raw_root.resolve() / "Binary_Drone_Audio"
    chunks = []
    for i, row in enumerate(manifest.itertuples(index=False)):
        power, meta = file_to_power(raw_root / Path(*str(row.clip_id).split("/")), fe)
        expected_frames = int(index.iloc[i].n_frames)
        if power.shape != (fe.n_mels, expected_frames):
            raise RuntimeError(f"raw frontend frame count differs for {row.clip_id}")
        chunks.append(power)
        if (i + 1) % 1000 == 0:
            print(f"rebuilt {i+1}/7607 training clips", flush=True)
    raw_power = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    v2 = np.load(data / "cache/mel_power_train.npy", mmap_mode="r", allow_pickle=False)
    if raw_power.shape != v2.shape:
        raise RuntimeError(f"raw cache shape differs from V2: {raw_power.shape} != {v2.shape}")
    diff = np.abs(raw_power - v2)
    exact = bool(np.array_equal(raw_power, v2))
    max_error = float(diff.max(initial=0.0))
    out = root / "results/cnn_v3/prepared"
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "raw_power_train.npy"
    np.save(cache_path, raw_power, allow_pickle=False)
    report = {"status": "PASS" if exact else "DIFFERENT_FROM_FROZEN_V2_CACHE",
              "split": "train", "count": len(manifest), "shape": list(raw_power.shape),
              "raw_cache_sha256": sha(cache_path), "v2_cache_sha256": sha(data / "cache/mel_power_train.npy"),
              "bitwise_equal": exact, "max_abs_power_error": max_error,
              "train_manifest_sha256": sha(manifest_path), "train_index_sha256": sha(train_index_path),
              "test_access": False}
    (out / "raw_train_cache_report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[1])
