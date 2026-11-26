"""Multiple-label and multiple-offset V2 cache/raw frontend parity gate."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import pytest

from dronedet.config import config_from_mapping
from dronedet.cnn_v3.dataset import cache_tensor_for_offset
from dronedet.features import fit_length, load_wave, mel_power, transform


def test_v3_raw_and_cached_frontends_match_across_labels_and_offsets():
    root = Path(__file__).resolve().parents[2]
    data = root / "results/cnn_v2/data"
    doc = yaml.safe_load((root / "config/cnn_v2.yaml").read_text(encoding="utf-8"))
    fe = config_from_mapping(doc["frontend"])
    manifest = pd.read_csv(data / "manifests/val.csv", keep_default_na=False)
    index = pd.read_csv(data / "cache/index_val.csv")
    stats = np.load(data / "normalization_stats.npz", allow_pickle=False)
    cached_all = np.load(data / "cache/mel_power_val.npy", mmap_mode="r", allow_pickle=False)
    raw_root = Path(os.environ["DRONE_AUDIO_DATASET_ROOT"]) if "DRONE_AUDIO_DATASET_ROOT" in os.environ else root / "archive/CNN_V2_restore_validation_20261126/DroneAudioDataset_binary_frozen_20261126/dataset"
    if not (raw_root / "Binary_Drone_Audio").is_dir():
        pytest.skip("raw dataset unavailable; set DRONE_AUDIO_DATASET_ROOT")
    model = torch.jit.load(str(root / "results/cnn_v2/export/CNN_V2.0/cnn_v2_torchscript.pt"), map_location="cpu")
    model.eval()
    threshold = 2.823960542678833
    comparisons = 0
    for label in (0, 1):
        rows = np.flatnonzero(manifest.label.to_numpy(dtype=int) == label)[:2]
        for row_i in rows:
            row, meta = manifest.iloc[row_i], index.iloc[row_i]
            waveform = fit_length(load_wave(raw_root / "Binary_Drone_Audio" / Path(*str(row.clip_id).split("/")), fe), fe)
            frames = int(meta.n_frames)
            max_offset = frames - fe.n_win
            cached_power = cached_all[:, int(meta.start_frame):int(meta.start_frame) + frames]
            for offset in sorted({0, max_offset // 2, max_offset}):
                window = waveform[offset * fe.hop:offset * fe.hop + fe.win_samples]
                raw = transform(mel_power(window, fe), fe, {"mean": stats["mean"], "std": stats["std"]})
                cached = cache_tensor_for_offset(cached_power, offset, fe, stats["mean"], stats["std"])
                assert raw.shape == cached.shape == (1, fe.n_mels, fe.n_win)
                assert float(np.max(np.abs(raw - cached))) <= 1e-4
                with torch.inference_mode():
                    a = model(torch.from_numpy(raw[None, ...])).cpu().numpy()
                    b = model(torch.from_numpy(cached[None, ...])).cpu().numpy()
                assert float(np.max(np.abs(a - b))) <= 1e-4
                np.testing.assert_array_equal(a >= threshold, b >= threshold)
                comparisons += 1
    assert comparisons == 12
