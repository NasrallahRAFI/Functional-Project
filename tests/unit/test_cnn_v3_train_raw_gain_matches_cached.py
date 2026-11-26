"""Check the exact A0a crop/gain recipe across raw and cached inputs."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from dronedet.config import config_from_mapping
from dronedet.features import fit_length, load_wave, mel_power, to_db, transform


def test_matched_training_offsets_and_gains_reproduce_cached_tensors():
    root = Path(__file__).resolve().parents[2]
    doc = yaml.safe_load((root / "config/cnn_v3.yaml").read_text(encoding="utf-8"))
    fe = config_from_mapping(doc["frontend"])
    data = root / "results/cnn_v2/data"
    manifest = pd.read_csv(data / "manifests/train.csv", keep_default_na=False)
    index = pd.read_csv(data / "cache/index_train.csv")
    power = np.load(data / "cache/mel_power_train.npy", mmap_mode="r", allow_pickle=False)
    stats = np.load(data / "normalization_stats.npz", allow_pickle=False)
    from dronedet.cnn_v3.dataset import cache_tensor_for_offset

    env_root = __import__("os").environ.get("DRONE_AUDIO_DATASET_ROOT")
    raw_root = Path(env_root) if env_root else root / "archive/CNN_V2_restore_validation_20261126/DroneAudioDataset_binary_frozen_20261126/dataset"
    if not (raw_root / "Binary_Drone_Audio").is_dir():
        pytest.skip("raw dataset unavailable; set DRONE_AUDIO_DATASET_ROOT")
    targets = {0, 17, 501, 1, 15, 34, 99, 205, 1000, 2000, 3000, 4500, 6000, 7606}
    epoch_seed = 42 * 1_000_003
    rng = np.random.default_rng(epoch_seed)
    order = np.random.default_rng(epoch_seed ^ 0x5EED).permutation(len(manifest))
    draws = {}
    for batch_start in range(0, len(order), 64):
        ids = order[batch_start:batch_start + 64]
        offsets = [int(rng.integers(0, int(index.iloc[i].n_frames) - fe.n_win + 1)) for i in ids]
        gains = rng.uniform(-12.0, 12.0, len(ids))
        for i, offset, gain in zip(ids, offsets, gains):
            if int(i) in targets:
                draws[int(i)] = (offset, gain)
        if targets.issubset(draws):
            break
    for i in sorted(targets):
        offset, gain = draws[i]
        row, meta = manifest.iloc[i], index.iloc[i]
        audio = fit_length(load_wave(raw_root / "Binary_Drone_Audio" / Path(*str(row.clip_id).split("/")), fe), fe)
        audio = audio * np.float32(10.0 ** (float(gain) / 20.0))
        window = audio[offset * fe.hop:offset * fe.hop + fe.win_samples]
        raw = transform(mel_power(window, fe), fe, {"mean": stats["mean"], "std": stats["std"]})
        clip_power = power[:, int(meta.start_frame):int(meta.start_frame) + int(meta.n_frames)]
        clip_power = clip_power * np.float32(10.0 ** (float(gain) / 10.0))
        cached = cache_tensor_for_offset(clip_power, offset, fe, stats["mean"], stats["std"])
        # Allow floating point FFT/magnitude rounding, but require near-identical model input.
        np.testing.assert_allclose(raw, cached, rtol=0, atol=1e-4)
