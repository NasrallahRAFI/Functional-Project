"""Contract tests for the registered V3 waveform augmentation stages."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dronedet.cnn_v3.augment import (
    AugmentConfig,
    ChannelAugment,
    NoiseBank,
    NoiseDonor,
    band_limit,
    colored_noise,
    rms_db,
    room_response,
    smooth_eq,
    stable_rng,
)


def test_seeded_channel_view_is_repeatable_finite_and_label_independent():
    t = np.arange(16000, dtype=np.float64) / 16000
    x = (0.03 * np.sin(2 * np.pi * 220 * t) + 0.02 * np.sin(2 * np.pi * 2400 * t)).astype(np.float32)
    augmenter = ChannelAugment(config=AugmentConfig(p_clean_recipe=0.0))
    a, pa = augmenter(x, stable_rng(42, 0, "train/a.wav", 2), label=1)
    b, pb = augmenter(x, stable_rng(42, 0, "train/a.wav", 2), label=0)
    assert a.shape == x.shape
    assert a.dtype == np.float32
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)
    assert pa == pb


def test_stable_rng_separates_repeated_sampler_occurrences():
    a = stable_rng(42, 3, "train/a.wav", 0).normal(size=8)
    b = stable_rng(42, 3, "train/a.wav", 1).normal(size=8)
    c = stable_rng(42, 3, "train/a.wav", 0).normal(size=8)
    np.testing.assert_array_equal(a, c)
    assert not np.array_equal(a, b)


def test_v2_recipe_branch_bypasses_channel_noise_and_limiter():
    t = np.arange(16000, dtype=np.float32) / 16000
    x = 0.2 * np.sin(2 * np.pi * 220 * t)
    cfg = AugmentConfig(p_clean_recipe=1.0, p_floor=1.0, p_noise=1.0,
                        p_band_limit=1.0, p_eq=1.0, p_soft_clip=1.0, p_reverb=1.0)
    out, record = ChannelAugment(config=cfg)(x, stable_rng(19, 0, "train/clean.wav"))
    expected = x * np.float32(10 ** (record["gain_db"] / 20.0))
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-7)
    assert record["branch"] == "v2_recipe"


def test_channel_path_preserves_over_range_float_gain_without_per_clip_normalization():
    x = np.full(4096, 0.8, dtype=np.float32)
    cfg = AugmentConfig(p_clean_recipe=0.0, p_band_limit=0.0, p_eq=0.0,
                        p_soft_clip=0.0, p_reverb=0.0, p_noise=0.0, p_floor=0.0,
                        gain_db=(6.0, 6.0))
    out, record = ChannelAugment(config=cfg)(x, stable_rng(23, 0, "train/gain.wav"))
    assert float(np.max(np.abs(out))) > 1.0
    assert record["final_limiter_used"] is False


def test_microphone_floor_respects_total_added_snr_for_near_silent_nonzero_audio():
    x = np.full(4096, 1e-5, dtype=np.float32)
    cfg = AugmentConfig(p_clean_recipe=0.0, p_band_limit=0.0, p_eq=0.0,
                        p_soft_clip=0.0, p_reverb=0.0, p_noise=0.0, p_floor=1.0,
                        gain_db=(0.0, 0.0), floor_dbfs=(-45.0, -45.0),
                        minimum_positive_snr_db=0.0)
    out, record = ChannelAugment(config=cfg)(x, stable_rng(31, 0, "train/near_silence.wav"), label=1)
    assert record["nuisance_limited"] is True
    assert record["measured_added_noise_snr_db"] >= -1e-5
    assert np.isfinite(out).all()


def test_band_limit_attenuates_out_of_band_energy_and_preserves_shape():
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr
    x = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 3000 * t)
    y, params = band_limit(x, np.random.default_rng(7), hp_hz=(500, 500), lp_hz=(4000, 4000))
    assert y.shape == x.shape
    assert np.isfinite(y).all()
    assert params["hp_hz"] == pytest.approx(500.0)
    assert abs(rms_db(y)) < 6
    assert np.std(y[1000:]) < np.std(x[1000:])


def test_smooth_eq_is_finite_length_preserving_and_reproducible():
    rng = np.random.default_rng(9)
    x = rng.standard_normal(4096)
    a, pa = smooth_eq(x, np.random.default_rng(11))
    b, pb = smooth_eq(x, np.random.default_rng(11))
    assert a.shape == x.shape
    np.testing.assert_allclose(a, b, rtol=0, atol=0)
    assert pa == pb
    assert np.isfinite(a).all()


def test_room_tail_has_requested_energy_ratio_and_length():
    x = np.zeros(16000, dtype=np.float64)
    x[4000] = 1.0
    y, p = room_response(x, np.random.default_rng(12), rt60_s=(0.5, 0.5), drr_db=(3.0, 3.0))
    assert y.shape == x.shape
    assert np.isfinite(y).all()
    assert p["rt60_s"] == 0.5
    # The input is a unit impulse, so output energies expose the IR ratio.
    expected_ratio = 10 ** (-3.0 / 10.0)
    measured_ratio = float(np.sum(y[4001:] ** 2)) / float(y[4000] ** 2)
    assert measured_ratio == pytest.approx(expected_ratio, rel=5e-5)
    assert np.max(np.abs(y)) > 0


def test_colored_noise_is_finite_zero_mean_and_unit_rms():
    x = colored_noise(32000, 1.0, np.random.default_rng(4))
    assert np.isfinite(x).all()
    assert abs(float(np.mean(x))) < 1e-10
    assert float(np.sqrt(np.mean(x * x))) == pytest.approx(1.0, abs=1e-10)


def test_noise_bank_rejects_silent_donors_and_keeps_provenance():
    bank = NoiseBank()
    with pytest.raises(ValueError, match="excluded"):
        bank.add(NoiseDonor("unknown/silence.wav", "g0", np.zeros(1024), "abc"))
    donor = np.random.default_rng(1).normal(0, 0.01, 4000).astype(np.float32)
    bank.add(NoiseDonor("unknown/noise.wav", "g1", donor, "sha"))
    drawn, record = bank.draw(2000, np.random.default_rng(8))
    assert drawn.shape == (2000,)
    assert np.isfinite(drawn).all()
    assert record["kind"] in {"training_negative", "procedural"}
    if record["kind"] == "training_negative":
        assert record["group_id"] == "g1"


def test_v3_raw_validation_waveform_matches_v2_cached_tensor():
    from pathlib import Path
    import pandas as pd
    import yaml

    from dronedet.config import config_from_mapping
    from dronedet.features import fit_length, load_wave, mel_power, transform
    from dronedet.cnn_v3.dataset import cache_tensor_for_offset, frame_aligned_crop

    root = Path(__file__).resolve().parents[2]
    recipe = yaml.safe_load((root / "config/cnn_v2.yaml").read_text(encoding="utf-8"))
    config = config_from_mapping(recipe["frontend"])
    data = root / "results/cnn_v2/data"
    manifest = pd.read_csv(data / "manifests/val.csv", keep_default_na=False)
    index = pd.read_csv(data / "cache/index_val.csv")
    stats = np.load(data / "normalization_stats.npz", allow_pickle=False)
    row_i = int(np.flatnonzero(manifest.label.to_numpy(dtype=int) == 1)[0])
    meta = index.iloc[row_i]
    env_root = __import__("os").environ.get("DRONE_AUDIO_DATASET_ROOT")
    dataset_parent = Path(env_root) if env_root else root / "archive/CNN_V2_restore_validation_20261126/DroneAudioDataset_binary_frozen_20261126/dataset"
    audio_root = dataset_parent / "Binary_Drone_Audio"
    clip_id = str(manifest.iloc[row_i].clip_id)
    waveform = fit_length(load_wave(audio_root / Path(*clip_id.split("/")), config), config)
    expected_window, offset = frame_aligned_crop(waveform, config.win_samples, config.hop, np.random.default_rng(1), training=False)
    # The validation crop is centered over legal waveform offsets. V2 centers
    # over mel frames; map the saved cache start to the corresponding sample crop.
    saved_offset = int(meta.center_start_frame)
    expected_window = waveform[saved_offset * config.hop : saved_offset * config.hop + config.win_samples]
    raw = transform(mel_power(expected_window, config), config, {"mean": stats["mean"], "std": stats["std"]})
    cache = np.load(data / "cache/mel_power_val.npy", mmap_mode="r", allow_pickle=False)
    cached_power = cache[:, int(meta.start_frame) : int(meta.start_frame) + int(meta.n_frames)]
    cached = cache_tensor_for_offset(cached_power, saved_offset, config, stats["mean"], stats["std"])
    assert raw.shape == cached.shape == (1, 64, 28)
    tensor_error = float(np.max(np.abs(raw - cached)))
    assert tensor_error <= 1e-4
    model = torch.jit.load(str(root / "results/cnn_v2/export/CNN_V2.0/cnn_v2_torchscript.pt"), map_location="cpu")
    model.eval()
    raw_batch = torch.from_numpy(raw[None, ...])
    cache_batch = torch.from_numpy(cached[None, ...])
    with torch.inference_mode():
        raw_logit = model(raw_batch).cpu().numpy()
        cache_logit = model(cache_batch).cpu().numpy()
    logit_error = float(np.max(np.abs(raw_logit - cache_logit)))
    threshold = 2.823960542678833
    assert logit_error <= 1e-4
    np.testing.assert_array_equal(raw_logit >= threshold, cache_logit >= threshold)
