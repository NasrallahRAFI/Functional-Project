"""Enforced train-only statistics and librosa PCEN implementation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from scipy import signal

from .config import FrontEndConfig
from .features import to_db


def pcen_transform(power: np.ndarray, config: FrontEndConfig) -> np.ndarray:
    x = np.asarray(power, dtype=np.float32)
    if x.ndim != 2 or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("PCEN expects finite nonnegative (mel, frames) power")
    magnitude = np.sqrt(x) * np.float32(config.pcen_input_scale)
    b = float(1.0 - np.exp(-config.hop / (config.sr * config.pcen_time_constant)))
    # librosa's lfilter state has the input dimensions with a singleton along time.
    zi = ((1.0 - b) * magnitude.mean(axis=-1, keepdims=True)).astype(np.float32)
    result = librosa.pcen(magnitude, sr=config.sr, hop_length=config.hop,
        time_constant=config.pcen_time_constant, gain=config.pcen_gain,
        bias=config.pcen_bias, power=config.pcen_power, eps=config.pcen_eps,
        zi=zi, axis=-1)
    return np.asarray(result, dtype=np.float32)


class Normalizer:
    """Fit dB or PCEN statistics from explicit train-manifest rows only."""
    def __init__(self, mode: str, mean: np.ndarray | float, std: np.ndarray | float,
                 excluded_count: int = 0) -> None:
        self.mode = mode
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.excluded_count = int(excluded_count)
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.std)) or np.any(self.std <= 0):
            raise ValueError("normalization stats must be finite and std positive")

    @classmethod
    def fit(cls, manifest_rows, config: FrontEndConfig, power_by_file: dict[str, np.ndarray]) -> "Normalizer":
        rows = list(manifest_rows)
        if any(str(row["split"]).lower() != "train" for row in rows):
            raise ValueError("Normalizer.fit accepts train rows only")
        if config.mode == "logstft":
            raise NotImplementedError("logstft requires a separate 513-bin cache")
        values, excluded = [], 0
        from .features import stats_exclude
        for row in rows:
            silent = bool(row.get("silent", row.get("all_zero", False)))
            near_silent = bool(row.get("near_silent", False))
            crop_all_zero = bool(row.get("crop_all_zero", False))
            if stats_exclude(silent, near_silent, crop_all_zero):
                excluded += 1
                continue
            name = str(row.get("file", row.get("clip_id", "")))
            if not name: raise ValueError("manifest rows require file or clip_id")
            p = np.asarray(power_by_file[name], dtype=np.float32)
            # Full-clip dB moments use every frame; PCEN/per-window modes use the demo-sized crop.
            if config.mode in {"pcen", "per_window_mean"}:
                from .features import centre_window
                p = centre_window(p, config)
            if config.mode == "pcen":
                v = pcen_transform(p, config)
            else:
                v = to_db(p, config)
            if config.mode == "per_window_mean":
                v = v - v.mean()
            values.append(v.astype(np.float64))
        if not values: raise ValueError("no eligible training clips for normalization")
        if config.mode == "global_stats":
            joined = np.concatenate(values, axis=1)
            # Floor-clamped dB cells carry the mel-normalization-dependent
            # amin offset into otherwise per-band-invariant statistics. Keep
            # the train-clip eligibility rule, and additionally omit cells at
            # the configured power_to_db floor from each band's moments.
            floor_db = 10.0 * float(np.log10(config.amin))
            floor_epsilon_db = 1e-6
            mean = np.empty(config.n_mels, dtype=np.float64)
            std = np.empty(config.n_mels, dtype=np.float64)
            for band in range(config.n_mels):
                cells = joined[band, joined[band] > floor_db + floor_epsilon_db]
                if cells.size == 0:
                    raise ValueError(f"mel band {band} has no non-floor training cells")
                mean[band] = cells.mean(dtype=np.float64)
                std[band] = cells.std(dtype=np.float64)
        else:
            joined = np.concatenate([v.ravel() for v in values])
            mean, std = float(joined.mean()), float(joined.std())
        return cls(config.mode, mean, np.maximum(std, 1e-6), excluded)

    def state_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "excluded_silent_near_silent": self.excluded_count}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state_dict(), indent=2) + "\n", encoding="utf8")

    def save_float32(self, path: str | Path) -> None:
        """Persist typed float32 statistics for cache/bundle consumers."""
        np.savez_compressed(path, mode=np.asarray(self.mode), mean=self.mean.astype(np.float32),
                            std=self.std.astype(np.float32),
                            excluded_silent_near_silent=np.asarray(self.excluded_count, dtype=np.int64))


def apply_gain(power: np.ndarray, rng: np.random.Generator, config: FrontEndConfig, *, training: bool) -> np.ndarray:
    if not training:
        return np.asarray(power, dtype=np.float32)
    lo, hi = config.gain_range_db
    gain = float(rng.uniform(lo, hi))
    return (np.asarray(power, dtype=np.float32) * np.float32(10 ** (gain / 10))).astype(np.float32)


def apply_fixed_gain(power: np.ndarray, gain_db: float) -> np.ndarray:
    return (np.asarray(power, dtype=np.float32) * np.float32(10 ** (gain_db / 10))).astype(np.float32)


def pcen_gain_check(train_windows: list[np.ndarray], config: FrontEndConfig) -> dict[str, float | str]:
    """Measure gain robustness on provided train windows; no train data means deferred."""
    if not train_windows: return {"status": "deferred_no_frozen_train_manifest"}
    changes=[]
    for p in train_windows:
        base=pcen_transform(p,config)
        std=max(float(base.std()),1e-12)
        for gain in (-12.0,12.0):
            changed=pcen_transform(apply_fixed_gain(p,gain),config)
            changes.append(float(np.mean(np.abs(changed-base))/std))
    largest=max(changes)
    return {"status": "pass" if largest <= .2 else "fail", "max_change_in_std": largest,
            "acceptance_limit": .2, "window_count": len(train_windows)}


def stress_filter(audio: np.ndarray, sample_rate: int, kind: str) -> np.ndarray:
    specs = {"highpass_150": ("highpass", 150), "highpass_300": ("highpass", 300), "lowpass_5000": ("lowpass", 5000)}
    if kind not in specs: raise ValueError("unknown stress filter")
    typ, cutoff = specs[kind]
    sos = signal.butter(2, cutoff, btype=typ, fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(audio, dtype=np.float32)).astype(np.float32)
