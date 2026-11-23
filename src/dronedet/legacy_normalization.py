"""Training-set-only per-bin normalization and causal PCEN."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import warnings

warnings.warn("dronedet.legacy_normalization is deprecated; use dronedet.features", DeprecationWarning, stacklevel=2)


def apply_power_gain(mel_power: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply gain in the cached mel-power domain."""
    x = np.asarray(mel_power, dtype=np.float32)
    if not np.all(np.isfinite(x)) or np.any(x < 0) or not np.isfinite(gain_db):
        raise ValueError("mel power and gain must be finite; mel power must be nonnegative")
    return (x * np.float32(10.0 ** (gain_db / 10.0))).astype(np.float32)


def sample_gain(rng: np.random.Generator, low_db: float = -12.0, high_db: float = 12.0) -> float:
    if not low_db < high_db:
        raise ValueError("gain interval must be increasing")
    return float(rng.uniform(low_db, high_db))


def fit_scalar_normalization(windows: Iterable[np.ndarray], *, per_window_center: bool = False) -> dict[str, float]:
    """Fit scalar train statistics for per-window dB or PCEN normalization."""
    count = 0
    mean = 0.0
    m2 = 0.0
    for window in windows:
        x = np.asarray(window, dtype=np.float64)
        if x.ndim != 2 or x.size == 0 or not np.all(np.isfinite(x)):
            raise ValueError("each training window must be a finite nonempty 2D array")
        if per_window_center:
            x = x - x.mean()
        flat = x.reshape(-1)
        n = flat.size
        batch_mean = float(flat.mean())
        batch_m2 = float(np.square(flat - batch_mean).sum())
        total = count + n
        delta = batch_mean - mean
        m2 += batch_m2 + delta * delta * count * n / total
        mean += delta * n / total
        count = total
    if not count:
        raise ValueError("training windows are required")
    std = max(float(np.sqrt(m2 / count)), 1e-6)
    return {"mean": mean, "std": std, "count": count}


class LogMelNormalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray, db_floor: float = -100.0) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.ndim != 1 or self.std.shape != self.mean.shape:
            raise ValueError("mean and std must be one-dimensional with matching shapes")
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.std)) or np.any(self.std <= 0):
            raise ValueError("normalization statistics must be finite and std positive")
        self.db_floor = float(db_floor)

    @classmethod
    def fit(cls, training_spectrograms: Iterable[np.ndarray], db_floor: float = -120.0) -> "LogMelNormalizer":
        raise RuntimeError("use dronedet.features.Normalizer.fit(manifest_rows)")
        m2: np.ndarray | None = None
        count = 0
        silent_count = 0
        for item in training_spectrograms:
            silent = False
            if isinstance(item, tuple) and len(item) == 2:
                item, silent = item
            if silent:
                silent_count += 1
                continue
            values = np.asarray(item, dtype=np.float32)
            if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
                raise ValueError("each input must have shape (n_mels, nonzero_frames)")
            if mean is not None and values.shape[0] != mean.size:
                raise ValueError("all inputs must use the same number of mel bins")
            clipped = np.maximum(values, db_floor).astype(np.float64)
            batch_count = clipped.shape[1]
            batch_mean = clipped.mean(axis=1)
            batch_m2 = np.sum((clipped - batch_mean[:, None]) ** 2, axis=1)
            if mean is None:
                mean, m2, count = batch_mean, batch_m2, batch_count
                continue
            combined = count + batch_count
            delta = batch_mean - mean
            m2 += batch_m2 + delta**2 * (count * batch_count / combined)
            mean += delta * (batch_count / combined)
            count = combined
        if mean is None or m2 is None or count == 0:
            raise ValueError("training spectrograms are required")
        std = np.sqrt(np.maximum(m2 / count, 0.0))
        result = cls(mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32), db_floor)
        result.excluded_silent_count = silent_count
        return result

    def transform(self, spectrogram: np.ndarray) -> np.ndarray:
        x = np.asarray(spectrogram, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] != self.mean.size:
            raise ValueError("spectrogram must have shape (n_mels, frames)")
        return (np.maximum(x, self.db_floor) - self.mean[:, None]) / self.std[:, None]

    def state_dict(self) -> dict[str, object]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "db_floor": self.db_floor,
                "excluded_silent_count": int(getattr(self, "excluded_silent_count", 0))}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "LogMelNormalizer":
        return cls(np.asarray(state["mean"], dtype=np.float32), np.asarray(state["std"], dtype=np.float32), float(state["db_floor"]))


def prepare_model_input(
    logmel_db: np.ndarray,
    normalizer: LogMelNormalizer,
    *,
    n_frames: int = 128,
    rng: np.random.Generator | None = None,
    random_crop: bool = False,
    frequency_coordinate: bool = True,
) -> np.ndarray:
    """Normalize, fixed-length crop/pad, and add the optional frequency-index channel.

    Returns one sample with shape (channels, mel_bins, frames). Short clips must
    be padded at waveform level before feature extraction; dB-domain padding is rejected.
    """
    db = np.asarray(logmel_db, dtype=np.float32)
    if db.ndim != 2 or db.shape[0] != normalizer.mean.size or db.shape[1] == 0 or n_frames <= 0:
        raise ValueError("logmel_db must have shape (matching mel bins, nonzero frames)")
    if not np.all(np.isfinite(db)):
        raise ValueError("log-mel spectrogram contains non-finite values")
    if db.shape[1] > n_frames:
        if random_crop:
            if rng is None:
                raise ValueError("random cropping requires an explicit random generator")
            start = int(rng.integers(0, db.shape[1] - n_frames + 1))
        else:
            start = (db.shape[1] - n_frames) // 2
        db = db[:, start:start + n_frames]
    elif db.shape[1] < n_frames:
        raise ValueError("short clips must be wrap-padded at waveform level before feature extraction")
    normalized = normalizer.transform(db)
    if not frequency_coordinate:
        return normalized[None, ...].astype(np.float32)
    coordinate = np.broadcast_to(
        np.linspace(0.0, 1.0, db.shape[0], dtype=np.float32)[:, None], normalized.shape
    )
    return np.stack((normalized, coordinate), axis=0).astype(np.float32)


def spec_augment(
    features: np.ndarray,
    rng: np.random.Generator,
    *,
    frequency_masks: int = 2,
    max_frequency_width: int = 6,
    time_masks: int = 2,
    max_time_width: int = 16,
) -> np.ndarray:
    """Apply light zero-mask SpecAugment to feature data, leaving coordinate channels intact."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim not in {2, 3}:
        raise ValueError("features must have shape (mel, frames) or (channels, mel, frames)")
    if not np.all(np.isfinite(x)):
        raise ValueError("features contain NaN or infinite values")
    if min(frequency_masks, max_frequency_width, time_masks, max_time_width) < 0:
        raise ValueError("mask counts and widths must be nonnegative")
    result = x.copy()
    spectrogram = result if result.ndim == 2 else result[0]
    n_mels, n_frames = spectrogram.shape
    for _ in range(frequency_masks):
        width = int(rng.integers(0, min(max_frequency_width, n_mels) + 1))
        if width:
            start = int(rng.integers(0, n_mels - width + 1))
            spectrogram[start:start + width, :] = 0.0
    for _ in range(time_masks):
        width = int(rng.integers(0, min(max_time_width, n_frames) + 1))
        if width:
            start = int(rng.integers(0, n_frames - width + 1))
            spectrogram[:, start:start + width] = 0.0
    return result
