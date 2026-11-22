"""soxr HQ resampling helpers used by offline and streaming paths."""
from __future__ import annotations

import numpy as np
import soxr


def resample(audio: np.ndarray, source_rate: int, target_rate: int, *, quality: str = "HQ") -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1 or source_rate <= 0 or target_rate <= 0 or not np.all(np.isfinite(x)):
        raise ValueError("resample expects finite mono audio and positive rates")
    if source_rate == target_rate:
        return x.copy()
    return np.asarray(soxr.resample(x, source_rate, target_rate, quality=quality), dtype=np.float32)


def resample_stream(source_rate: int, target_rate: int, *, quality: str = "HQ"):
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    return soxr.ResampleStream(source_rate, target_rate, 1, dtype="float32", quality=quality)


def measure_swept_passband(sample_rate_in: int, sample_rate_out: int = 16000, *,
                           quality: str = "HQ", duration_s: float = 6.0,
                           points: int = 12000) -> tuple[float, np.ndarray, np.ndarray]:
    """Measure the first -1 dB edge using a log chirp and instantaneous frequency."""
    from scipy.signal import chirp, hilbert
    if sample_rate_in <= sample_rate_out or duration_s <= 1 or points < 100:
        raise ValueError("measurement needs downsampling, duration > 1s and sufficient points")
    count = round(duration_s * sample_rate_in)
    t = np.arange(count, dtype=np.float64) / sample_rate_in
    f0, f1 = 80.0, min(sample_rate_out * .499, sample_rate_in * .45)
    sweep = chirp(t, f0=f0, f1=f1, t1=duration_s, method="logarithmic").astype(np.float32)
    out = resample(sweep, sample_rate_in, sample_rate_out, quality=quality)
    analytic = hilbert(out.astype(np.float64))
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.gradient(phase) * sample_rate_out / (2 * np.pi)
    amplitude = np.abs(analytic)
    valid = (inst_freq > 100) & (inst_freq < sample_rate_out * .49)
    bins = np.geomspace(200, sample_rate_out * .49, points + 1)
    centers = np.sqrt(bins[:-1] * bins[1:])
    level = np.full(points, np.nan)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = valid & (inst_freq >= lo) & (inst_freq < hi)
        if np.any(mask):
            level[i] = 20 * np.log10(np.sqrt(np.mean(amplitude[mask] ** 2)) + 1e-15)
    ref = np.nanmedian(level[(centers >= 900) & (centers <= 1100)])
    relative = level - ref
    hits = np.flatnonzero((centers > 1000) & (relative <= -1.0))
    edge = float(centers[hits[0]]) if hits.size else sample_rate_out / 2
    return edge, centers, relative


def fmax_below_passband(edge_hz: float) -> float:
    if not np.isfinite(edge_hz) or edge_hz <= 0:
        raise ValueError("passband edge must be positive and finite")
    return float(min(8000, np.floor((edge_hz - 1e-9) / 250) * 250))


def measure_minus_1db_point(*, sample_rate_in: int = 48_000, sample_rate_out: int = 16_000,
                            quality: str = "HQ", duration_s: float = 6.0,
                            step_hz: float = 25.0) -> float:
    """Compatibility name; the implementation uses the V8 log-swept measurement."""
    edge, _, _ = measure_swept_passband(sample_rate_in, sample_rate_out,
                                         quality=quality, duration_s=duration_s)
    return edge
