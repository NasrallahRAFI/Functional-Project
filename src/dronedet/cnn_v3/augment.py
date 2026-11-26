"""Deterministic waveform channel transforms for the registered CNN V3 study."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math

import numpy as np
from scipy import signal


SR = 16_000
EPS = 1e-12


def rms_db(x: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).all():
        raise ValueError("RMS requires a non-empty finite waveform")
    return float(20.0 * np.log10(np.sqrt(np.mean(a * a)) + EPS))


def _db_rms_scale(x: np.ndarray, db: float) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) * (10.0 ** (db / 20.0))


def _safe_reflect_pad(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return x
    if x.size < 2:
        return np.pad(x, (n, n), mode="edge")
    return np.pad(x, (n, n), mode="reflect")


def band_limit(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    hp_hz: tuple[float, float] = (150.0, 600.0),
    lp_hz: tuple[float, float] = (3500.0, 7500.0),
    order: int = 2,
    sample_rate: int = SR,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Butterworth speaker band limit with reflected context to reduce edge transients."""
    if sample_rate <= 0 or order < 1:
        raise ValueError("sample rate and filter order must be positive")
    lo = float(np.exp(rng.uniform(np.log(hp_hz[0]), np.log(hp_hz[1]))))
    hi = float(np.exp(rng.uniform(np.log(lp_hz[0]), np.log(lp_hz[1]))))
    if not (0 < lo < hi < sample_rate / 2):
        raise ValueError("sampled band limits are invalid for this sample rate")
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=sample_rate, output="sos")
    pad = min(max(8 * order, sample_rate // 10), max(0, x.size - 1))
    padded = _safe_reflect_pad(np.asarray(x, dtype=np.float64), pad)
    filtered = signal.sosfilt(sos, padded)
    if pad:
        filtered = filtered[pad:-pad]
    return filtered, {"hp_hz": lo, "lp_hz": hi, "order": order, "pad_samples": pad}


def smooth_eq(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    control_points: int = 6,
    max_gain_db: float = 6.0,
    sample_rate: int = SR,
    taps: int = 513,
) -> tuple[np.ndarray, dict[str, list[float] | int]]:
    """Apply a smooth, symmetric FIR EQ with explicit reflected edge context."""
    if control_points < 2 or taps < 31 or taps % 2 == 0 or max_gain_db < 0:
        raise ValueError("EQ needs >=2 points, an odd FIR length >=31 and nonnegative gain")
    nyq = sample_rate / 2
    frequencies = np.geomspace(40.0, min(0.98 * nyq, 7800.0), control_points)
    gains = rng.uniform(-max_gain_db, max_gain_db, control_points)
    frequencies = np.concatenate(([0.0], frequencies, [nyq]))
    gains = np.concatenate(([gains[0]], gains, [gains[-1]]))
    linear_gain = 10.0 ** (gains / 20.0)
    kernel = signal.firwin2(taps, frequencies, linear_gain, fs=sample_rate, window="hann")
    pad = taps // 2
    padded = _safe_reflect_pad(np.asarray(x, dtype=np.float64), pad)
    filtered = signal.fftconvolve(padded, kernel, mode="same")[pad:-pad]
    return filtered, {"frequency_hz": frequencies.tolist(), "gain_db": gains.tolist(), "taps": taps}


def soft_clip(x: np.ndarray, rng: np.random.Generator, drive_db: tuple[float, float] = (0.0, 18.0)) -> tuple[np.ndarray, dict[str, float]]:
    a = np.asarray(x, dtype=np.float64)
    drive = float(rng.uniform(*drive_db))
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    if peak <= EPS:
        return a.copy(), {"drive_db": drive, "peak_before": peak}
    y = np.tanh((10 ** (drive / 20.0)) * a / peak)
    before = np.sqrt(np.mean(a * a))
    after = np.sqrt(np.mean(y * y))
    if after > EPS:
        y *= before / after
    return y, {"drive_db": drive, "peak_before": peak}


def room_response(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    rt60_s: tuple[float, float] = (0.2, 1.0),
    drr_db: tuple[float, float] = (-5.0, 10.0),
    sample_rate: int = SR,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Convolve with a direct impulse plus a tail whose energy meets sampled DRR."""
    rt = float(rng.uniform(*rt60_s))
    drr = float(rng.uniform(*drr_db))
    n_tail = max(2, int(round(rt * sample_rate)))
    t = np.arange(n_tail, dtype=np.float64) / sample_rate
    tail = rng.standard_normal(n_tail) * np.exp(-6.907755278982137 * t / rt)
    # The direct impulse has unit energy; scale the entire tail to the requested
    # direct-to-reverberant energy ratio without overwriting its first tap.
    tail_energy = float(np.dot(tail, tail))
    if tail_energy > EPS:
        tail *= math.sqrt(10.0 ** (-drr / 10.0) / tail_energy)
    h = np.concatenate(([1.0], tail))
    pad = min(n_tail, max(0, x.size - 1))
    padded = _safe_reflect_pad(np.asarray(x, dtype=np.float64), pad)
    wet = signal.fftconvolve(padded, h, mode="full")
    start = pad
    y = wet[start : start + x.size]
    if y.size < x.size:
        y = np.pad(y, (0, x.size - y.size))
    return y, {"rt60_s": rt, "drr_db": drr, "tail_samples": n_tail, "pad_samples": pad}


def colored_noise(n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Generate finite unit-RMS 1/f**beta noise without a DC singularity."""
    if n <= 0 or beta < 0 or beta > 2:
        raise ValueError("noise length must be positive and beta must be in [0,2]")
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freq = np.fft.rfftfreq(n, d=1.0 / SR)
    freq[0] = freq[1] if freq.size > 1 else 1.0
    spectrum /= freq ** (beta / 2.0)
    noise = np.fft.irfft(spectrum, n=n)
    noise -= np.mean(noise)
    noise /= np.sqrt(np.mean(noise * noise)) + EPS
    return noise


def _fit_noise(noise: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    a = np.asarray(noise, dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or not np.isfinite(a).all():
        raise ValueError("noise donor must be a non-empty finite mono waveform")
    if a.size >= n:
        start = int(rng.integers(0, a.size - n + 1))
        out = a[start : start + n].copy()
    else:
        start = int(rng.integers(0, a.size))
        first = a[start:].copy()
        out = first[:n]
        # Append loop periods with an overlap crossfade; this avoids a hard
        # discontinuity without circularly shifting unrelated donor samples.
        fade = min(SR // 100, a.size // 4, n // 4)
        while out.size < n:
            remaining = n - out.size
            if fade > 0 and out.size >= fade:
                count = min(fade, remaining)
                ramp = np.linspace(0.0, 1.0, count, endpoint=True)
                joined = out[-count:] * (1.0-ramp) + a[:count] * ramp
                out = np.concatenate((out[:-count], joined, a[count:]))
            else:
                count = min(a.size, remaining)
                out = np.concatenate((out, a[:count]))
        out = out[:n].copy()
    out -= np.mean(out)
    out /= np.sqrt(np.mean(out * out)) + EPS
    return out


@dataclass
class NoiseDonor:
    clip_id: str
    group_id: str
    samples: np.ndarray
    sha256: str


@dataclass
class NoiseBank:
    """Training-negative-only donor pool; donor admission is explicit and auditable."""
    donors: list[NoiseDonor] = field(default_factory=list)

    def add(self, donor: NoiseDonor) -> None:
        x = np.asarray(donor.samples, dtype=np.float32)
        if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
            raise ValueError("noise donor must be non-empty finite mono")
        if np.all(x == 0) or rms_db(x) < -70.0:
            raise ValueError("silent and below -70 dBFS noise donors are excluded")
        if not donor.clip_id or not donor.group_id or not donor.sha256:
            raise ValueError("noise donor requires clip, group and SHA-256 provenance")
        self.donors.append(NoiseDonor(donor.clip_id, donor.group_id, x.copy(), donor.sha256))

    def draw(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, str]]:
        if self.donors and rng.random() < 0.5:
            donor = self.donors[int(rng.integers(0, len(self.donors)))]
            return _fit_noise(donor.samples, n, rng), {
                "kind": "training_negative", "clip_id": donor.clip_id,
                "group_id": donor.group_id, "sha256": donor.sha256,
            }
        beta = float(rng.uniform(0.0, 2.0))
        return colored_noise(n, beta, rng), {"kind": "procedural", "beta": str(beta)}


@dataclass(frozen=True)
class AugmentConfig:
    sample_rate: int = SR
    p_band_limit: float = 0.7
    p_eq: float = 0.5
    p_soft_clip: float = 0.3
    gain_db: tuple[float, float] = (-30.0, 6.0)
    hp_hz: tuple[float, float] = (150.0, 600.0)
    lp_hz: tuple[float, float] = (3500.0, 7500.0)
    eq_max_gain_db: float = 6.0
    p_reverb: float = 0.5
    rt60_s: tuple[float, float] = (0.2, 1.0)
    drr_db: tuple[float, float] = (-5.0, 10.0)
    p_noise: float = 0.6
    p_floor: float = 1.0
    p_dc: float = 0.0  # Late A5a ablation; disabled for the core A1-A4 chain.
    dc_amplitude: tuple[float, float] = (-0.01, 0.01)
    p_hard_clip: float = 0.0  # Late A5a ablation; disabled for the core chain.
    hard_clip_fraction: tuple[float, float] = (0.3, 0.9)
    noise_snr_db: tuple[float, float] = (0.0, 30.0)
    floor_dbfs: tuple[float, float] = (-75.0, -45.0)
    max_added_noise_snr_db: float = 0.0
    minimum_positive_snr_db: float = 0.0
    p_clean_recipe: float = 0.2
    clean_gain_db: tuple[float, float] = (-12.0, 12.0)
    p_specaugment: float = 0.0

    def __post_init__(self) -> None:
        if self.sample_rate < 8000:
            raise ValueError("sample rate is implausibly low")
        for name in ("p_band_limit", "p_eq", "p_soft_clip", "p_reverb", "p_noise", "p_floor", "p_dc", "p_hard_clip", "p_clean_recipe", "p_specaugment"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if self.gain_db[0] > self.gain_db[1] or self.floor_dbfs[0] > self.floor_dbfs[1]:
            raise ValueError("augmentation ranges must be ordered")

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ChannelAugment:
    """One seeded, class-independent V3 view; returns audio and sampled provenance."""

    def __init__(self, noise_bank: NoiseBank | None = None, config: AugmentConfig | None = None):
        self.noise_bank = noise_bank or NoiseBank()
        self.config = config or AugmentConfig()

    def __call__(self, waveform: np.ndarray, rng: np.random.Generator, *, label: int | None = None) -> tuple[np.ndarray, dict]:
        del label  # Deliberately class-independent; labels never select a transform.
        x = np.asarray(waveform, dtype=np.float64)
        if x.ndim != 1 or x.size < 1024 or not np.isfinite(x).all():
            raise ValueError("augmentation expects at least 1024 finite mono samples")
        params: dict = {"config_sha256": self.config.sha256, "branch": "channel"}
        c = self.config
        clean_recipe = rng.random() < c.p_clean_recipe
        if clean_recipe:
            gain = float(rng.uniform(*c.clean_gain_db))
            x = _db_rms_scale(x, gain)
            params.update(branch="v2_recipe", gain_db=gain)
        else:
            if rng.random() < c.p_band_limit:
                x, p = band_limit(x, rng, hp_hz=c.hp_hz, lp_hz=c.lp_hz, sample_rate=c.sample_rate); params["band_limit"] = p
            if rng.random() < c.p_eq:
                x, p = smooth_eq(x, rng, max_gain_db=c.eq_max_gain_db, sample_rate=c.sample_rate); params["eq"] = p
            if rng.random() < c.p_soft_clip:
                x, p = soft_clip(x, rng); params["soft_clip"] = p
            gain = float(rng.uniform(*c.gain_db))
            x = _db_rms_scale(x, gain)
            params["gain_db"] = gain
            if rng.random() < c.p_reverb:
                x, p = room_response(x, rng, rt60_s=c.rt60_s, drr_db=c.drr_db, sample_rate=c.sample_rate); params["room"] = p

        if not np.isfinite(x).all() or x.size != waveform.size:
            raise RuntimeError("channel stages changed length or produced non-finite audio")

        if clean_recipe:
            # Preserve the V2 branch exactly: gain only, with no microphone
            # floor, background, limiter, or other channel transform.
            return np.asarray(x, dtype=np.float32), params

        if rng.random() < c.p_floor:
            floor_db = float(rng.uniform(*c.floor_dbfs))
            floor = _db_rms_scale(colored_noise(x.size, float(rng.uniform(0.0, 1.5)), rng), floor_db)
        else:
            floor_db = None
            floor = np.zeros_like(x)
        noise, donor = self.noise_bank.draw(x.size, rng) if rng.random() < c.p_noise else (np.zeros_like(x), {"kind": "none"})
        target_snr = float(rng.uniform(*c.noise_snr_db)) if donor["kind"] != "none" else math.inf
        # Scale the combined nuisance (floor + background) to the registered
        # added-noise SNR. This prevents a second floor from silently pushing
        # the mixture below its declared minimum.
        nuisance = floor.copy()
        if donor["kind"] != "none":
            bg = _db_rms_scale(noise, rms_db(x) - target_snr)
            nuisance += bg
        source_level_db = rms_db(x)
        if source_level_db - rms_db(nuisance) < c.minimum_positive_snr_db and np.any(x != 0.0):
            scale = 10.0 ** ((source_level_db - c.minimum_positive_snr_db - rms_db(nuisance)) / 20.0)
            nuisance *= scale
            params["nuisance_limited"] = True
        else:
            params["nuisance_limited"] = False
        total_snr = rms_db(x) - rms_db(nuisance) if np.any(nuisance) else math.inf
        x += nuisance
        params.update({"floor_dbfs": floor_db, "noise_source": donor, "requested_background_snr_db": target_snr, "measured_added_noise_snr_db": total_snr})

        if rng.random() < c.p_dc:
            dc = float(rng.uniform(*c.dc_amplitude)); x += dc; params["dc_offset"] = dc
        if rng.random() < c.p_hard_clip:
            threshold = float(rng.uniform(*c.hard_clip_fraction)) * (float(np.max(np.abs(x))) + EPS)
            params["hard_clip_threshold"] = threshold
            params["hard_clip_fraction"] = float(np.mean(np.abs(x) > threshold))
            x = np.clip(x, -threshold, threshold)
        # Preserve absolute gain in float audio. PCM export for listening uses
        # one shared montage attenuation; per-clip peak normalization would
        # erase the sampled device-level difference.
        peak = float(np.max(np.abs(x)))
        params["final_limiter_used"] = False
        params["peak_before_limiter"] = peak
        params["peak_after_limiter"] = peak
        out = np.asarray(x, dtype=np.float32)
        if out.shape != waveform.shape or not np.isfinite(out).all():
            raise RuntimeError("augmentation violated finite/length contract")
        return out, params


def stable_rng(base_seed: int, epoch: int, clip_id: str, draw_id: int = 0) -> np.random.Generator:
    """Derive an order-independent generator, with occurrence ID for repeats."""
    if epoch < 0 or draw_id < 0:
        raise ValueError("epoch and draw IDs must be nonnegative")
    payload = f"{int(base_seed)}\0{int(epoch)}\0{clip_id}\0{int(draw_id)}".encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)
