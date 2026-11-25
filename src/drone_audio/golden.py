"""Reproducible golden vectors and front-end parity checks."""

from __future__ import annotations

import warnings

warnings.warn(
    "drone_audio.golden preserves historical reference vectors; use "
    "dronedet.bundle.verify_bundle for current production bundle goldens",
    DeprecationWarning,
    stacklevel=2,
)

from pathlib import Path

import numpy as np
import soundfile as sf

from .dsp import mel_filterbank
from .frontend import FrontEnd, FrontEndConfig
from .resample import resample


def synthetic_golden_audio(
    sample_rate: int, *, duration_s: float = 2.2, seed: int = 2025
) -> np.ndarray:
    if sample_rate not in {16_000, 48_000}:
        raise ValueError("golden sample rates are 16 kHz and 48 kHz")
    count = round(sample_rate * duration_s)
    time = np.arange(count, dtype=np.float64) / sample_rate
    chirp = 0.08 * np.sin(2 * np.pi * (180 * time + 0.5 * 900 * time**2 / duration_s))
    noise = np.random.default_rng(seed).standard_normal(count) * 0.005
    return (chirp + noise).astype(np.float32)


def create_golden(
    directory: str | Path, config: FrontEndConfig | None = None
) -> list[Path]:
    """Save deterministic 16/48 kHz WAVs and their reference log-mel arrays."""
    cfg = config or FrontEndConfig()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    frontend = FrontEnd(cfg)
    paths: list[Path] = []
    for rate in (16_000, 48_000):
        audio = synthetic_golden_audio(rate)
        wav_path = root / f"synthetic_{rate}.wav"
        npy_path = root / f"synthetic_{rate}_logmel.npy"
        if wav_path.exists() or npy_path.exists():
            raise FileExistsError(f"golden files already exist for {rate} Hz")
        sf.write(wav_path, audio, rate, subtype="FLOAT")
        standardized = (
            audio
            if rate == cfg.sample_rate
            else resample(audio, rate, cfg.sample_rate, quality=cfg.resample_quality)
        )
        np.save(npy_path, frontend.offline(standardized), allow_pickle=False)
        paths.extend((wav_path, npy_path))
    return paths


def verify_golden(
    directory: str | Path,
    config: FrontEndConfig | None = None,
    *,
    tolerance_db: float = 1e-3,
) -> dict[str, float]:
    cfg = config or FrontEndConfig()
    frontend = FrontEnd(cfg)
    errors: dict[str, float] = {}
    for rate in (16_000, 48_000):
        audio, source_rate = sf.read(
            Path(directory) / f"synthetic_{rate}.wav", dtype="float32"
        )
        if source_rate != rate or audio.ndim != 1:
            raise AssertionError(f"golden WAV metadata mismatch at {rate} Hz")
        standardized = (
            audio
            if rate == cfg.sample_rate
            else resample(audio, rate, cfg.sample_rate, quality=cfg.resample_quality)
        )
        expected = np.load(
            Path(directory) / f"synthetic_{rate}_logmel.npy", allow_pickle=False
        )
        actual = frontend.offline(standardized)
        error = float(np.max(np.abs(actual - expected)))
        errors[str(rate)] = error
        if error > tolerance_db:
            raise AssertionError(
                f"{rate} Hz golden drift {error:.6g} dB exceeds {tolerance_db}"
            )
    return errors


def verify_frame_crop(
    frontend: FrontEnd, audio: np.ndarray, *, start_frame: int, n_frames: int
) -> float:
    if start_frame < 0 or n_frames <= 0:
        raise ValueError("frame crop must be nonnegative and nonempty")
    frame_audio = np.asarray(audio, dtype=np.float32)[
        start_frame * frontend.config.hop :
    ]
    frame_audio = frame_audio[
        : frontend.config.n_fft + (n_frames - 1) * frontend.config.hop
    ]
    full_crop = frontend.offline(audio)[:, start_frame : start_frame + n_frames]
    waveform_crop = frontend.offline(frame_audio)
    if full_crop.shape != waveform_crop.shape:
        raise AssertionError("frame and waveform crops have different shapes")
    error = float(np.max(np.abs(full_crop - waveform_crop)))
    if error > 1e-3:
        raise AssertionError(f"frame crop differs from waveform crop by {error:.6g} dB")
    return error


def verify_report_filterbank(*, tolerance: float = 1e-6) -> dict[str, float]:
    """Compare librosa HTK/Slaney filters with the preserved report implementation."""
    reference_cfg = FrontEndConfig.reference()
    actual = FrontEnd(reference_cfg).mel
    report = mel_filterbank(
        reference_cfg.sample_rate,
        reference_cfg.n_fft,
        reference_cfg.n_mels,
        reference_cfg.fmin,
        reference_cfg.fmax,
    )
    support_equal = bool(np.array_equal(actual != 0, report != 0))
    nonzero = actual != 0
    relative = float(
        np.max(
            np.abs(actual[nonzero] - report[nonzero])
            / np.maximum(np.abs(report[nonzero]), 1e-30)
        )
    )
    if not support_equal or relative > tolerance:
        raise AssertionError(
            f"filterbank mismatch (support={support_equal}, relative={relative:.6g})"
        )
    return {"support_equal": float(support_equal), "max_relative_error": relative}
