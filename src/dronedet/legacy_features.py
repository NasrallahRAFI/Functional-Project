"""Window selection, cached mel power and fixed-length tabular features."""

from __future__ import annotations

import csv
from pathlib import Path
from collections.abc import Iterable

import librosa
import numpy as np
import warnings

warnings.warn("dronedet.legacy_features is deprecated; use dronedet.features", DeprecationWarning, stacklevel=2)

from .legacy_frontend import FrontEnd, FrontEndConfig
from .resample import resample


def centred_window(power: np.ndarray, n_win: int) -> np.ndarray:
    """Take the fixed centred frame window used by validation, test and tabular inputs."""
    x = np.asarray(power, dtype=np.float32)
    if x.ndim != 2 or n_win <= 0 or x.shape[1] < n_win:
        raise ValueError("power must contain at least n_win frames")
    start = (x.shape[1] - n_win) // 2
    return x[:, start:start + n_win]


def random_training_window(power: np.ndarray, n_win: int, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(power, dtype=np.float32)
    if x.ndim != 2 or n_win <= 0 or x.shape[1] < n_win:
        raise ValueError("power must contain at least n_win frames")
    start = int(rng.integers(0, x.shape[1] - n_win + 1))
    return x[:, start:start + n_win]


def mel_tabular_features(mel_power: np.ndarray, config: FrontEndConfig) -> np.ndarray:
    """Return the specified 84 level-free MFCC and mel-domain statistics."""
    power = np.asarray(mel_power, dtype=np.float32)
    if power.ndim != 2 or power.shape[0] != config.n_mels or power.shape[1] == 0:
        raise ValueError("mel_power must have shape (n_mels, nonzero_frames)")
    db = librosa.power_to_db(power, ref=1.0, amin=config.amin, top_db=None)
    mfcc = librosa.feature.mfcc(S=db, n_mfcc=20, dct_type=2, norm="ortho")[1:20]
    descriptors = (
        librosa.feature.spectral_centroid(S=power, freq=librosa.mel_frequencies(
            n_mels=config.n_mels, fmin=config.fmin, fmax=config.fmax,
            htk=config.mel_scale == "htk")),
        librosa.feature.spectral_bandwidth(S=power, freq=librosa.mel_frequencies(
            n_mels=config.n_mels, fmin=config.fmin, fmax=config.fmax,
            htk=config.mel_scale == "htk")),
        librosa.feature.spectral_rolloff(S=power, freq=librosa.mel_frequencies(
            n_mels=config.n_mels, fmin=config.fmin, fmax=config.fmax,
            htk=config.mel_scale == "htk")),
        librosa.feature.spectral_flatness(S=power),
    )
    values = [stat for row in mfcc for stat in (row.mean(), row.std(), row.min(), row.max())]
    values.extend(stat for descriptor in descriptors for stat in (descriptor.mean(), descriptor.std()))
    output = np.asarray(values, dtype=np.float32)
    if output.shape != (84,) or not np.all(np.isfinite(output)):
        raise ValueError("tabular front end failed to produce 84 finite values")
    return output


def cache_split_power(
    clips: Iterable[tuple[str, np.ndarray]],
    *, split: str, output_dir: str | Path, frontend: FrontEnd,
) -> tuple[Path, Path]:
    """Cache each full standardized clip's mel power and write its frame index."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    arrays: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    offset = 0
    for name, audio in clips:
        power = frontend.power(audio)
        if power.shape[1] == 0:
            raise ValueError(f"clip {name!r} is shorter than n_fft")
        rows.append({"file": name, "start_frame": offset, "n_frames": power.shape[1]})
        arrays.append(power.T)
        offset += power.shape[1]
    if not arrays:
        raise ValueError("no clips were provided for the power cache")
    array_path = root / f"{split}.npy"
    index_path = root / f"{split}.csv"
    if array_path.exists() or index_path.exists():
        raise FileExistsError(f"refusing to overwrite cache for split {split!r}")
    np.save(array_path, np.concatenate(arrays, axis=0).astype(np.float32), allow_pickle=False)
    with index_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("file", "start_frame", "n_frames"))
        writer.writeheader()
        writer.writerows(rows)
    return array_path, index_path


def prepare_waveform_window(audio: np.ndarray, config: FrontEndConfig) -> tuple[np.ndarray, bool]:
    """Apply the project's deterministic short-clip wrap rule and return its silent flag."""
    if config.clip_length_s is None:
        raise ValueError("clip_length_s awaits the dataset duration audit")
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("audio must be finite mono waveform data")
    silent = bool(x.size == 0 or np.all(x == 0))
    if x.size < config.win_samples:
        if x.size < config.win_samples / 2:
            raise ValueError("clip is below half the usable window and must be rejected")
        x = np.pad(x, (0, config.win_samples - x.size), mode="wrap")
    return x, silent


def standardize_audio(audio: np.ndarray, source_rate: int, config: FrontEndConfig) -> tuple[np.ndarray, dict[str, bool]]:
    """Select the configured channel, resample with soxr HQ, and apply clip rules."""
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 2:
        if not 0 <= config.channel < values.shape[1]:
            raise ValueError("configured channel is absent from the audio")
        values = values[:, config.channel]
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("audio must be finite mono or frames-by-channels data")
    flags = FrontEnd.quality_flags(values)
    if source_rate != config.sample_rate:
        values = resample(values, source_rate, config.sample_rate, quality=config.resample_quality)
    values, _silent = prepare_waveform_window(values, config)
    return values.astype(np.float32), flags


def choose_clip_length(
    durations_by_class: dict[str, Iterable[float]],
    *, sample_rate: int = 16_000, n_fft: int = 1024, hop: int = 256,
    candidates: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0), minimum_fraction: float = 0.95,
) -> dict[str, object]:
    """Apply the brief's per-class 95% usable-window duration rule."""
    if not durations_by_class or sample_rate <= 0 or n_fft <= 0 or hop <= 0:
        raise ValueError("class duration audit results and valid front-end dimensions are required")
    values = {name: np.asarray(tuple(durations), dtype=np.float64) for name, durations in durations_by_class.items()}
    if any(array.size == 0 or not np.all(np.isfinite(array)) or np.any(array < 0) for array in values.values()):
        raise ValueError("every class needs finite nonnegative duration measurements")
    report: dict[str, dict[str, float]] = {}
    selected = None
    for length in candidates:
        n_win = 1 + (int(length * sample_rate) - n_fft) // hop
        if n_win <= 0:
            report[str(length)] = {name: 0.0 for name in values}
            continue
        usable = n_fft + (n_win - 1) * hop
        fractions = {name: float(np.mean(samples * sample_rate >= usable)) for name, samples in values.items()}
        report[str(length)] = fractions
        if all(fraction >= minimum_fraction for fraction in fractions.values()):
            selected = float(length)
    if selected is None:
        raise RuntimeError("even the 0.5 s candidate failed the per-class 95% rule; owner decision required")
    n_win = 1 + (int(selected * sample_rate) - n_fft) // hop
    return {"clip_length_s": selected, "n_win": n_win,
            "win_samples": n_fft + (n_win - 1) * hop, "class_fractions": report}
