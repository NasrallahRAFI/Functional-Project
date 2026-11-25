"""Window selection, cached mel power and fixed-length tabular features."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path

import librosa
import numpy as np

warnings.warn(
    "dronedet.legacy_features is deprecated; use dronedet.features",
    DeprecationWarning,
    stacklevel=2,
)

from .features import _select_channel, _select_window, fit_length, standardize
from .legacy_frontend import FrontEnd, FrontEndConfig


def centred_window(power: np.ndarray, n_win: int) -> np.ndarray:
    """Take the fixed centred frame window used by validation, test and tabular inputs."""
    return _select_window(power, n_win)


def random_training_window(
    power: np.ndarray, n_win: int, rng: np.random.Generator
) -> np.ndarray:
    return _select_window(power, n_win, rng=rng)


def mel_tabular_features(mel_power: np.ndarray, config: FrontEndConfig) -> np.ndarray:
    """Return the legacy 84-value descriptor.

    This is intentionally not the canonical tabular adapter: it evaluates
    descriptor frequencies at ``n_mels`` points and retains librosa's default
    spectral-flatness ``amin`` and power. See W10's caller inventory for why
    old saved feature values must keep this behavior.
    """
    power = np.asarray(mel_power, dtype=np.float32)
    if power.ndim != 2 or power.shape[0] != config.n_mels or power.shape[1] == 0:
        raise ValueError("mel_power must have shape (n_mels, nonzero_frames)")
    db = librosa.power_to_db(power, ref=1.0, amin=config.amin, top_db=None)
    mfcc = librosa.feature.mfcc(S=db, n_mfcc=20, dct_type=2, norm="ortho")[1:20]
    frequencies = _legacy_mel_frequencies(config)
    descriptors = (
        librosa.feature.spectral_centroid(S=power, freq=frequencies),
        librosa.feature.spectral_bandwidth(S=power, freq=frequencies),
        librosa.feature.spectral_rolloff(S=power, freq=frequencies),
        librosa.feature.spectral_flatness(S=power),
    )
    values = [
        stat for row in mfcc for stat in (row.mean(), row.std(), row.min(), row.max())
    ]
    values.extend(
        stat
        for descriptor in descriptors
        for stat in (descriptor.mean(), descriptor.std())
    )
    output = np.asarray(values, dtype=np.float32)
    if output.shape != (84,) or not np.all(np.isfinite(output)):
        raise ValueError("tabular front end failed to produce 84 finite values")
    return output


def cache_split_power(
    clips: Iterable[tuple[str, np.ndarray]],
    *,
    split: str,
    output_dir: str | Path,
    frontend: FrontEnd,
) -> tuple[Path, Path]:
    """Reject the retired cache writer that emitted unbound transposed arrays."""
    warnings.warn(
        "legacy_features.cache_split_power is deprecated; use "
        "dronedet.features.cache_split_power with frozen paths and source_rate",
        DeprecationWarning,
        stacklevel=2,
    )
    raise RuntimeError(
        "the legacy cache format is retired; use "
        "dronedet.features.cache_split_power(clips, split=..., output_dir=..., "
        "config=..., paths=..., source_rate=...)"
    )


def prepare_waveform_window(
    audio: np.ndarray, config: FrontEndConfig
) -> tuple[np.ndarray, bool]:
    """Apply the deterministic wrap rule and return its silent flag.

    Current project windows are at least twice ``n_fft`` and use the canonical
    length checker. Shorter historical windows retain their old half-window
    threshold because it differs from the current minimum-length rule.
    """
    if config.clip_length_s is None:
        raise ValueError("clip_length_s awaits the dataset duration audit")
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("audio must be finite mono waveform data")
    silent = bool(x.size == 0 or np.all(x == 0))
    if config.win_samples >= 2 * config.n_fft:
        return fit_length(x, config), silent
    if x.size < config.win_samples:
        if x.size < config.win_samples / 2:
            raise ValueError(
                "clip is below half the usable window and must be rejected"
            )
        x = np.pad(x, (0, config.win_samples - x.size), mode="wrap")
    return x, silent


def standardize_audio(
    audio: np.ndarray, source_rate: int, config: FrontEndConfig
) -> tuple[np.ndarray, dict[str, bool]]:
    """Compatibility adapter to canonical channel selection and resampling."""
    selected = _select_channel(audio, config.channel)
    flags = FrontEnd.quality_flags(selected)
    values = standardize(audio, source_rate, config)
    values, _silent = prepare_waveform_window(values, config)
    return values.astype(np.float32, copy=False), {
        "silent": flags["silent"],
        "clipped": flags["clipped"],
    }


def _legacy_mel_frequencies(config: FrontEndConfig) -> np.ndarray:
    return librosa.mel_frequencies(
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        htk=config.mel_scale == "htk",
    )


def choose_clip_length(
    durations_by_class: dict[str, Iterable[float]],
    *,
    sample_rate: int = 16_000,
    n_fft: int = 1024,
    hop: int = 256,
    candidates: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
    minimum_fraction: float = 0.95,
) -> dict[str, object]:
    """Apply the brief's per-class 95% usable-window duration rule."""
    if not durations_by_class or sample_rate <= 0 or n_fft <= 0 or hop <= 0:
        raise ValueError(
            "class duration audit results and valid front-end dimensions are required"
        )
    values = {
        name: np.asarray(tuple(durations), dtype=np.float64)
        for name, durations in durations_by_class.items()
    }
    if any(
        array.size == 0 or not np.all(np.isfinite(array)) or np.any(array < 0)
        for array in values.values()
    ):
        raise ValueError("every class needs finite nonnegative duration measurements")
    report: dict[str, dict[str, float]] = {}
    selected = None
    for length in candidates:
        n_win = 1 + (int(length * sample_rate) - n_fft) // hop
        if n_win <= 0:
            report[str(length)] = {name: 0.0 for name in values}
            continue
        usable = n_fft + (n_win - 1) * hop
        fractions = {
            name: float(np.mean(samples * sample_rate >= usable))
            for name, samples in values.items()
        }
        report[str(length)] = fractions
        if all(fraction >= minimum_fraction for fraction in fractions.values()):
            selected = float(length)
    if selected is None:
        raise RuntimeError(
            "even the 0.5 s candidate failed the per-class 95% rule; owner decision required"
        )
    n_win = 1 + (int(selected * sample_rate) - n_fft) // hop
    return {
        "clip_length_s": selected,
        "n_win": n_win,
        "win_samples": n_fft + (n_win - 1) * hop,
        "class_fractions": report,
    }
