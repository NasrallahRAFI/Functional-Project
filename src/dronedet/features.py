"""Single public front-end API shared by notebooks, caches and demo."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from .config import FrontEndConfig
from .resample import resample

log = logging.getLogger(__name__)


def load_audio(path: str | Path, config: FrontEndConfig) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[1] <= config.channel:
        raise ValueError(f"configured channel {config.channel} is absent")
    return audio[:, config.channel], int(sr)


def standardize(audio: np.ndarray, source_rate: int, config: FrontEndConfig) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        if config.channel >= x.shape[1]:
            raise ValueError("configured channel is absent")
        x = x[:, config.channel]
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("training audio must be finite mono or frames-by-channels")
    if source_rate != config.sr:
        x = resample(x, source_rate, config.sr, quality=config.resample_quality)
    return x


def load_wave(path: str | Path, config: FrontEndConfig) -> np.ndarray:
    """Read the configured channel and resample it to the model sample rate."""
    audio, source_rate = load_audio(path, config)
    return standardize(audio, source_rate, config)


def clip_flags(audio: np.ndarray, config: FrontEndConfig) -> dict[str, bool]:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("quality flags require finite mono audio")
    silent = bool(x.size == 0 or np.all(x == 0))
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0
    dbfs = 20 * np.log10(max(rms, 1e-15))
    return {"silent": silent, "near_silent": bool(not silent and dbfs < config.near_silent_dbfs),
            "clipped": bool(x.size and np.mean(np.abs(x) >= config.clipped_peak) > config.clipped_fraction)}


def stats_exclude(silent: bool, near_silent: bool, crop_all_zero: bool) -> bool:
    """One exclusion predicate shared by split indexing and normalizer fitting."""
    return bool(silent or near_silent or crop_all_zero)


def fix_length(audio: np.ndarray, config: FrontEndConfig) -> tuple[np.ndarray, str]:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("waveform must be finite and mono")
    if x.size < config.n_fft:
        return np.empty(0, dtype=np.float32), "rejected_below_n_fft"
    if x.size < config.win_samples / 2:
        return np.empty(0, dtype=np.float32), "rejected_below_half_window"
    if x.size < config.win_samples:
        return np.pad(x, (0, config.win_samples - x.size), mode="wrap"), "wrap_padded"
    return x, "unchanged"


def fit_length(audio: np.ndarray, config: FrontEndConfig) -> np.ndarray:
    """Apply the plan's length rule and raise when a clip is below its minimum."""
    fixed, status = fix_length(audio, config)
    if not fixed.size:
        raise ValueError(status)
    return fixed


def mel_power(audio: np.ndarray, config: FrontEndConfig) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("mel_power expects finite mono audio")
    if x.size < config.n_fft:
        raise ValueError("audio shorter than n_fft")
    norm = np.inf if config.mel_norm == "peak" else "slaney"
    return np.asarray(librosa.feature.melspectrogram(
        y=x, sr=config.sr, n_fft=config.n_fft, hop_length=config.hop,
        win_length=config.win_length, window=config.window, center=config.center,
        power=config.power, n_mels=config.n_mels, fmin=config.fmin,
        fmax=config.fmax, htk=config.mel_scale == "htk", norm=norm,
        dtype=np.float32), dtype=np.float32)


def to_db(power: np.ndarray, config: FrontEndConfig) -> np.ndarray:
    return np.asarray(librosa.power_to_db(np.asarray(power, dtype=np.float32), ref=1.0,
        amin=config.amin, top_db=config.top_db), dtype=np.float32)


def file_to_power(path: str | Path, config: FrontEndConfig) -> tuple[np.ndarray, dict[str, Any]]:
    audio, sr = load_audio(path, config)
    audio = standardize(audio, sr, config)
    flags = clip_flags(audio, config)
    fixed, status = fix_length(audio, config)
    if not fixed.size:
        raise ValueError(status)
    return mel_power(fixed, config), {**flags, "length_status": status, "sample_rate": config.sr}


def centre_window(power: np.ndarray, config: FrontEndConfig) -> np.ndarray:
    x = np.asarray(power, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != config.n_mels or x.shape[1] < config.n_win:
        raise ValueError("power must contain at least n_win mel frames")
    start = (x.shape[1] - config.n_win) // 2
    return x[:, start:start + config.n_win]


def mel_centres(config: FrontEndConfig) -> np.ndarray:
    """Return HTK/Slaney mel centre frequencies from the filter-edge sequence."""
    return librosa.mel_frequencies(n_mels=config.n_mels + 2, fmin=config.fmin,
                                   fmax=config.fmax, htk=config.mel_scale == "htk")[1:-1]


def random_window(power: np.ndarray, config: FrontEndConfig, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(power, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != config.n_mels or x.shape[1] < config.n_win:
        raise ValueError("power must contain at least n_win mel frames")
    start = int(rng.integers(0, x.shape[1] - config.n_win + 1))
    return x[:, start:start + config.n_win]


def transform(power_window: np.ndarray, config: FrontEndConfig, stats: dict | None = None) -> np.ndarray:
    x = np.asarray(power_window, dtype=np.float32)
    if config.mode == "pcen":
        from .normalization import pcen_transform
        out = pcen_transform(x, config)
        if stats is None:
            raise ValueError("PCEN training statistics are required")
        out = (out - float(stats["mean"])) / float(stats["std"])
    elif config.mode == "per_window_mean":
        if stats is None: raise ValueError("per-window standard deviation is required")
        db = to_db(x, config)
        out = (db - db.mean()) / float(stats["std"])
    elif config.mode == "global_stats":
        if stats is None: raise ValueError("training per-bin statistics are required")
        out = (to_db(x, config) - np.asarray(stats["mean"], np.float32)[:, None]) / np.asarray(stats["std"], np.float32)[:, None]
    elif config.mode == "logstft":
        raise NotImplementedError("logstft ablation uses its own 513-bin cache")
    else:
        raise ValueError(f"unknown mode {config.mode}")
    return np.asarray(out, dtype=np.float32)[None, ...]


def tabular(full_clip_power: np.ndarray, config: FrontEndConfig) -> np.ndarray:
    return tabular_features(centre_window(full_clip_power, config), config)


def tabular_features(window_power: np.ndarray, config: FrontEndConfig) -> np.ndarray:
    """Build the 84 plan-defined values from one centred mel-power window."""
    p = np.asarray(window_power, dtype=np.float32)
    if p.shape != (config.n_mels, config.n_win) or not np.isfinite(p).all():
        raise ValueError("tabular feature window has the wrong shape or non-finite values")
    db = to_db(p, config)
    mfcc = librosa.feature.mfcc(S=db, n_mfcc=20, dct_type=2, norm="ortho")[1:20]
    freqs = mel_centres(config)
    desc = (librosa.feature.spectral_centroid(S=p, freq=freqs),
            librosa.feature.spectral_bandwidth(S=p, freq=freqs, p=2),
            librosa.feature.spectral_rolloff(S=p, freq=freqs, roll_percent=0.85),
            librosa.feature.spectral_flatness(S=p, amin=config.amin, power=1.0))
    vals = [v for row in mfcc for v in (row.mean(), row.std(), row.min(), row.max())]
    vals.extend(v for d in desc for v in (d.mean(), d.std()))
    out = np.asarray(vals, dtype=np.float32)
    if out.shape != (84,) or not np.all(np.isfinite(out)):
        raise ValueError("tabular feature extraction did not yield 84 finite values")
    return out


def feature_names() -> list[str]:
    stats = ("mean", "std", "min", "max")
    names = [f"mfcc{c:02d}_{stat}" for c in range(1, 20) for stat in stats]
    names.extend(f"{name}_{stat}" for name in ("centroid", "bandwidth", "rolloff", "flatness") for stat in ("mean", "std"))
    return names


def build_split_cache(df, fe, root):
    """Return concatenated mel powers, clip index, and 84-column feature matrix."""
    import pandas as pd
    rows, powers, vectors = [], [], []
    cursor = 0
    for r in df.itertuples():
        path = Path(root) / "Binary_Drone_Audio" / r.clip_id
        audio, sr = load_audio(path, fe)
        audio = standardize(audio, sr, fe)
        fixed, status = fix_length(audio, fe)
        if not fixed.size:
            raise ValueError(f"{r.clip_id}: {status}")
        power = mel_power(fixed, fe)
        frames = power.shape[1]
        win_start = (frames - fe.n_win) // 2
        window_power = power[:, win_start:win_start + fe.n_win]
        crop_zero = bool(np.all(window_power == 0.0))
        silent = bool(r.all_zero)
        near = bool(r.near_silent)
        rows.append({"clip_id": r.clip_id, "start_frame": cursor, "n_frames": frames,
                     "win_start_frame": win_start, "silent": silent, "near_silent": near,
                     "crop_all_zero": crop_zero,
                     "stats_exclude": stats_exclude(silent, near, crop_zero),
                     "stats_eligible": not stats_exclude(silent, near, crop_zero),
                     "length_status": status})
        vectors.append(tabular(window_power, fe))
        powers.append(power)
        cursor += frames
    return np.concatenate(powers, axis=1), pd.DataFrame(rows), np.vstack(vectors)


def global_stats(cache: np.ndarray, index, config):
    pieces = [to_db(cache[:, int(r.start_frame):int(r.start_frame) + int(r.n_frames)], config)
              for r in index[index.stats_eligible].itertuples()]
    values = np.concatenate(pieces, axis=1)
    return values.mean(axis=1, dtype=np.float64), values.std(axis=1, dtype=np.float64), values.shape[1]


def load_tabular(paths: dict, name: str) -> dict[str, np.ndarray]:
    """Load a frozen split feature archive without enabling pickle."""
    archive = np.load(Path(paths["tab"]) / f"{name}.npz", allow_pickle=False)
    return {key: archive[key] for key in archive.files}


def get_window(cache: np.ndarray, row, config: FrontEndConfig, rng: np.random.Generator | None = None) -> np.ndarray:
    """Read one mel window; fixed centre for evaluation, seeded random crop for training."""
    n_frames = int(row.n_frames)
    if n_frames < config.n_win:
        raise ValueError("clip has fewer frames than the configured window")
    offset = int(row.win_start_frame) if rng is None else int(rng.integers(0, n_frames - config.n_win + 1))
    start = int(row.start_frame) + offset
    return np.asarray(cache[:, start:start + config.n_win])


def cache_split_power(clips, *, split: str, output_dir: str | Path, config: FrontEndConfig,
                      logged_override: str | None = None) -> tuple[Path, Path]:
    """Write one concatenated float32 power cache and frame index for one split."""
    config.validate_cache_write(logged_override=logged_override)
    if split not in {"train", "validation", "val", "test", "fixture"}: raise ValueError("unknown split")
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    if config.passband_status != "measured":
        reason = logged_override or config.passband_override_reason
        (root / "passband_override.json").write_text(
            __import__("json").dumps({"reason": reason, "fmax": config.fmax}, indent=2) + "\n",
            encoding="utf8")
    rows, arrays, offset = [], [], 0
    for name, audio in clips:
        fixed, status = fix_length(standardize(audio, config.sr, config), config)
        if not fixed.size: raise ValueError(f"{name}: {status}")
        power = mel_power(fixed, config)
        rows.append({"file": str(name), "start_frame": offset, "n_frames": power.shape[1]})
        arrays.append(power.T); offset += power.shape[1]
    if not arrays: raise ValueError("no clips provided")
    array_path, index_path = root / f"{split}.npy", root / f"{split}.csv"
    if array_path.exists() or index_path.exists(): raise FileExistsError(f"refusing to overwrite {split} cache")
    np.save(array_path, np.concatenate(arrays).astype(np.float32), allow_pickle=False)
    import csv
    with index_path.open("x", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=("file", "start_frame", "n_frames")); w.writeheader(); w.writerows(rows)
    return array_path, index_path


def cache_split_tabular(clips, *, split: str, output_dir: str | Path, config: FrontEndConfig) -> Path:
    config.validate_cache_write()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for name, power in clips:
        rows.append((str(name), tabular(power, config)))
    if not rows: raise ValueError("no clips provided")
    path = Path(output_dir) / f"{split}_tabular.npz"
    if path.exists(): raise FileExistsError(f"refusing to overwrite {path.name}")
    matrix = np.stack([x[1] for x in rows]).astype(np.float32)
    if matrix.shape != (len(rows),84) or not np.all(np.isfinite(matrix)): raise ValueError("invalid tabular cache")
    np.savez_compressed(path, files=np.asarray([x[0] for x in rows]), features=matrix)
    return path


def wave_to_input(audio: np.ndarray, config: FrontEndConfig, stats: dict | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(audio, dtype=np.float32)
    flags = clip_flags(x, config)
    fixed, status = fix_length(x, config)
    if not fixed.size: raise ValueError(status)
    offset = ((len(fixed) - config.win_samples) // 2 // config.hop) * config.hop
    window = fixed[offset:offset + config.win_samples]
    return transform(mel_power(window, config), config, stats), {**flags, "length_status": status}


def run_edge_case_checks(config: FrontEndConfig) -> dict[str, bool]:
    zeros = np.zeros(config.win_samples, np.float32)
    assert np.isfinite(to_db(mel_power(zeros, config), config)).all()
    assert clip_flags(zeros, config)["silent"]
    clipped = np.full(config.win_samples, 0.999, np.float32)
    assert clip_flags(clipped, config)["clipped"]
    try: mel_power(np.array([np.nan], np.float32), config)
    except ValueError: pass
    else: raise AssertionError("NaN audio must be rejected")
    assert fix_length(np.zeros(config.n_fft - 1, np.float32), config)[1] == "rejected_below_n_fft"
    return {"silent_finite_floor": True, "clipped_flag": True, "nan_rejected": True, "too_short_rejected": True}
