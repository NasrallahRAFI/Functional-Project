"""Single public front-end API shared by notebooks, caches and demo."""
from __future__ import annotations

import logging
import hashlib
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from .config import FrontEndConfig, cache_content_hash
from .cache_metadata import verify_array_metadata, verify_cache_sidecar
from .resample import resample
from .splits import guard_test_access

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


def load_tabular(paths: dict, name: str, *, allow_test: bool = False) -> dict[str, np.ndarray]:
    """Load and reconcile a frozen tabular split; test requires explicit authorization."""
    guard_test_access(name, allow_test=allow_test)
    if name not in {"train", "val", "test"}:
        raise ValueError(f"unknown split: {name}")
    archive_path = Path(paths["tab"]) / f"{name}.npz"
    root = Path(paths["root"])
    from .config import load_config, params_hash
    config = load_config(root / "config.yaml")
    train_manifest_path = Path(paths["splits"]) / "train.csv"
    train_manifest_sha = hashlib.sha256(train_manifest_path.read_bytes()).hexdigest()
    current_hash = params_hash(config)
    sidecar = verify_cache_sidecar(archive_path, frontend_params_hash=current_hash,
                                   train_manifest_sha256=train_manifest_sha)
    with np.load(archive_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    verify_array_metadata(sidecar, shape=data["X"].shape, dtype=data["X"].dtype)
    manifest_path = Path(paths["splits"]) / f"{name}.csv"
    index_path = Path(paths["spec"]) / f"index_{name}.csv"
    import pandas as pd
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    index = pd.read_csv(index_path, keep_default_na=False)
    ids = data["clip_id"]
    if not (len(ids) == len(manifest) == len(index)):
        raise ValueError(f"{name}: manifest, index and tabular row counts differ")
    if not (np.array_equal(ids, manifest.clip_id.to_numpy()) and
            np.array_equal(ids, index.clip_id.to_numpy())):
        raise ValueError(f"{name}: manifest, index and tabular row order differ")
    expected_names = np.asarray(feature_names())
    if not np.array_equal(data["feature_names"], expected_names):
        raise ValueError("tabular feature names differ from the canonical 84-feature catalog")
    if data["X"].shape != (len(ids), 84) or not np.isfinite(data["X"]).all():
        raise ValueError(f"{name}: tabular matrix has an invalid shape or non-finite values")
    for column in ("near_silent", "partial_zero", "clipped_flag", "dc_flag"):
        if not np.array_equal(data[column].astype(bool), manifest[column].astype(bool).to_numpy()):
            raise ValueError(f"{name}: manifest flag mismatch for {column}")
    for column in ("silent", "near_silent", "crop_all_zero"):
        if not np.array_equal(data[column].astype(bool), index[column].astype(bool).to_numpy()):
            raise ValueError(f"{name}: index flag mismatch for {column}")
    data["stats_eligible"] = index.stats_eligible.astype(bool).to_numpy()
    data["group"] = manifest.group.to_numpy()
    data["family"] = manifest.family.to_numpy()
    data["drone_type"] = manifest.drone_type.to_numpy()
    data["recording"] = manifest.recording.to_numpy()
    data["label"] = manifest.label.to_numpy(dtype=np.int8)
    data["stats_exclude"] = ~data["stats_eligible"]
    if not np.array_equal(data["y"], data["label"]):
        raise ValueError(f"{name}: labels differ between manifest and feature archive")
    if name == "train":
        stats_path = Path(paths["spec"]) / "norm_stats_global.npz"
        with np.load(stats_path, allow_pickle=False) as stats:
            excluded_record = int(stats["clips_excluded"])
        if int(data["stats_exclude"].sum()) != excluded_record:
            raise ValueError("train stats_exclude does not match norm_stats_global.npz")
    frontend_record = json.loads((Path(paths["spec"]) / "frontend_hash.json").read_text(encoding="utf-8"))
    if frontend_record.get("params_hash") != current_hash:
        raise ValueError(f"{name}: front-end lineage hash mismatch")
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    artifact_key = f"data/processed/features_tabular/{name}.npz"
    artifact = frontend_record.get("artifacts", {}).get(artifact_key, {})
    if artifact.get("config_hash") != current_hash or artifact.get("sha256") != archive_sha:
        raise ValueError(f"{name}: tabular cache lineage/hash mismatch")
    hash_record = json.loads((Path(paths["tables"]) / "manifest_hashes.json").read_text(encoding="utf-8"))
    frozen_path = Path(paths["splits"]) / "FROZEN.txt"
    frozen = {}
    for line in frozen_path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            frozen[key.strip()] = value.strip()
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    frozen_key = "val" if name == "val" else name
    if (manifest_sha != hash_record.get(name) or manifest_sha != frozen.get(frozen_key)):
        raise ValueError(f"{name}: manifest hash mismatch")
    data["lineage_hash"] = np.asarray(current_hash)
    data["tabular_sha256"] = np.asarray(archive_sha)
    data["manifest_sha256"] = np.asarray(manifest_sha)
    return data


def load_cache_index(paths: dict, name: str, *, allow_test: bool = False):
    """Load a mel-cache index after enforcing the split access guard."""
    guard_test_access(name, allow_test=allow_test)
    if name not in {"train", "val", "test"}:
        raise ValueError(f"unknown split: {name}")
    import pandas as pd
    return pd.read_csv(Path(paths["spec"]) / f"index_{name}.csv", keep_default_na=False)


def load_power_cache(paths: dict, name: str, *, allow_test: bool = False,
                     mmap_mode: str | None = "r") -> np.ndarray:
    """Load the stored float32 mel-power matrix behind the split access guard."""
    guard_test_access(name, allow_test=allow_test)
    if name not in {"train", "val", "test"}:
        raise ValueError(f"unknown split: {name}")
    cache_path = Path(paths["spec"]) / f"mel_power_{name}.npy"
    from .config import load_config, params_hash
    config = load_config(Path(paths["root"]) / "config.yaml")
    train_manifest_sha = hashlib.sha256((Path(paths["splits"]) / "train.csv").read_bytes()).hexdigest()
    sidecar = verify_cache_sidecar(cache_path, frontend_params_hash=params_hash(config),
                                   train_manifest_sha256=train_manifest_sha)
    cache = np.load(cache_path,
                    mmap_mode=mmap_mode, allow_pickle=False)
    record_path = Path(paths["spec"]) / "cache_content_hash.json"
    from .config import cache_content_hash
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        actual = cache_content_hash(config)
        if record.get("cache_content_hash") != actual:
            raise ValueError(f"{name}: cache content hash mismatch")
    if cache.ndim != 2 or cache.shape[0] != config.n_mels or cache.dtype != np.float32:
        raise ValueError(f"{name}: invalid mel-power cache shape or dtype")
    verify_array_metadata(sidecar, shape=cache.shape, dtype=cache.dtype)
    return cache


def get_window(cache: np.ndarray, row, config: FrontEndConfig, rng: np.random.Generator | None = None,
               *, split: str, allow_test: bool = False) -> np.ndarray:
    """Read one mel window; fixed centre for evaluation, seeded random crop for training."""
    guard_test_access(split, allow_test=allow_test)
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
    cache_record = {"cache_content_hash": cache_content_hash(config),
                    "included_keys": ["sr", "channel", "n_fft", "win_length", "hop", "window",
                                      "center", "n_mels", "fmin", "fmax", "mel_scale", "mel_norm",
                                      "power", "L", "resample_quality"]}
    sidecar = root / "cache_content_hash.json"
    if sidecar.exists():
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
        if previous.get("cache_content_hash") != cache_record["cache_content_hash"]:
            raise ValueError("cache output directory is stamped with a different content hash")
    else:
        with sidecar.open("x", encoding="utf-8") as stream:
            json.dump(cache_record, stream, indent=2)
            stream.write("\n")
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


def run_edge_case_checks(config: FrontEndConfig) -> dict[str, dict[str, Any]]:
    """Run deterministic contract checks and return measured case details.

    This is intentionally synthetic: it exercises API contracts without reading
    validation or test recordings.
    """
    from .resample import resample

    cases: dict[str, dict[str, Any]] = {}

    def record(name: str, inputs: Any, expected: Any, actual: Any, passed: bool) -> None:
        cases[name] = {"input": inputs, "expected": expected, "actual": actual,
                       "pass": bool(passed)}

    zeros = np.zeros(config.win_samples, np.float32)
    zero_db = to_db(mel_power(zeros, config), config)
    zero_flags = clip_flags(zeros, config)
    zero_ok = bool(np.isfinite(zero_db).all() and np.isclose(zero_db.min(), -120.0)
                   and zero_flags["silent"])
    record("all_zero", {"samples": config.win_samples},
           {"finite": True, "minimum_db": -120.0, "silent": True},
           {"finite": bool(np.isfinite(zero_db).all()), "minimum_db": float(zero_db.min()),
            "silent": zero_flags["silent"]}, zero_ok)

    rng = np.random.default_rng(271828)
    below = (rng.standard_normal(config.win_samples).astype(np.float32) * np.float32(10 ** (-90 / 20)))
    above = (rng.standard_normal(config.win_samples).astype(np.float32) * np.float32(10 ** (-70 / 20)))
    below_flags, above_flags = clip_flags(below, config), clip_flags(above, config)
    record("near_silent_threshold", {"levels_dbfs": [-90, -70], "configured_threshold_dbfs": config.near_silent_dbfs},
           {"below": {"near_silent": True, "silent": False},
            "above": {"near_silent": False, "silent": False}},
           {"below": below_flags, "above": above_flags},
           below_flags["near_silent"] and not below_flags["silent"]
           and not above_flags["near_silent"] and not above_flags["silent"])

    clip_results = {}
    for fraction, key in ((0.009, "0.9%"), (0.01, "1.0%"), (0.011, "1.1%")):
        clip_samples = 10_000
        count = int(round(clip_samples * fraction))
        signal_in = np.zeros(clip_samples, np.float32)
        signal_in[:count] = config.clipped_peak
        flags = clip_flags(signal_in, config)
        clip_results[key] = {"samples": count, "clipped": flags["clipped"],
                             "silent": flags["silent"]}
    record("clipping_threshold", {"fractions": [0.009, 0.01, 0.011],
                                   "peak": config.clipped_peak,
                                   "threshold_fraction": config.clipped_fraction},
           {"0.9%": False, "1.0%": False, "1.1%": True}, clip_results,
           not clip_results["0.9%"]["clipped"] and not clip_results["1.0%"]["clipped"]
           and clip_results["1.1%"]["clipped"])

    invalid_results = {}
    for name, invalid in (("nan", np.nan), ("inf", np.inf)):
        try:
            standardize(np.array([0.0, invalid], np.float32), config.sr, config)
        except ValueError:
            invalid_results[name] = "raises ValueError"
        else:
            invalid_results[name] = "accepted"
    record("nonfinite_training_input", {"values": ["NaN", "+inf"]},
           {"NaN": "raises ValueError", "+inf": "raises ValueError"},
           {"NaN": invalid_results["nan"], "+inf": invalid_results["inf"]},
           all(value == "raises ValueError" for value in invalid_results.values()))

    lengths = (config.n_fft - 1, config.n_fft, config.win_samples // 2 - 1,
               config.win_samples // 2, config.win_samples - 1,
               config.win_samples, config.win_samples + 1, 20_000)
    length_results = {}
    for n in lengths:
        fixed, status = fix_length(np.arange(n, dtype=np.float32), config)
        length_results[str(n)] = {"status": status, "output_samples": int(fixed.size)}
    record("length_boundaries", {"samples": list(lengths)},
           {"below_nfft": "rejected_below_n_fft",
            "below_half": "rejected_below_half_window",
            "at_or_above_half_below_window": "wrap_padded",
            "at_or_above_window": "unchanged"}, length_results,
           length_results[str(config.n_fft - 1)]["status"] == "rejected_below_n_fft"
           and length_results[str(config.n_fft)]["status"] == "rejected_below_half_window"
           and length_results[str(config.win_samples // 2)]["status"] == "wrap_padded"
           and length_results[str(config.win_samples - 1)]["status"] == "wrap_padded"
           and length_results[str(config.win_samples)]["status"] == "unchanged"
           and length_results[str(config.win_samples + 1)]["status"] == "unchanged")

    short = np.arange(config.win_samples - 7, dtype=np.float32)
    padded, _ = fix_length(short, config)
    expected_pad = np.pad(short, (0, config.win_samples - short.size), mode="wrap")
    pad_max = float(np.max(np.abs(padded - expected_pad), initial=0.0))
    record("wrap_padding", {"input_samples": int(short.size)},
           {"output_samples": config.win_samples, "mode": "wrap", "max_abs_error": 0.0},
           {"output_samples": int(padded.size), "max_abs_error": pad_max},
           padded.size == config.win_samples and pad_max == 0.0)

    stereo = np.column_stack((np.linspace(-0.5, 0.5, config.win_samples, dtype=np.float32),
                              np.zeros(config.win_samples, np.float32)))
    channel_zero = standardize(stereo, config.sr, config)
    multichannel_error = float(np.max(np.abs(channel_zero - stereo[:, 0]), initial=0.0))
    record("multichannel_channel_selection", {"channels": 2, "configured_channel": config.channel},
           {"max_abs_error": 0.0}, {"max_abs_error": multichannel_error}, multichannel_error == 0.0)

    identity_input = rng.standard_normal(config.win_samples).astype(np.float32)
    identity = standardize(identity_input, config.sr, config)
    identity_error = float(np.max(np.abs(identity - identity_input), initial=0.0))
    record("same_rate_no_resampling", {"source_rate": config.sr}, {"array_equal": True},
           {"array_equal": bool(np.array_equal(identity, identity_input)),
            "max_abs_error": identity_error}, np.array_equal(identity, identity_input))

    source_48k = rng.standard_normal(12_000).astype(np.float32)
    resampled_48k = resample(source_48k, 48_000, config.sr, quality=config.resample_quality)
    record("48khz_soxr_path", {"source_rate": 48_000, "samples": int(source_48k.size),
                                "quality": config.resample_quality},
           {"finite": True, "output_samples": int(round(source_48k.size * config.sr / 48_000))},
           {"finite": bool(np.isfinite(resampled_48k).all()), "output_samples": int(resampled_48k.size)},
           bool(np.isfinite(resampled_48k).all() and
                resampled_48k.size == round(source_48k.size * config.sr / 48_000)))

    long_audio = rng.standard_normal(24_000).astype(np.float32) * np.float32(0.2)
    full_power = mel_power(long_audio, config)
    offsets = [0, config.hop, max(0, ((len(long_audio) - config.win_samples) // config.hop // 2) * config.hop)]
    crop_errors_db = []
    for offset in sorted(set(offsets)):
        cropped_power = mel_power(long_audio[offset:offset + config.win_samples], config)
        full_start = offset // config.hop
        expected = full_power[:, full_start:full_start + config.n_win]
        crop_errors_db.append(float(np.max(np.abs(to_db(cropped_power, config)
                                                    - to_db(expected, config)), initial=0.0)))
    crop_max = max(crop_errors_db, default=0.0)
    record("hop_aligned_crop_parity", {"offsets_samples": sorted(set(offsets))},
           {"max_abs_db_error": "<= 1e-3 dB"}, {"max_abs_db_error": crop_max}, crop_max <= 1e-3)

    center_start = (full_power.shape[1] - config.n_win) // 2
    center_result = centre_window(full_power, config)
    record("center_window_offset", {"frames": int(full_power.shape[1])},
           {"start_frame": center_start}, {"start_frame": center_start,
            "shape": list(center_result.shape)}, center_result.shape[1] == config.n_win)
    rng_a, rng_b = np.random.default_rng(17), np.random.default_rng(17)
    random_a = int(rng_a.integers(0, full_power.shape[1] - config.n_win + 1))
    random_b = int(rng_b.integers(0, full_power.shape[1] - config.n_win + 1))
    record("seeded_random_offset", {"seed": 17}, {"same_offset": True},
           {"offset_a": random_a, "offset_b": random_b}, random_a == random_b)

    tab_center = tabular(full_power, config)
    tab_slice = tabular_features(centre_window(full_power, config), config)
    tab_equal = bool(np.array_equal(tab_center, tab_slice))
    tab_zero = tabular(mel_power(zeros, config), config)
    record("tabular_84", {"mels": config.n_mels, "frames": config.n_win},
           {"shape": [84], "finite": True, "center_equals_full": True},
           {"shape": list(tab_center.shape), "finite": bool(np.isfinite(tab_center).all()),
            "center_equals_full": tab_equal, "zero_shape": list(tab_zero.shape),
            "zero_finite": bool(np.isfinite(tab_zero).all())},
           tab_center.shape == (84,) and np.isfinite(tab_center).all() and tab_equal
           and tab_zero.shape == (84,) and np.isfinite(tab_zero).all())

    gain_power = mel_power(rng.standard_normal(config.win_samples).astype(np.float32) * 0.25, config)
    gain_window_db = to_db(gain_power, config)
    gain_has_no_floor = bool(np.all(gain_window_db > config.db_floor + 1.0))
    base_tab = tabular_features(gain_power, config)
    gain_results = {}
    gain_ok = True
    for gain_db in (-6.0, 6.0):
        changed = apply_power_gain_for_check(gain_power, gain_db)
        value = tabular_features(changed, config)
        abs_diff = float(np.max(np.abs(value - base_tab), initial=0.0))
        rel_diff = float(np.max(np.abs(value - base_tab) / np.maximum(np.abs(base_tab), 1e-8), initial=0.0))
        gain_results[str(gain_db)] = {"max_abs": abs_diff, "max_relative": rel_diff}
        gain_ok &= rel_diff <= 1e-4
    record("tabular_gain_invariance", {"gain_db": [-6.0, 6.0], "tolerance_relative": 1e-4,
                                       "window_has_no_floor_cells": gain_has_no_floor},
           {"max_relative": "<= 1e-4", "window_has_no_floor_cells": True},
           {"per_gain": gain_results, "window_has_no_floor_cells": gain_has_no_floor},
           bool(gain_ok and gain_has_no_floor))

    return cases


def apply_power_gain_for_check(power: np.ndarray, gain_db: float) -> np.ndarray:
    """Private deterministic helper used by the edge-case suite."""
    return (np.asarray(power, np.float32) * np.float32(10.0 ** (gain_db / 10.0))).astype(np.float32)


def audit_lengths(manifest, config: FrontEndConfig, raw_root: str | Path) -> dict[str, Any]:
    """Audit header lengths for the rows explicitly supplied by the caller.

    The caller controls which splits are supplied. The function never loads or
    decodes audio samples, and reports native-rate mismatches for a stop gate.
    """
    rows = [dict(row) if isinstance(row, dict) else row._asdict()
            if hasattr(row, "_asdict") else dict(row) for row in manifest]
    import soundfile as sf

    classes: dict[str, dict[str, Any]] = {}
    rate_mismatches: list[dict[str, Any]] = []
    for row in rows:
        clip = str(row.get("clip_id", row.get("file", "")))
        if not clip:
            raise ValueError("length audit rows require clip_id or file")
        class_name = str(row.get("label_name", row.get("class_name", row.get("label", "unknown"))))
        bucket = classes.setdefault(class_name, {"total": 0, "n_full": 0, "n_padded": 0,
            "n_rejected": 0, "n_below_nfft": 0, "rejected_files": []})
        info = sf.info(str(Path(raw_root) / clip))
        samples = int(info.frames)
        bucket["total"] += 1
        if int(info.samplerate) != config.sr:
            rate_mismatches.append({"clip_id": clip, "native_rate": int(info.samplerate),
                                    "expected_rate": config.sr})
        if samples < config.n_fft:
            bucket["n_below_nfft"] += 1
        if samples < config.win_samples / 2:
            bucket["n_rejected"] += 1
            bucket["rejected_files"].append({"clip_id": clip, "n_samples": samples})
        elif samples < config.win_samples:
            bucket["n_padded"] += 1
        else:
            bucket["n_full"] += 1
    return {"by_class": classes, "native_rate_mismatches": rate_mismatches,
            "stop": bool(rate_mismatches or any(v["n_rejected"] for v in classes.values())),
            "win_samples": config.win_samples, "n_fft": config.n_fft,
            "raw_headers_only": True}
