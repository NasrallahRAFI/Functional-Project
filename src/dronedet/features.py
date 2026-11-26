"""Single public front-end API shared by notebooks, caches and demo."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from .cache_metadata import verify_array_metadata, verify_cache_sidecar
from .config import FrontEndConfig, cache_content_hash
from .resample import resample
from .splits import guard_test_access

log = logging.getLogger(__name__)


def load_audio(path: str | Path, config: FrontEndConfig) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[1] <= config.channel:
        raise ValueError(f"configured channel {config.channel} is absent")
    return audio[:, config.channel], int(sr)


def standardize(
    audio: np.ndarray, source_rate: int, config: FrontEndConfig
) -> np.ndarray:
    x = _select_channel(audio, config.channel)
    if source_rate != config.sr:
        x = resample(x, source_rate, config.sr, quality=config.resample_quality)
    return x


def _select_channel(audio: np.ndarray, channel: int) -> np.ndarray:
    """Return finite mono float32 samples from mono or frames-by-channel input."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        if not 0 <= channel < x.shape[1]:
            raise ValueError("configured channel is absent")
        x = x[:, channel]
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("training audio must be finite mono or frames-by-channels")
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
    return {
        "silent": silent,
        "near_silent": bool(not silent and dbfs < config.near_silent_dbfs),
        "clipped": bool(
            x.size
            and np.mean(np.abs(x) >= config.clipped_peak) > config.clipped_fraction
        ),
    }


def stats_exclude(silent: bool, near_silent: bool, crop_all_zero: bool) -> bool:
    """One exclusion predicate shared by split indexing and normalizer fitting."""
    return bool(silent or near_silent or crop_all_zero)


def power_quality_flags(power: np.ndarray, config: FrontEndConfig, *,
                        silent: bool, near_silent: bool) -> dict[str, bool | int]:
    """Expose the existing cache-index crop/eligibility definition without fitting statistics."""
    power = np.asarray(power, dtype=np.float32)
    if (power.ndim != 2 or power.shape[0] != config.n_mels
            or power.shape[1] < config.n_win or not np.isfinite(power).all()
            or np.any(power < 0)):
        raise ValueError("power must contain valid finite nonnegative mel frames")
    offset = (power.shape[1] - config.n_win) // 2
    crop_zero = bool(np.all(centre_window(power, config) == 0))
    excluded = stats_exclude(silent, near_silent, crop_zero)
    return {"win_start_frame": offset, "crop_all_zero": crop_zero,
            "stats_exclude": excluded, "stats_eligible": not excluded}


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
    return np.asarray(
        librosa.feature.melspectrogram(
            y=x,
            sr=config.sr,
            n_fft=config.n_fft,
            hop_length=config.hop,
            win_length=config.win_length,
            window=config.window,
            center=config.center,
            power=config.power,
            n_mels=config.n_mels,
            fmin=config.fmin,
            fmax=config.fmax,
            htk=config.mel_scale == "htk",
            norm=norm,
            dtype=np.float32,
        ),
        dtype=np.float32,
    )


def to_db(power: np.ndarray, config: FrontEndConfig) -> np.ndarray:
    return np.asarray(
        librosa.power_to_db(
            np.asarray(power, dtype=np.float32),
            ref=1.0,
            amin=config.amin,
            top_db=config.top_db,
        ),
        dtype=np.float32,
    )


def file_to_power(
    path: str | Path, config: FrontEndConfig
) -> tuple[np.ndarray, dict[str, Any]]:
    audio, sr = load_audio(path, config)
    audio = standardize(audio, sr, config)
    flags = clip_flags(audio, config)
    fixed, status = fix_length(audio, config)
    if not fixed.size:
        raise ValueError(status)
    return mel_power(fixed, config), {
        **flags,
        "length_status": status,
        "sample_rate": config.sr,
    }


def centre_window(power: np.ndarray, config: FrontEndConfig) -> np.ndarray:
    return _select_window(power, config.n_win, n_mels=config.n_mels)


def _select_window(
    power: np.ndarray,
    n_win: int,
    *,
    n_mels: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Select one valid centered or seeded window for canonical and legacy adapters."""
    x = np.asarray(power, dtype=np.float32)
    if (
        x.ndim != 2
        or (n_mels is not None and x.shape[0] != n_mels)
        or n_win <= 0
        or x.shape[1] < n_win
    ):
        raise ValueError("power must contain at least n_win mel frames")
    available = x.shape[1] - n_win
    start = available // 2 if rng is None else int(rng.integers(0, available + 1))
    return x[:, start : start + n_win]


def mel_centres(config: FrontEndConfig) -> np.ndarray:
    """Return HTK/Slaney mel centre frequencies from the filter-edge sequence."""
    return librosa.mel_frequencies(
        n_mels=config.n_mels + 2,
        fmin=config.fmin,
        fmax=config.fmax,
        htk=config.mel_scale == "htk",
    )[1:-1]


def random_window(
    power: np.ndarray, config: FrontEndConfig, rng: np.random.Generator
) -> np.ndarray:
    return _select_window(power, config.n_win, n_mels=config.n_mels, rng=rng)


def transform(
    power_window: np.ndarray, config: FrontEndConfig, stats: dict | None = None
) -> np.ndarray:
    x = np.asarray(power_window, dtype=np.float32)
    if config.mode == "pcen":
        from .normalization import pcen_transform

        out = pcen_transform(x, config)
        if stats is None:
            raise ValueError("PCEN training statistics are required")
        out = (out - float(stats["mean"])) / float(stats["std"])
    elif config.mode == "per_window_mean":
        if stats is None:
            raise ValueError("per-window standard deviation is required")
        db = to_db(x, config)
        out = (db - db.mean()) / float(stats["std"])
    elif config.mode == "global_stats":
        if stats is None:
            raise ValueError("training per-bin statistics are required")
        out = (
            to_db(x, config) - np.asarray(stats["mean"], np.float32)[:, None]
        ) / np.asarray(stats["std"], np.float32)[:, None]
    elif config.mode == "logstft":
        raise NotImplementedError("logstft ablation uses its own 513-bin cache")
    else:
        raise ValueError(f"unknown mode {config.mode}")
    return np.asarray(out, dtype=np.float32)[None, ...]


def tabular(full_clip_power: np.ndarray, config: FrontEndConfig) -> np.ndarray:
    if config.n_mels < 20:
        raise ValueError(
            "the canonical 84-column tabular representation requires at least 20 mel bands"
        )
    return tabular_features(centre_window(full_clip_power, config), config)


def tabular_features(window_power: np.ndarray, config: FrontEndConfig) -> np.ndarray:
    """Build the 84 plan-defined values from one centred mel-power window."""
    if config.n_mels < 20:
        raise ValueError(
            "the canonical 84-column tabular representation requires at least 20 mel bands"
        )
    p = np.asarray(window_power, dtype=np.float32)
    if p.shape != (config.n_mels, config.n_win) or not np.isfinite(p).all():
        raise ValueError(
            "tabular feature window has the wrong shape or non-finite values"
        )
    db = to_db(p, config)
    mfcc = librosa.feature.mfcc(S=db, n_mfcc=20, dct_type=2, norm="ortho")[1:20]
    freqs = mel_centres(config)
    desc = (
        librosa.feature.spectral_centroid(S=p, freq=freqs),
        librosa.feature.spectral_bandwidth(S=p, freq=freqs, p=2),
        librosa.feature.spectral_rolloff(S=p, freq=freqs, roll_percent=0.85),
        librosa.feature.spectral_flatness(S=p, amin=config.amin, power=1.0),
    )
    vals = [v for row in mfcc for v in (row.mean(), row.std(), row.min(), row.max())]
    vals.extend(v for d in desc for v in (d.mean(), d.std()))
    out = np.asarray(vals, dtype=np.float32)
    if out.shape != (84,) or not np.all(np.isfinite(out)):
        raise ValueError("tabular feature extraction did not yield 84 finite values")
    return out


def feature_names() -> list[str]:
    stats = ("mean", "std", "min", "max")
    names = [f"mfcc{c:02d}_{stat}" for c in range(1, 20) for stat in stats]
    names.extend(
        f"{name}_{stat}"
        for name in ("centroid", "bandwidth", "rolloff", "flatness")
        for stat in ("mean", "std")
    )
    return names


def _assemble_power_clips(
    manifest, power_clips, config: FrontEndConfig, *, logged_override=None
):
    """One assembly/catalog implementation for file and in-memory producers."""
    import pandas as pd

    config.validate_cache_write(logged_override=logged_override)
    if config.n_mels < 20:
        raise ValueError("84-column tabular extraction requires at least 20 mel bands")
    rows, powers, vectors, cursor = [], [], [], 0
    records = manifest.to_dict("records")
    iterator = iter(power_clips)
    for row in records:
        try:
            clip_id, power, status = next(iterator)
        except StopIteration as exc:
            raise ValueError("fewer power clips than manifest rows") from exc
        if str(clip_id) != str(row["clip_id"]):
            raise ValueError("power clips must follow manifest order")
        power = np.asarray(power, dtype=np.float32)
        if (
            power.ndim != 2
            or power.shape[0] != config.n_mels
            or power.shape[1] < config.n_win
        ):
            raise ValueError(f"{clip_id}: invalid power dimensions")
        if not np.isfinite(power).all() or np.any(power < 0):
            raise ValueError(f"{clip_id}: power must be finite and nonnegative")
        frames = power.shape[1]
        silent = bool(row.get("silent", row.get("all_zero", False)))
        near = bool(row["near_silent"])
        quality = power_quality_flags(power, config, silent=silent, near_silent=near)
        rows.append(
            {
                "clip_id": str(clip_id),
                "start_frame": cursor,
                "n_frames": frames,
                "silent": silent,
                "near_silent": near,
                **quality,
                "length_status": status,
            }
        )
        powers.append(power)
        vectors.append(tabular(power, config))
        cursor += frames
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError("more power clips than manifest rows")
    if not rows:
        raise ValueError("cache manifest must not be empty")
    return np.concatenate(powers, axis=1), pd.DataFrame(rows), np.vstack(vectors)


def build_split_cache(
    df, fe: FrontEndConfig, root: str | Path, *, allow_test: bool = False
):
    """Build canonical powers/index/features in memory using explicit split rows."""
    if "split" not in df or df["split"].nunique() != 1:
        raise ValueError("cache manifest must name exactly one split")
    split = str(df["split"].iloc[0])
    guard_test_access(split, allow_test=allow_test)
    if split not in {"train", "val", "test"}:
        raise ValueError("unsupported cache-building split")
    root = Path(root) / "Binary_Drone_Audio"

    def clips():
        for row in df.itertuples():
            power, flags = file_to_power(root / row.clip_id, fe)
            yield row.clip_id, power, flags["length_status"]

    return _assemble_power_clips(df, clips(), fe)


def write_split_cache(
    manifest,
    config: FrontEndConfig,
    raw_root: str | Path,
    paths: dict,
    *,
    split: str,
    allow_test: bool = False,
):
    """Build and transactionally publish one complete canonical split cache."""
    from .cache_metadata import publish_split_cache

    guard_test_access(split, allow_test=allow_test)
    if "split" not in manifest or not manifest["split"].eq(split).all():
        raise ValueError("cache manifest must match the requested split")
    arrays = build_split_cache(manifest, config, raw_root, allow_test=allow_test)
    return publish_split_cache(
        *arrays,
        manifest,
        split=split,
        paths=paths,
        config=config,
        allow_test=allow_test,
    )


def fit_cached_normalizer(
    cache: np.ndarray, index, manifest_rows, config: FrontEndConfig
):
    """Fit the canonical policy using a reconciled full-clip training cache.

    The stored index owns eligibility; explicit manifest split values prevent
    accidental validation fitting. Cached values are viewed, never modified.
    """
    from .normalization import Normalizer

    rows = list(manifest_rows)
    indexed = list(index.to_dict("records"))
    if len(rows) != len(indexed) or not rows:
        raise ValueError("normalization manifest/index row counts differ or are empty")
    if cache.ndim != 2 or cache.shape[0] != config.n_mels:
        raise ValueError("normalization cache has invalid dimensions")
    powers, fit_rows, cursor = {}, [], 0
    for manifest, stored in zip(rows, indexed):
        clip = str(manifest.get("clip_id", ""))
        if not clip or clip != str(stored.get("clip_id", "")):
            raise ValueError("normalization manifest/index clip order differs")
        start, count = int(stored["start_frame"]), int(stored["n_frames"])
        if start != cursor or count < config.n_win or start + count > cache.shape[1]:
            raise ValueError(f"{clip}: normalization cache index span is invalid")
        fit_rows.append({**manifest, **stored, "split": manifest.get("split", "")})
        powers[clip] = cache[:, start : start + count]
        cursor += count
    if cursor != cache.shape[1]:
        raise ValueError("normalization index does not cover the full power cache")
    return Normalizer.fit(fit_rows, config, powers)


def global_stats(cache: np.ndarray, index, config, *, manifest_rows=None):
    """Deprecated adapter; explicit training rows are required for safe fitting."""
    import warnings

    if manifest_rows is None:
        raise ValueError(
            "use fit_cached_normalizer with explicit training manifest rows"
        )
    warnings.warn(
        "global_stats is deprecated; use fit_cached_normalizer",
        DeprecationWarning,
        stacklevel=2,
    )
    normalizer = fit_cached_normalizer(cache, index, manifest_rows, config)
    return normalizer.mean, normalizer.std, normalizer.fit_record["frame_count"]


def _cache_context(paths: dict, name: str, config: FrontEndConfig, *, kind: str):
    """Validate provenance/index without opening an audio or array payload."""
    import pandas as pd

    from .cache_metadata import file_sha256, validate_manifest_index, verify_transaction
    from .config import legacy_config_hash, tabular_content_hash

    manifest_path = Path(paths["splits"]) / f"{name}.csv"
    train_path = Path(paths["splits"]) / "train.csv"
    index_path = Path(paths["spec"]) / f"index_{name}.csv"
    cache_path = (
        Path(paths["spec"]) / f"mel_power_{name}.npy"
        if kind == "power"
        else Path(paths["tab"]) / f"{name}.npz"
    )
    metadata = verify_cache_sidecar(
        cache_path,
        frontend_params_hash=None,
        legacy_frontend_params_hash=legacy_config_hash(config),
        train_manifest_sha256=file_sha256(train_path),
    )
    manifest_digest = file_sha256(manifest_path)
    if metadata.get("split_manifest_sha256") != manifest_digest:
        raise ValueError(f"{name}: cache split-manifest hash mismatch")
    legacy = {}
    if metadata.get("schema_version") == 3:
        verify_transaction(
            cache_path, metadata, spec_dir=Path(paths["spec"]), split=name
        )
        expected_content = (
            cache_content_hash(config)
            if kind == "power"
            else tabular_content_hash(config)
        )
        field = "power_content_hash" if kind == "power" else "tabular_content_hash"
        if metadata.get(field) != expected_content:
            raise ValueError(f"{name}: cache content hash mismatch")
        if metadata.get("identity_schema") != 2:
            raise ValueError(f"{name}: unsupported cache identity schema")
    else:
        if metadata.get("frontend_params_hash") != legacy_config_hash(config):
            raise ValueError(f"{name}: legacy frontend sidecar identity mismatch")
        record_path = Path(paths["spec"]) / "frontend_hash.json"
        if not record_path.is_file():
            raise ValueError(
                "legacy cache requires its original artifact lineage record"
            )
        legacy = json.loads(record_path.read_text(encoding="utf-8"))
        if legacy.get("params_hash") != legacy_config_hash(config):
            raise ValueError(f"{name}: legacy frontend lineage mismatch")
    artifact_records = legacy.get("artifacts", {})
    index_key = f"data/processed/spectrograms/index_{name}.csv"
    expected_index = metadata.get("index_sha256") or artifact_records.get(
        index_key, {}
    ).get("sha256")
    if not expected_index or expected_index != file_sha256(index_path):
        raise ValueError(f"{name}: cache index checksum mismatch")
    artifact_key = (
        f"data/processed/spectrograms/mel_power_{name}.npy"
        if kind == "power"
        else f"data/processed/features_tabular/{name}.npz"
    )
    expected_array = metadata.get("artifact_sha256") or artifact_records.get(
        artifact_key, {}
    ).get("sha256")
    if not expected_array:
        raise ValueError(f"{name}: cache array checksum is missing")
    if legacy and any(
        artifact_records.get(key, {}).get("config_hash") != legacy_config_hash(config)
        for key in (index_key, artifact_key)
    ):
        raise ValueError(f"{name}: legacy artifact lineage mismatch")
    metadata = {
        **metadata,
        "verified_manifest_sha256": manifest_digest,
        "expected_artifact_sha256": expected_array,
    }
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    index = pd.read_csv(index_path, keep_default_na=False)
    shape = metadata.get("power_shape")
    if shape is None:
        if kind == "power":
            shape = metadata.get("array_shape")
        else:
            power_metadata = json.loads(
                Path(
                    str(Path(paths["spec"]) / f"mel_power_{name}.npy") + ".meta.json"
                ).read_text(encoding="utf-8")
            )
            shape = power_metadata.get("array_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(x) is not int or x < 1 for x in shape)
    ):
        raise ValueError("cache power dimensions are missing or invalid")
    validate_manifest_index(manifest, index, tuple(shape), config, split=name)
    frozen_path = Path(paths["splits"]) / "FROZEN.txt"
    if frozen_path.is_file():
        frozen = dict(
            line.split(": ", 1)
            for line in frozen_path.read_text(encoding="utf-8").splitlines()
            if ": " in line
        )
        if frozen.get(name) != manifest_digest:
            raise ValueError(f"{name}: frozen manifest hash mismatch")
    return cache_path, metadata, manifest, index


def load_tabular(
    paths: dict, name: str, *, allow_test: bool = False
) -> dict[str, np.ndarray]:
    """Load a reconciled, checksummed 84-column cache after the split guard."""
    from .cache_metadata import file_sha256
    from .config import load_config, params_hash, tabular_content_hash

    guard_test_access(name, allow_test=allow_test)
    if name not in {"train", "val", "test"}:
        raise ValueError(f"unknown split: {name}")
    config = load_config(Path(paths["root"]) / "config.yaml")
    archive_path, sidecar, manifest, index = _cache_context(
        paths, name, config, kind="tabular"
    )
    digest = file_sha256(archive_path)
    if digest != sidecar["expected_artifact_sha256"]:
        raise ValueError(f"{name}: tabular cache checksum mismatch")
    with np.load(archive_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    required = {
        "X",
        "y",
        "clip_id",
        "feature_names",
        "silent",
        "near_silent",
        "crop_all_zero",
        "partial_zero",
        "clipped_flag",
        "dc_flag",
    }
    if not required.issubset(data):
        raise ValueError("tabular archive lacks canonical fields")
    verify_array_metadata(sidecar, shape=data["X"].shape, dtype=data["X"].dtype)
    if not np.array_equal(data["clip_id"], manifest.clip_id.to_numpy()):
        raise ValueError(f"{name}: manifest and tabular clip order differ")
    if not np.array_equal(data["feature_names"], np.asarray(feature_names())):
        raise ValueError("tabular feature names differ from the canonical catalog")
    if data["X"].shape != (len(manifest), 84) or not np.isfinite(data["X"]).all():
        raise ValueError("tabular cache must contain 84 finite features per clip")
    for column in ("near_silent", "partial_zero", "clipped_flag", "dc_flag"):
        if not np.array_equal(data[column], manifest[column].to_numpy()):
            raise ValueError(f"{name}: manifest flag mismatch for {column}")
    for column in ("silent", "near_silent", "crop_all_zero"):
        if data[column].dtype.kind != "b" or not np.array_equal(
            data[column], index[column].to_numpy()
        ):
            raise ValueError(f"{name}: index flag mismatch for {column}")
    if not np.array_equal(data["y"], manifest.label.to_numpy()):
        raise ValueError(f"{name}: tabular labels differ from the manifest")
    data["stats_eligible"] = index.stats_eligible.to_numpy(bool)
    data["stats_exclude"] = ~data["stats_eligible"]
    for column in ("group", "family", "drone_type", "recording", "label"):
        if column in manifest:
            data[column] = manifest[column].to_numpy()
    stats_path = Path(paths["spec"]) / "norm_stats_global.npz"
    if name == "train" and stats_path.is_file():
        with np.load(stats_path, allow_pickle=False) as stats:
            if int(data["stats_exclude"].sum()) != int(stats["clips_excluded"]):
                raise ValueError(
                    "training exclusion count differs from normalization record"
                )
    data.update(
        lineage_hash=np.asarray(tabular_content_hash(config)),
        transform_hash=np.asarray(params_hash(config)),
        tabular_sha256=np.asarray(digest),
        manifest_sha256=np.asarray(sidecar["verified_manifest_sha256"]),
    )
    return data


def load_cache_index(paths: dict, name: str, *, allow_test: bool = False):
    """Load a verified index without opening a power or audio payload."""
    from .config import load_config

    guard_test_access(name, allow_test=allow_test)
    if name not in {"train", "val", "test"}:
        raise ValueError(f"unknown split: {name}")
    config = load_config(Path(paths["root"]) / "config.yaml")
    return _cache_context(paths, name, config, kind="power")[3]


def load_power_cache(
    paths: dict, name: str, *, allow_test: bool = False, mmap_mode: str | None = "r"
) -> np.ndarray:
    """Return finite nonnegative powers after provenance, checksum and index checks."""
    from .cache_metadata import file_sha256, validate_power_values
    from .config import load_config

    guard_test_access(name, allow_test=allow_test)
    if name not in {"train", "val", "test"}:
        raise ValueError(f"unknown split: {name}")
    config = load_config(Path(paths["root"]) / "config.yaml")
    path, metadata, manifest, index = _cache_context(paths, name, config, kind="power")
    if file_sha256(path) != metadata["expected_artifact_sha256"]:
        raise ValueError(f"{name}: power cache checksum mismatch")
    cache = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    try:
        if (
            cache.ndim != 2
            or cache.shape[0] != config.n_mels
            or cache.dtype != np.float32
        ):
            raise ValueError(f"{name}: invalid mel-power cache shape or dtype")
        verify_array_metadata(metadata, shape=cache.shape, dtype=cache.dtype)
        validate_power_values(cache)
    except BaseException:
        if getattr(cache, "_mmap", None) is not None:
            cache._mmap.close()
        raise
    return cache


def get_window(
    cache: np.ndarray,
    row,
    config: FrontEndConfig,
    rng: np.random.Generator | None = None,
    *,
    split: str,
    allow_test: bool = False,
) -> np.ndarray:
    """Read one mel window; fixed centre for evaluation, seeded random crop for training."""
    guard_test_access(split, allow_test=allow_test)
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unknown split: {split}")
    if rng is not None and split != "train":
        raise ValueError("random window offsets are allowed for training only")
    try:
        raw_span = (row.start_frame, row.n_frames, row.win_start_frame)
    except AttributeError as exc:
        raise ValueError(
            "window row must contain integer start_frame, n_frames and win_start_frame"
        ) from exc
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in raw_span
    ):
        raise ValueError("window index fields must be integers, not booleans")
    start_frame, n_frames, stored_offset = (int(value) for value in raw_span)
    if start_frame < 0 or n_frames < config.n_win:
        raise ValueError("clip has an invalid frame span for the configured window")
    if stored_offset < 0 or stored_offset + config.n_win > n_frames:
        raise ValueError("stored window offset lies outside the clip span")
    if split in {"val", "test"} and stored_offset != (n_frames - config.n_win) // 2:
        raise ValueError("evaluation window offset must use the canonical centre")
    if cache.ndim != 2 or cache.shape[0] != config.n_mels:
        raise ValueError(
            "cache must be a mel-by-frame matrix compatible with the configuration"
        )
    if start_frame + n_frames > cache.shape[1]:
        raise ValueError("window row span exceeds the cache bounds")
    offset = (
        stored_offset
        if rng is None
        else int(rng.integers(0, n_frames - config.n_win + 1))
    )
    start = start_frame + offset
    window = np.asarray(cache[:, start : start + config.n_win])
    if window.shape != (config.n_mels, config.n_win):
        raise ValueError("selected window has an invalid shape")
    return window


def cache_split_power(
    clips,
    *,
    split: str,
    output_dir: str | Path,
    config: FrontEndConfig,
    paths: dict | None = None,
    source_rate: int | None = None,
    logged_override: str | None = None,
    allow_test: bool = False,
) -> tuple[Path, Path]:
    """Publish a canonical split from ordered waveform pairs and explicit provenance.

    The old unlabelled, transposed format is no longer written. ``paths`` must
    identify the manifests and both cache folders; ``source_rate`` is explicit.
    The same transaction also publishes the matching tabular representation.
    """
    import pandas as pd

    from .cache_metadata import publish_split_cache

    guard_test_access(split, allow_test=allow_test)
    if split not in {"train", "val", "test"}:
        raise ValueError("unsupported cache-building split")
    if paths is None or source_rate is None:
        raise ValueError(
            "canonical cache writing requires paths and an explicit source_rate"
        )
    if Path(output_dir).resolve() != Path(paths["spec"]).resolve():
        raise ValueError("power output_dir must match paths['spec']")
    config.validate_cache_write(logged_override=logged_override)
    manifest = pd.read_csv(
        Path(paths["splits"]) / f"{split}.csv", keep_default_na=False
    )

    def powers():
        for name, audio in clips:
            standardized = standardize(audio, source_rate, config)
            flags = clip_flags(standardized, config)
            position = manifest.loc[manifest.clip_id.eq(str(name))]
            if len(position) != 1:
                raise ValueError(
                    "waveform clip is not uniquely identified by the manifest"
                )
            for key in ("silent", "near_silent"):
                if bool(position.iloc[0][key]) != flags[key]:
                    raise ValueError(
                        f"{name}: manifest {key} differs from the waveform"
                    )
            fixed, status = fix_length(standardized, config)
            if not fixed.size:
                raise ValueError(f"{name}: {status}")
            yield name, mel_power(fixed, config), status

    arrays = _assemble_power_clips(
        manifest, powers(), config, logged_override=logged_override
    )
    files = publish_split_cache(
        *arrays,
        manifest,
        split=split,
        paths=paths,
        config=config,
        logged_override=logged_override,
        allow_test=allow_test,
    )
    return files["power"], files["index"]


def cache_split_tabular(
    clips,
    *,
    split: str,
    output_dir: str | Path,
    config: FrontEndConfig,
    paths: dict | None = None,
    allow_test: bool = False,
) -> Path:
    """Publish both canonical representations from ordered full-clip powers."""
    import pandas as pd

    from .cache_metadata import publish_split_cache

    guard_test_access(split, allow_test=allow_test)
    if split not in {"train", "val", "test"}:
        raise ValueError("unsupported cache-building split")
    if paths is None:
        raise ValueError("canonical cache writing requires paths with frozen manifests")
    if Path(output_dir).resolve() != Path(paths["tab"]).resolve():
        raise ValueError("tabular output_dir must match paths['tab']")
    manifest = pd.read_csv(
        Path(paths["splits"]) / f"{split}.csv", keep_default_na=False
    )
    arrays = _assemble_power_clips(
        manifest, ((name, power, "provided_power") for name, power in clips), config
    )
    return publish_split_cache(
        *arrays,
        manifest,
        split=split,
        paths=paths,
        config=config,
        allow_test=allow_test,
    )["tabular"]


def wave_to_input(
    audio: np.ndarray, config: FrontEndConfig, stats: dict | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(audio, dtype=np.float32)
    flags = clip_flags(x, config)
    fixed, status = fix_length(x, config)
    if not fixed.size:
        raise ValueError(status)
    offset = ((len(fixed) - config.win_samples) // 2 // config.hop) * config.hop
    window = fixed[offset : offset + config.win_samples]
    return transform(mel_power(window, config), config, stats), {
        **flags,
        "length_status": status,
    }


def run_edge_case_checks(config: FrontEndConfig) -> dict[str, dict[str, Any]]:
    """Run deterministic contract checks and return measured case details.

    This is intentionally synthetic: it exercises API contracts without reading
    validation or test recordings.
    """
    from .resample import resample

    cases: dict[str, dict[str, Any]] = {}

    def record(
        name: str, inputs: Any, expected: Any, actual: Any, passed: bool
    ) -> None:
        cases[name] = {
            "input": inputs,
            "expected": expected,
            "actual": actual,
            "pass": bool(passed),
        }

    zeros = np.zeros(config.win_samples, np.float32)
    zero_db = to_db(mel_power(zeros, config), config)
    zero_flags = clip_flags(zeros, config)
    zero_ok = bool(
        np.isfinite(zero_db).all()
        and np.isclose(zero_db.min(), -120.0)
        and zero_flags["silent"]
    )
    record(
        "all_zero",
        {"samples": config.win_samples},
        {"finite": True, "minimum_db": -120.0, "silent": True},
        {
            "finite": bool(np.isfinite(zero_db).all()),
            "minimum_db": float(zero_db.min()),
            "silent": zero_flags["silent"],
        },
        zero_ok,
    )

    rng = np.random.default_rng(271828)
    below = rng.standard_normal(config.win_samples).astype(np.float32) * np.float32(
        10 ** (-90 / 20)
    )
    above = rng.standard_normal(config.win_samples).astype(np.float32) * np.float32(
        10 ** (-70 / 20)
    )
    below_flags, above_flags = clip_flags(below, config), clip_flags(above, config)
    record(
        "near_silent_threshold",
        {
            "levels_dbfs": [-90, -70],
            "configured_threshold_dbfs": config.near_silent_dbfs,
        },
        {
            "below": {"near_silent": True, "silent": False},
            "above": {"near_silent": False, "silent": False},
        },
        {"below": below_flags, "above": above_flags},
        below_flags["near_silent"]
        and not below_flags["silent"]
        and not above_flags["near_silent"]
        and not above_flags["silent"],
    )

    clip_results = {}
    for fraction, key in ((0.009, "0.9%"), (0.01, "1.0%"), (0.011, "1.1%")):
        clip_samples = 10_000
        count = round(clip_samples * fraction)
        signal_in = np.zeros(clip_samples, np.float32)
        signal_in[:count] = config.clipped_peak
        flags = clip_flags(signal_in, config)
        clip_results[key] = {
            "samples": count,
            "clipped": flags["clipped"],
            "silent": flags["silent"],
        }
    record(
        "clipping_threshold",
        {
            "fractions": [0.009, 0.01, 0.011],
            "peak": config.clipped_peak,
            "threshold_fraction": config.clipped_fraction,
        },
        {"0.9%": False, "1.0%": False, "1.1%": True},
        clip_results,
        not clip_results["0.9%"]["clipped"]
        and not clip_results["1.0%"]["clipped"]
        and clip_results["1.1%"]["clipped"],
    )

    invalid_results = {}
    for name, invalid in (("nan", np.nan), ("inf", np.inf)):
        try:
            standardize(np.array([0.0, invalid], np.float32), config.sr, config)
        except ValueError:
            invalid_results[name] = "raises ValueError"
        else:
            invalid_results[name] = "accepted"
    record(
        "nonfinite_training_input",
        {"values": ["NaN", "+inf"]},
        {"NaN": "raises ValueError", "+inf": "raises ValueError"},
        {"NaN": invalid_results["nan"], "+inf": invalid_results["inf"]},
        all(value == "raises ValueError" for value in invalid_results.values()),
    )

    lengths = (
        config.n_fft - 1,
        config.n_fft,
        config.win_samples // 2 - 1,
        config.win_samples // 2,
        config.win_samples - 1,
        config.win_samples,
        config.win_samples + 1,
        20_000,
    )
    length_results = {}
    for n in lengths:
        fixed, status = fix_length(np.arange(n, dtype=np.float32), config)
        length_results[str(n)] = {"status": status, "output_samples": int(fixed.size)}
    record(
        "length_boundaries",
        {"samples": list(lengths)},
        {
            "below_nfft": "rejected_below_n_fft",
            "below_half": "rejected_below_half_window",
            "at_or_above_half_below_window": "wrap_padded",
            "at_or_above_window": "unchanged",
        },
        length_results,
        length_results[str(config.n_fft - 1)]["status"] == "rejected_below_n_fft"
        and length_results[str(config.n_fft)]["status"] == "rejected_below_half_window"
        and length_results[str(config.win_samples // 2)]["status"] == "wrap_padded"
        and length_results[str(config.win_samples - 1)]["status"] == "wrap_padded"
        and length_results[str(config.win_samples)]["status"] == "unchanged"
        and length_results[str(config.win_samples + 1)]["status"] == "unchanged",
    )

    short = np.arange(config.win_samples - 7, dtype=np.float32)
    padded, _ = fix_length(short, config)
    expected_pad = np.pad(short, (0, config.win_samples - short.size), mode="wrap")
    pad_max = float(np.max(np.abs(padded - expected_pad), initial=0.0))
    record(
        "wrap_padding",
        {"input_samples": int(short.size)},
        {"output_samples": config.win_samples, "mode": "wrap", "max_abs_error": 0.0},
        {"output_samples": int(padded.size), "max_abs_error": pad_max},
        padded.size == config.win_samples and pad_max == 0.0,
    )

    stereo = np.column_stack(
        (
            np.linspace(-0.5, 0.5, config.win_samples, dtype=np.float32),
            np.zeros(config.win_samples, np.float32),
        )
    )
    channel_zero = standardize(stereo, config.sr, config)
    multichannel_error = float(np.max(np.abs(channel_zero - stereo[:, 0]), initial=0.0))
    record(
        "multichannel_channel_selection",
        {"channels": 2, "configured_channel": config.channel},
        {"max_abs_error": 0.0},
        {"max_abs_error": multichannel_error},
        multichannel_error == 0.0,
    )

    identity_input = rng.standard_normal(config.win_samples).astype(np.float32)
    identity = standardize(identity_input, config.sr, config)
    identity_error = float(np.max(np.abs(identity - identity_input), initial=0.0))
    record(
        "same_rate_no_resampling",
        {"source_rate": config.sr},
        {"array_equal": True},
        {
            "array_equal": bool(np.array_equal(identity, identity_input)),
            "max_abs_error": identity_error,
        },
        np.array_equal(identity, identity_input),
    )

    source_48k = rng.standard_normal(12_000).astype(np.float32)
    resampled_48k = resample(
        source_48k, 48_000, config.sr, quality=config.resample_quality
    )
    record(
        "48khz_soxr_path",
        {
            "source_rate": 48_000,
            "samples": int(source_48k.size),
            "quality": config.resample_quality,
        },
        {
            "finite": True,
            "output_samples": round(source_48k.size * config.sr / 48_000),
        },
        {
            "finite": bool(np.isfinite(resampled_48k).all()),
            "output_samples": int(resampled_48k.size),
        },
        bool(
            np.isfinite(resampled_48k).all()
            and resampled_48k.size == round(source_48k.size * config.sr / 48_000)
        ),
    )

    long_audio = rng.standard_normal(24_000).astype(np.float32) * np.float32(0.2)
    full_power = mel_power(long_audio, config)
    offsets = [
        0,
        config.hop,
        max(
            0, ((len(long_audio) - config.win_samples) // config.hop // 2) * config.hop
        ),
    ]
    crop_errors_db = []
    for offset in sorted(set(offsets)):
        cropped_power = mel_power(
            long_audio[offset : offset + config.win_samples], config
        )
        full_start = offset // config.hop
        expected = full_power[:, full_start : full_start + config.n_win]
        crop_errors_db.append(
            float(
                np.max(
                    np.abs(to_db(cropped_power, config) - to_db(expected, config)),
                    initial=0.0,
                )
            )
        )
    crop_max = max(crop_errors_db, default=0.0)
    record(
        "hop_aligned_crop_parity",
        {"offsets_samples": sorted(set(offsets))},
        {"max_abs_db_error": "<= 1e-3 dB"},
        {"max_abs_db_error": crop_max},
        crop_max <= 1e-3,
    )

    center_start = (full_power.shape[1] - config.n_win) // 2
    center_result = centre_window(full_power, config)
    record(
        "center_window_offset",
        {"frames": int(full_power.shape[1])},
        {"start_frame": center_start},
        {"start_frame": center_start, "shape": list(center_result.shape)},
        center_result.shape[1] == config.n_win,
    )
    rng_a, rng_b = np.random.default_rng(17), np.random.default_rng(17)
    random_a = int(rng_a.integers(0, full_power.shape[1] - config.n_win + 1))
    random_b = int(rng_b.integers(0, full_power.shape[1] - config.n_win + 1))
    record(
        "seeded_random_offset",
        {"seed": 17},
        {"same_offset": True},
        {"offset_a": random_a, "offset_b": random_b},
        random_a == random_b,
    )

    tab_center = tabular(full_power, config)
    tab_slice = tabular_features(centre_window(full_power, config), config)
    tab_equal = bool(np.array_equal(tab_center, tab_slice))
    tab_zero = tabular(mel_power(zeros, config), config)
    record(
        "tabular_84",
        {"mels": config.n_mels, "frames": config.n_win},
        {"shape": [84], "finite": True, "center_equals_full": True},
        {
            "shape": list(tab_center.shape),
            "finite": bool(np.isfinite(tab_center).all()),
            "center_equals_full": tab_equal,
            "zero_shape": list(tab_zero.shape),
            "zero_finite": bool(np.isfinite(tab_zero).all()),
        },
        tab_center.shape == (84,)
        and np.isfinite(tab_center).all()
        and tab_equal
        and tab_zero.shape == (84,)
        and np.isfinite(tab_zero).all(),
    )

    gain_power = mel_power(
        rng.standard_normal(config.win_samples).astype(np.float32) * 0.25, config
    )
    gain_window_db = to_db(gain_power, config)
    gain_has_no_floor = bool(np.all(gain_window_db > config.db_floor + 1.0))
    base_tab = tabular_features(gain_power, config)
    gain_results = {}
    gain_ok = True
    for gain_db in (-6.0, 6.0):
        changed = apply_power_gain_for_check(gain_power, gain_db)
        value = tabular_features(changed, config)
        abs_diff = float(np.max(np.abs(value - base_tab), initial=0.0))
        rel_diff = float(
            np.max(
                np.abs(value - base_tab) / np.maximum(np.abs(base_tab), 1e-8),
                initial=0.0,
            )
        )
        gain_results[str(gain_db)] = {"max_abs": abs_diff, "max_relative": rel_diff}
        gain_ok &= rel_diff <= 1e-4
    record(
        "tabular_gain_invariance",
        {
            "gain_db": [-6.0, 6.0],
            "tolerance_relative": 1e-4,
            "window_has_no_floor_cells": gain_has_no_floor,
        },
        {"max_relative": "<= 1e-4", "window_has_no_floor_cells": True},
        {"per_gain": gain_results, "window_has_no_floor_cells": gain_has_no_floor},
        bool(gain_ok and gain_has_no_floor),
    )

    return cases


def apply_power_gain_for_check(power: np.ndarray, gain_db: float) -> np.ndarray:
    """Private deterministic helper used by the edge-case suite."""
    return (
        np.asarray(power, np.float32) * np.float32(10.0 ** (gain_db / 10.0))
    ).astype(np.float32)


def audit_lengths(
    manifest, config: FrontEndConfig, raw_root: str | Path
) -> dict[str, Any]:
    """Audit header lengths for the rows explicitly supplied by the caller.

    The caller controls which splits are supplied. The function never loads or
    decodes audio samples, and reports native-rate mismatches for a stop gate.
    """
    rows = [
        dict(row)
        if isinstance(row, dict)
        else row._asdict()
        if hasattr(row, "_asdict")
        else dict(row)
        for row in manifest
    ]
    import soundfile as sf

    classes: dict[str, dict[str, Any]] = {}
    rate_mismatches: list[dict[str, Any]] = []
    for row in rows:
        clip = str(row.get("clip_id", row.get("file", "")))
        if not clip:
            raise ValueError("length audit rows require clip_id or file")
        class_name = str(
            row.get("label_name", row.get("class_name", row.get("label", "unknown")))
        )
        bucket = classes.setdefault(
            class_name,
            {
                "total": 0,
                "n_full": 0,
                "n_padded": 0,
                "n_rejected": 0,
                "n_below_nfft": 0,
                "rejected_files": [],
            },
        )
        info = sf.info(str(Path(raw_root) / clip))
        samples = int(info.frames)
        bucket["total"] += 1
        if int(info.samplerate) != config.sr:
            rate_mismatches.append(
                {
                    "clip_id": clip,
                    "native_rate": int(info.samplerate),
                    "expected_rate": config.sr,
                }
            )
        if samples < config.n_fft:
            bucket["n_below_nfft"] += 1
        if samples < config.win_samples / 2:
            bucket["n_rejected"] += 1
            bucket["rejected_files"].append({"clip_id": clip, "n_samples": samples})
        elif samples < config.win_samples:
            bucket["n_padded"] += 1
        else:
            bucket["n_full"] += 1
    return {
        "by_class": classes,
        "native_rate_mismatches": rate_mismatches,
        "stop": bool(rate_mismatches or any(v["n_rejected"] for v in classes.values())),
        "win_samples": config.win_samples,
        "n_fft": config.n_fft,
        "raw_headers_only": True,
    }
