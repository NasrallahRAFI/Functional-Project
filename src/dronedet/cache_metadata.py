"""Cache provenance sidecars for split power and tabular arrays."""

from __future__ import annotations

import json
import hashlib
import os
import platform
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

CACHE_SCHEMA_VERSION = 2


def sidecar_path(cache_path: str | Path) -> Path:
    return Path(f"{Path(cache_path)}.meta.json")


def verify_cache_sidecar(cache_path: str | Path, *, frontend_params_hash: str,
                         train_manifest_sha256: str) -> dict[str, Any]:
    """Require current parameter and train-manifest provenance before loading a cache."""
    metadata_path = sidecar_path(cache_path)
    if not metadata_path.is_file():
        raise ValueError(f"cache provenance sidecar missing: {metadata_path.name}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cache provenance sidecar is unreadable: {metadata_path.name}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("cache provenance sidecar must contain an object")
    if "schema_version" in metadata and metadata["schema_version"] != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported cache metadata schema version")
    if metadata.get("frontend_params_hash") != frontend_params_hash:
        raise ValueError(f"cache parameter hash mismatch: {metadata_path.name}")
    if metadata.get("train_manifest_sha256") != train_manifest_sha256:
        raise ValueError(f"cache train-manifest hash mismatch: {metadata_path.name}")
    return metadata


def verify_array_metadata(metadata: dict[str, Any], *, shape: tuple[int, ...], dtype: Any) -> None:
    """Reject a cache whose array header differs from the values recorded at stamping."""
    if metadata.get("array_shape") != list(shape):
        raise ValueError("cache array shape differs from provenance sidecar")
    if metadata.get("array_dtype") != str(dtype):
        raise ValueError("cache array dtype differs from provenance sidecar")


def file_sha256(path: str | Path) -> str:
    """Hash in bounded blocks, including the array header and every payload byte."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_power_values(power: np.ndarray) -> None:
    """Validate large power caches without allocating a full-size boolean array."""
    if power.ndim != 2:
        raise ValueError("mel-power cache must be two-dimensional")
    for start in range(0, power.shape[1], 8192):
        block = power[:, start:start + 8192]
        if not np.isfinite(block).all() or np.any(block < 0):
            raise ValueError("mel-power cache contains non-finite or negative values")


def validate_manifest_index(manifest, index, shape: tuple[int, ...], config, *, split: str) -> None:
    """Reconcile ordered clip identities, contiguous spans, offsets and flags."""
    required = {"clip_id", "start_frame", "n_frames", "win_start_frame", "stats_eligible",
                "silent", "near_silent", "crop_all_zero", "length_status"}
    if not required.issubset(index.columns) or "clip_id" not in manifest.columns:
        raise ValueError("cache index or manifest is missing required columns")
    if not len(index) or len(manifest) != len(index):
        raise ValueError("cache manifest/index row counts differ or are empty")
    if manifest.clip_id.duplicated().any() or index.clip_id.duplicated().any():
        raise ValueError("cache clip identifiers must be unique")
    if not np.array_equal(manifest.clip_id.to_numpy(), index.clip_id.to_numpy()):
        raise ValueError("cache manifest/index clip order differs")
    if "split" in manifest and not manifest["split"].eq(split).all():
        raise ValueError("cache manifest contains a different split")
    for name in ("start_frame", "n_frames", "win_start_frame"):
        if index[name].dtype.kind not in "iu":
            raise ValueError(f"cache index {name} must contain integers")
    starts = index.start_frame.to_numpy(dtype=np.int64)
    counts = index.n_frames.to_numpy(dtype=np.int64)
    offsets = index.win_start_frame.to_numpy(dtype=np.int64)
    if len(shape) != 2 or shape[0] != config.n_mels:
        raise ValueError("mel-power metadata has invalid dimensions")
    if np.any(counts < config.n_win) or starts[0] != 0 or np.any(starts < 0):
        raise ValueError("cache index frame span is invalid")
    if not np.array_equal(starts[1:], starts[:-1] + counts[:-1]):
        raise ValueError("cache index spans must be contiguous and non-overlapping")
    if int(starts[-1]) + int(counts[-1]) != shape[1]:
        raise ValueError("cache index does not cover the power array")
    if not np.array_equal(offsets, (counts - config.n_win) // 2):
        raise ValueError("cache centered offsets differ from the canonical rule")
    for name in ("stats_eligible", "silent", "near_silent", "crop_all_zero"):
        if index[name].dtype.kind != "b":
            raise ValueError(f"cache index {name} must contain booleans")
    excluded = index.silent | index.near_silent | index.crop_all_zero
    if not np.array_equal(index.stats_eligible.to_numpy(), (~excluded).to_numpy()):
        raise ValueError("cache eligibility contradicts exclusion flags")
    if "stats_exclude" in index and not np.array_equal(index.stats_exclude.to_numpy(), excluded.to_numpy()):
        raise ValueError("cache stats_exclude contradicts eligibility flags")
    for name in ("silent", "near_silent"):
        if name in manifest and not np.array_equal(manifest[name].to_numpy(), index[name].to_numpy()):
            raise ValueError(f"cache index flag differs from manifest: {name}")


def verify_transaction(cache_path: Path, metadata: dict, *, spec_dir: Path, split: str) -> None:
    """A v2 cache is visible only after all artifacts and sidecars are committed."""
    marker_path = spec_dir / f"cache_{split}.commit.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cache transaction is missing or unreadable") from exc
    if not isinstance(marker, dict):
        raise ValueError("cache transaction must contain an object")
    if marker.get("transaction_id") != metadata.get("transaction_id"):
        raise ValueError("cache transaction identity mismatch")
    if marker.get("artifact_sha256", {}).get(cache_path.name) != metadata.get("artifact_sha256"):
        raise ValueError("cache transaction artifact digest mismatch")
    required_paths = marker.get("required_paths")
    if (marker.get("schema_version") != CACHE_SCHEMA_VERSION or not isinstance(required_paths, list)
            or len(required_paths) != 5 or any(not isinstance(path, str) for path in required_paths)):
        raise ValueError("cache transaction has an invalid schema or artifact list")
    for required in required_paths:
        path = Path(required)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("cache transaction contains an unsafe path")
        # Paths are relative to data/processed, shared by spec and tabular folders.
        if not (spec_dir.parent / path).is_file():
            raise ValueError(f"cache transaction is incomplete: {required}")


def publish_split_cache(cache: np.ndarray, index, features: np.ndarray, manifest, *,
                        split: str, paths: dict, config,
                        logged_override: str | None = None,
                        allow_test: bool = False) -> dict[str, Path]:
    """Publish one validated array/index/tabular/sidecar set, then its commit marker.

    Existing files are never replaced. Failed writes remove only files created
    by this invocation; the commit marker is linked last.
    """
    from .config import cache_content_hash, params_hash
    from .features import feature_names
    from .splits import guard_test_access

    guard_test_access(split, allow_test=allow_test)
    if split not in {"train", "val", "test"}:
        raise ValueError("unknown cache publication split")
    config.validate_cache_write(logged_override=logged_override)
    validate_manifest_index(manifest, index, cache.shape, config, split=split)
    validate_power_values(cache)
    if cache.dtype != np.float32 or features.dtype != np.float32:
        raise ValueError("cache arrays must use float32")
    if features.shape != (len(manifest), 84) or not np.isfinite(features).all():
        raise ValueError("tabular cache must contain 84 finite features per clip")
    expected_columns = {"label", "group", "partial_zero", "clipped_flag", "dc_flag"}
    if not expected_columns.issubset(manifest.columns):
        raise ValueError("cache manifest lacks tabular labels/group/flags")
    if not manifest.label.isin([0, 1]).all():
        raise ValueError("cache labels must be binary")
    spec, tab = Path(paths["spec"]), Path(paths["tab"])
    if spec.parent.resolve() != tab.parent.resolve():
        raise ValueError("cache folders must share a processed-data parent")
    manifest_path = Path(paths["splits"]) / f"{split}.csv"
    train_path = Path(paths["splits"]) / "train.csv"
    # Re-read manifests before writing; caller-supplied rows must match their provenance.
    import pandas as pd
    frozen_rows = pd.read_csv(manifest_path, keep_default_na=False)
    if not manifest.reset_index(drop=True).equals(frozen_rows.reset_index(drop=True)):
        raise ValueError("cache rows differ from the manifest file")
    from .config import load_project_config
    raw_roots = {load_project_config(Path(paths["root"]) / "config.yaml")["P"]["raw_root"]}
    if paths.get("raw_root") is not None:
        raw_roots.add(Path(paths["raw_root"]))
    for raw_root in raw_roots:
        for destination in (spec, tab):
            resolved, raw = destination.resolve(), Path(raw_root).resolve()
            if resolved == raw or raw in resolved.parents:
                raise PermissionError("refusing to write caches under the raw dataset")
    files = {"power": spec / f"mel_power_{split}.npy",
             "index": spec / f"index_{split}.csv", "tabular": tab / f"{split}.npz"}
    files.update(power_meta=sidecar_path(files["power"]), tabular_meta=sidecar_path(files["tabular"]),
                 commit=spec / f"cache_{split}.commit.json")
    if any(path.exists() for path in files.values()):
        raise FileExistsError("refusing to overwrite an existing cache artifact")
    for directory in (spec, tab):
        directory.mkdir(parents=True, exist_ok=True)
    staged, published = {}, []

    def stage(key, writer):
        with tempfile.NamedTemporaryFile(dir=files[key].parent, suffix=".tmp", delete=False) as stream:
            staged[key] = Path(stream.name)
            writer(stream)

    try:
        stage("power", lambda stream: np.save(stream, cache, allow_pickle=False))
        stage("index", lambda stream: stream.write(index.to_csv(index=False, lineterminator="\n").encode("utf-8")))
        archive = {"X": features, "y": manifest.label.to_numpy(np.int8),
                   "clip_id": manifest.clip_id.to_numpy(str), "group": manifest.group.to_numpy(str),
                   "feature_names": np.asarray(feature_names())}
        archive.update({name: index[name].to_numpy(bool) for name in ("silent", "near_silent", "crop_all_zero")})
        archive.update({name: manifest[name].to_numpy(bool) for name in ("partial_zero", "clipped_flag", "dc_flag")})
        stage("tabular", lambda stream: np.savez_compressed(stream, **archive))
        transaction = uuid.uuid4().hex
        common = {"schema_version": CACHE_SCHEMA_VERSION, "transaction_id": transaction,
                  "frontend_params_hash": params_hash(config), "power_content_hash": cache_content_hash(config),
                  "train_manifest_sha256": file_sha256(train_path),
                  "split_manifest_sha256": file_sha256(manifest_path),
                  "index_sha256": file_sha256(staged["index"]),
                  "power_shape": list(cache.shape), "fmax": config.fmax,
                  "measured_passband_hz": config.measured_passband_hz,
                  "passband_status": config.passband_status,
                  "passband_override_reason": (logged_override or config.passband_override_reason)
                                                if config.passband_status != "measured" else None,
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "platform": platform.platform(), "provenance": "built_by_pipeline",
                  "class_counts": {str(key): int(value) for key, value in manifest.label.value_counts().items()},
                  "length_audit_counts": {str(key): int(value) for key, value in index.length_status.value_counts().items()},
                  "stats_eligible_clips": int(index.stats_eligible.sum()),
                  "stats_excluded_clips": int((~index.stats_eligible).sum()),
                  "feature_catalog_sha256": hashlib.sha256(json.dumps(feature_names()).encode()).hexdigest(),
                  "builder_source_sha256": {"cache_metadata.py": file_sha256(Path(__file__)),
                                            "features.py": file_sha256(Path(__file__).with_name("features.py"))},
                  "library_versions": {name: version(name) for name in ("numpy", "scipy", "librosa", "soxr", "soundfile")}}
        repo = Path(__file__).resolve().parents[2]
        common["git_commit"] = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        common["git_dirty"] = bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip())
        for key, array in (("power", cache), ("tabular", features)):
            metadata = {**common, "cache_file": files[key].name, "array_shape": list(array.shape),
                        "array_dtype": str(array.dtype), "artifact_sha256": file_sha256(staged[key])}
            stage(key + "_meta", lambda stream, value=metadata: stream.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")))
        marker = {"schema_version": CACHE_SCHEMA_VERSION, "transaction_id": transaction,
                  "artifact_sha256": {files[key].name: file_sha256(staged[key]) for key in ("power", "index", "tabular")},
                  "required_paths": [str(files[key].relative_to(spec.parent)).replace("\\", "/")
                                     for key in ("power", "index", "tabular", "power_meta", "tabular_meta")]}
        stage("commit", lambda stream: stream.write((json.dumps(marker, indent=2) + "\n").encode("utf-8")))
        for key in ("power", "index", "tabular", "power_meta", "tabular_meta", "commit"):
            os.link(staged[key], files[key])
            published.append(files[key])
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)
    return files
