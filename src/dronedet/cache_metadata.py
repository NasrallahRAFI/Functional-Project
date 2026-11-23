"""Cache provenance sidecars for split power and tabular arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
