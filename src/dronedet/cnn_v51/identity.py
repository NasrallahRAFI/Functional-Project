"""Canonical source and clip identities defined by the V5.1 plan."""

from __future__ import annotations

import hashlib
import unicodedata

import numpy as np


def canonical_relative_path(path: str) -> str:
    if not isinstance(path, str) or not path:
        raise ValueError("relative path must be a non-empty string")
    path = unicodedata.normalize("NFC", path.replace("\\", "/"))
    if path.startswith("/") or any(part in {"", ".", ".."} for part in path.split("/")):
        raise ValueError("path must be canonical relative form without traversal")
    return path


def source_file_id(source_revision: str, relative_path: str) -> str:
    if not source_revision or "\0" in source_revision:
        raise ValueError("invalid source revision")
    rel = canonical_relative_path(relative_path)
    return hashlib.sha256(f"{source_revision}\0{rel}".encode("utf-8")).hexdigest()


def clip_id(source: str, file_id: str, start_sample_16k: int, n_samples: int) -> str:
    if not source or "__" in source:
        raise ValueError("source code must be non-empty and must not contain '__'")
    if len(file_id) != 64 or any(char not in "0123456789abcdef" for char in file_id):
        raise ValueError("file_id must be lowercase SHA-256 hex")
    for name, value in (("start_sample_16k", start_sample_16k), ("n_samples", n_samples)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    return f"{source}__{file_id}__s{start_sample_16k:012d}__n{n_samples:08d}"


def safe_legacy_filename(legacy_clip_id: str) -> str:
    return hashlib.sha256(legacy_clip_id.encode("utf-8")).hexdigest()


def standardized_sample_sha256(samples: np.ndarray) -> str:
    """Hash the canonical finite float32 little-endian sample identity."""
    wave = np.asarray(samples, dtype=np.float32)
    if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
        raise ValueError("sample identity requires finite nonempty mono samples")
    payload = np.ascontiguousarray(wave, dtype="<f4").tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()
