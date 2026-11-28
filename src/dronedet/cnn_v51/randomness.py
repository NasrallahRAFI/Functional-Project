"""Stateless, namespaced deterministic draws from the V5.1 contract."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from typing import Any


def _validate_key(value: Any) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, float):
        raise TypeError("random keys accept only NFC strings, integers and nested arrays")
    if isinstance(value, str):
        if unicodedata.normalize("NFC", value) != value:
            raise ValueError("random-key strings must already be Unicode NFC")
        return
    if isinstance(value, int):
        return
    if isinstance(value, list):
        for item in value:
            _validate_key(item)
        return
    raise TypeError(f"unsupported random-key component: {type(value).__name__}")


def canonical_key(key: list[Any]) -> str:
    if not isinstance(key, list):
        raise TypeError("random key must be a JSON array")
    _validate_key(key)
    return json.dumps(key, ensure_ascii=False, separators=(",", ":"))


def keyed_sha1(purpose: str, key: list[Any]) -> str:
    if not isinstance(purpose, str) or not purpose:
        raise ValueError("purpose must be a non-empty string")
    if unicodedata.normalize("NFC", purpose) != purpose:
        raise ValueError("purpose must be Unicode NFC")
    message = f"51:{purpose}:{canonical_key(key)}".encode("utf-8")
    return hashlib.sha1(message).hexdigest()


def uniform(purpose: str, key: list[Any]) -> float:
    prefix = int(keyed_sha1(purpose, key)[:13], 16)
    return (prefix + 0.5) / (2**52)


def uniform_range(purpose: str, key: list[Any], low: float, high: float) -> float:
    if not (math.isfinite(low) and math.isfinite(high) and low <= high):
        raise ValueError("uniform bounds must be finite and ordered")
    return low + (high - low) * uniform(purpose, key)


def integer(purpose: str, key: list[Any], n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("integer draw size must be a positive integer")
    return math.floor(n * uniform(purpose, key))


def ranked(purpose: str, keyed_items: list[tuple[list[Any], Any]]) -> list[Any]:
    """Sort finite candidates by full digest, then canonical key."""
    return [item for key, item in sorted(
        keyed_items, key=lambda pair: (keyed_sha1(purpose, pair[0]), canonical_key(pair[0]))
    )]


def categorical(purpose: str, key: list[Any], candidates: list[tuple[str, float]]) -> str:
    if not candidates:
        raise ValueError("categorical draw needs candidates")
    if [name for name, _ in candidates] != sorted(name for name, _ in candidates):
        raise ValueError("categorical candidates must be in canonical ID order")
    probs = [float(p) for _, p in candidates]
    if any(not math.isfinite(p) or p < 0 for p in probs) or sum(probs) <= 0:
        raise ValueError("categorical probabilities must be finite, nonnegative and nonzero")
    total = math.fsum(probs)
    cumulative = 0.0
    draw = uniform(purpose, key)
    for index, ((name, _), probability) in enumerate(zip(candidates, probs)):
        cumulative = 1.0 if index == len(probs) - 1 else cumulative + probability / total
        if draw < cumulative:
            return name
    raise AssertionError("final categorical cumulative probability must be 1")
