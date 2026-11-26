"""Stable identity for the V4 live frontend bundle served by the detector."""
from __future__ import annotations

import hashlib
from pathlib import Path

FRONTEND_FILES = (
    "index.html",
    "app.js",
    "styles.css",
    "audio-capture-worklet.js",
)


def frontend_sha256(live_dir: str | Path) -> str:
    """Hash relative names and bytes so stale or incomplete bundles differ."""
    root = Path(live_dir)
    contents = {}
    for name in FRONTEND_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"live frontend file is missing: {path}")
        contents[name] = path.read_bytes()
    return frontend_sha256_from_sources(contents)


def frontend_sha256_from_sources(contents: dict[str, bytes]) -> str:
    """Hash an ordered name-to-bytes map from disk or an HTTP-served bundle."""
    digest = hashlib.sha256()
    for name in FRONTEND_FILES:
        if name not in contents:
            raise FileNotFoundError(f"live frontend source is missing: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(contents[name])
        digest.update(b"\0")
    return digest.hexdigest()
