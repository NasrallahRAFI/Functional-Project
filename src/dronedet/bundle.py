"""Front-end-only bundle integrity checks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

from .config import FrontEndConfig, config_hash
from .features import mel_power, to_db


def write_bundle(root: str | Path, config: FrontEndConfig, stats: dict[str, Any],
                 golden_audio: dict[str, np.ndarray]) -> Path:
    root = Path(root)
    if root.exists() and any(root.iterdir()): raise FileExistsError("bundle directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    parameters = {k: v for k, v in config.__dict__.items()}
    (root / "config.json").write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf8")
    (root / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf8")
    digest = config_hash(config, stats)
    (root / "hash.txt").write_text(digest + "\n", encoding="ascii")
    from importlib.metadata import PackageNotFoundError, version
    names = ("numpy", "scipy", "librosa", "soxr", "soundfile", "scikit-learn", "torch", "PyYAML")
    libraries = {}
    for name in names:
        try: libraries[name] = version(name)
        except PackageNotFoundError: libraries[name] = None
    libraries.update(model_weights=None, validation_threshold=None)
    (root / "libraries.json").write_text(json.dumps(libraries, indent=2)+"\n",encoding="utf8")
    (root / "golden").mkdir()
    for name, audio in golden_audio.items():
        np.save(root / "golden" / f"{name}_audio.npy", np.asarray(audio, np.float32), allow_pickle=False)
        np.save(root / "golden" / f"{name}_db.npy", to_db(mel_power(audio, config), config), allow_pickle=False)
    return root


def verify_bundle(root: str | Path, running_config: FrontEndConfig) -> dict[str, Any]:
    root = Path(root)
    params = json.loads((root / "config.json").read_text(encoding="utf8"))
    stats = json.loads((root / "stats.json").read_text(encoding="utf8"))
    bundled = FrontEndConfig(**params)
    stored = (root / "hash.txt").read_text(encoding="ascii").strip()
    actual = config_hash(bundled, stats)
    if stored != actual: raise ValueError("bundle configuration/statistics hash mismatch")
    if config_hash(running_config, stats) != stored: raise ValueError("bundle does not match running front end")
    count = 0
    for audio_path in sorted((root / "golden").glob("*_audio.npy")):
        name = audio_path.name.removesuffix("_audio.npy")
        audio = np.load(audio_path, allow_pickle=False)
        expected = np.load(root / "golden" / f"{name}_db.npy", allow_pickle=False)
        actual_db = to_db(mel_power(audio, running_config), running_config)
        if actual_db.shape != expected.shape or np.max(np.abs(actual_db - expected), initial=0) > 1e-3:
            raise ValueError(f"golden mismatch: {name}")
        count += 1
    if not count: raise ValueError("bundle has no golden arrays")
    return {"hash": stored, "golden_files_verified": count}
