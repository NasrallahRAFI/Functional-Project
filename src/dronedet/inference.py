"""Verified inference for the explicitly selected CNN V2.0 frontend package."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from .bundle import verify_bundle
from .config import FrontEndConfig
from .features import mel_power, transform


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, relative: str, *, label: str) -> Path:
    posix = PurePosixPath(relative.replace("\\", "/"))
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"{label} path must remain inside the project package")
    path = (root / Path(*posix.parts)).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"{label} path escapes the project root")
    return path


@dataclass(frozen=True)
class LoadedDetector:
    config: FrontEndConfig
    stats: dict[str, np.ndarray]
    model: torch.nn.Module
    threshold: float
    seed: int
    artifact: str
    artifact_sha256: str
    model_id: str
    model_version: str
    stage: str
    variant: str
    policy_id: str = "threshold-majority-v1"

    def score_window(self, window: np.ndarray) -> float:
        audio = np.asarray(window, dtype=np.float32)
        if audio.shape != (self.config.win_samples,) or not np.isfinite(audio).all():
            raise ValueError(f"window must be finite mono float32 with {self.config.win_samples} samples")
        features = transform(mel_power(audio, self.config), self.config, self.stats)
        if features.shape != (1, self.config.n_mels, self.config.n_win):
            raise ValueError(f"unexpected frontend tensor shape: {features.shape}")
        if features.dtype != np.float32 or not np.isfinite(features).all():
            raise ValueError("invalid frontend tensor")
        batch = torch.from_numpy(np.ascontiguousarray(features)).unsqueeze(0)
        with torch.inference_mode():
            output = self.model(batch)
        if output.shape != (1,) or not torch.isfinite(output).all():
            raise RuntimeError("invalid CNN V2.0 output")
        return float(output.item())

    def health_metadata(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "stage": self.stage,
            "variant": self.variant,
            "seed": self.seed,
            "checkpoint_sha256": self.artifact_sha256,
            "threshold": self.threshold,
            "score_kind": "raw_logit",
            "policy_id": self.policy_id,
            "smoothing_windows": self.config.smoothing_windows,
            "window_samples": self.config.win_samples,
            "cadence_seconds": round(
                self.config.hop
                * round(self.config.decision_interval_s * self.config.sr / self.config.hop)
                / self.config.sr,
                6,
            ),
        }


def load_detector(project_root: str | Path, manifest_path: str | Path | None = None) -> LoadedDetector:
    """Load and verify the selected V2 package; never fall back to a V1 model."""
    root = Path(project_root).resolve()
    manifest_file = Path(manifest_path) if manifest_path else root / "config/live_detector.json"
    manifest = _json(manifest_file)
    if (manifest.get("schema_version") != 2 or manifest.get("model_id") != "CNN_V2.0"
            or manifest.get("model_version") != "2.0"):
        raise ValueError("live detector must explicitly select CNN V2.0")
    if manifest.get("score_kind") != "raw_logit" or manifest.get("variant") != "depth_4":
        raise ValueError("unsupported CNN V2.0 score space or architecture")

    package = _inside(root, manifest["package_path"], label="model package")
    package_manifest_path = package / "PACKAGE_MANIFEST.json"
    package_manifest = _json(package_manifest_path)
    members = package_manifest.get("files")
    if not isinstance(members, list) or not members:
        raise ValueError("CNN V2.0 package manifest is empty or invalid")
    for entry in members:
        relative = str(entry.get("path", ""))
        path = _inside(package, relative, label="package member")
        if not path.is_file() or path.stat().st_size != int(entry["size"]) or _sha256(path) != entry["sha256"]:
            raise ValueError(f"CNN V2.0 package checksum mismatch: {relative}")

    metadata_path = package / "model_metadata.json"
    metadata = _json(metadata_path)
    model_path = _inside(package, metadata["model_file"], label="model")
    model_digest = _sha256(model_path)
    if model_digest != metadata["model_sha256"] or model_digest != manifest["model_sha256"]:
        raise ValueError("CNN V2.0 model digest differs from metadata or live selection")
    if metadata.get("package") != "CNN V2.0" or metadata.get("parameter_count") != 60641:
        raise ValueError("CNN V2.0 package metadata does not describe the selected model")
    if metadata.get("architecture", {}).get("channels") != [16, 32, 64, 64]:
        raise ValueError("CNN V2.0 channel layout differs from the selected depth_4 model")
    if manifest.get("threshold") != metadata.get("validation_threshold"):
        raise ValueError("live threshold differs from the validation-frozen V2 threshold")
    if manifest.get("seed") != metadata.get("training_seed"):
        raise ValueError("live seed differs from CNN V2.0 training metadata")

    params = _json(package / metadata["frontend_config_file"])
    params.pop("_identity_schema", None)
    params["gain_range_db"] = tuple(params["gain_range_db"])
    config = FrontEndConfig(**params)
    config.validate()
    if config.n_mels != 64 or config.n_win != 28 or config.win_samples != 7936:
        raise ValueError("CNN V2.0 frontend dimensions differ from the selected input contract")
    if (manifest.get("policy_id") != "threshold-majority-v1"
            or int(manifest.get("smoothing_windows", 0)) != config.smoothing_windows
            or float(manifest.get("interval_seconds", -1)) != config.decision_interval_s):
        raise ValueError("live decision settings differ from the CNN V2.0 runtime frontend")
    stats_path = package / metadata["normalization_file"]
    with np.load(stats_path, allow_pickle=False) as archive:
        if set(archive.files) != {"mean", "std"}:
            raise ValueError("CNN V2.0 normalization package must contain mean and std only")
        stats = {key: archive[key].astype(np.float32, copy=False) for key in ("mean", "std")}
    if any(value.shape != (64,) or not np.isfinite(value).all() for value in stats.values()):
        raise ValueError("CNN V2.0 normalization vectors are malformed")
    if np.any(stats["std"] <= 0):
        raise ValueError("CNN V2.0 normalization standard deviations must be positive")
    frontend_check = verify_bundle(package, config)
    if frontend_check.get("hash") != manifest.get("frontend_hash"):
        raise ValueError("CNN V2.0 frontend hash differs from the live model selection")
    if any(not value.get("verified") for value in frontend_check["normalized_outputs"].values()):
        raise ValueError("CNN V2.0 runtime frontend goldens failed verification")

    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()
    if not hasattr(model, "parameters") or sum(int(p.numel()) for p in model.parameters()) != 60641:
        raise ValueError("loaded TorchScript model does not contain 60,641 parameters")
    threshold = float(metadata["validation_threshold"])
    if not np.isfinite(threshold):
        raise ValueError("CNN V2.0 validation threshold is not finite")
    detector = LoadedDetector(
        config=config,
        stats=stats,
        model=model,
        threshold=threshold,
        seed=int(metadata["training_seed"]),
        artifact=str(model_path.relative_to(root)),
        artifact_sha256=model_digest,
        model_id="CNN_V2.0",
        model_version="2.0",
        stage="V2",
        variant="depth_4",
    )
    detector.score_window(np.zeros(config.win_samples, dtype=np.float32))
    return detector
