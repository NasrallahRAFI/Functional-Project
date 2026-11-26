"""Verified local inference for the explicitly selected CNN V2/V3 package."""
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
    validation_eligible: bool = True
    experimental_activation: bool = False
    validation_guard_failures: tuple[str, ...] = ()
    test_access: bool = False
    policy_id: str = "threshold-majority-v1"
    run_id: str | None = None
    validation_metrics: dict | None = None
    final_test_passed: bool = False

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
            raise RuntimeError(f"invalid {self.model_id} output")
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
            "validation_eligible": self.validation_eligible,
            "experimental_activation": self.experimental_activation,
            "validation_guard_failures": list(self.validation_guard_failures),
            "test_access": self.test_access,
            "run_id": self.run_id,
            "validation_metrics": self.validation_metrics,
            "final_test_passed": self.final_test_passed,
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
    """Load and verify the explicitly selected V2 baseline or V3 package."""
    root = Path(project_root).resolve()
    manifest_file = Path(manifest_path) if manifest_path else root / "config/live_detector.json"
    manifest = _json(manifest_file)
    model_id = manifest.get("model_id")
    model_version = manifest.get("model_version")
    if (manifest.get("schema_version") != 2
            or model_id not in {"CNN_V2.0", "CNN_V3.0"}
            or model_version != {"CNN_V2.0": "2.0", "CNN_V3.0": "3.0"}.get(model_id)):
        raise ValueError("live detector must explicitly select CNN V2.0 or CNN V3.0")
    if model_id == "CNN_V3.0":
        actual_failures = set(manifest.get("validation_guard_failures", []))
        eligible = manifest.get("validation_eligible") is True
        experimental = manifest.get("experimental_activation") is True
        final_test_passed = manifest.get("final_test_passed") is True
        if ((eligible and actual_failures) or (not eligible and not actual_failures)
                or (not experimental and not (eligible and final_test_passed))
                or manifest.get("test_access") is not False):
            raise ValueError("CNN V3.0 selection must preserve its validation result, experimental status, and training test-access record")
    if manifest.get("score_kind") != "raw_logit" or manifest.get("variant") != "depth_4":
        raise ValueError(f"unsupported {model_id} score space or architecture")

    try:
        package = _inside(root, manifest["package_path"], label="model package")
    except ValueError:
        # The workspace registry binds relocated releases to the exact selected
        # package. Individual package-member boundaries remain unchanged.
        from .workspace import resolve_registered_package
        package = resolve_registered_package(root, manifest["package_path"], manifest_file)
    if not package.is_dir():
        from .workspace import resolve_registered_package
        package = resolve_registered_package(root, manifest["package_path"], manifest_file)
    package_manifest_path = package / "PACKAGE_MANIFEST.json"
    package_manifest = _json(package_manifest_path)
    members = package_manifest.get("files")
    if not isinstance(members, list) or not members:
        raise ValueError(f"{model_id} package manifest is empty or invalid")
    if package_manifest.get("model_id", model_id) != model_id:
        raise ValueError("package manifest model identity differs from live selection")
    if model_id == "CNN_V3.0" and (
        package_manifest.get("validation_eligible") != manifest.get("validation_eligible")
        or package_manifest.get("experimental_activation") != manifest.get("experimental_activation")
        or package_manifest.get("test_access") is not False
        or set(package_manifest.get("validation_guard_failures", [])) != actual_failures
    ):
        raise ValueError("CNN V3.0 package status differs from the live selection")
    for entry in members:
        relative = str(entry.get("path", ""))
        path = _inside(package, relative, label="package member")
        if not path.is_file() or path.stat().st_size != int(entry["size"]) or _sha256(path) != entry["sha256"]:
            raise ValueError(f"{model_id} package checksum mismatch: {relative}")

    metadata_path = package / "model_metadata.json"
    metadata = _json(metadata_path)
    model_path = _inside(package, metadata["model_file"], label="model")
    model_digest = _sha256(model_path)
    if model_digest != metadata["model_sha256"] or model_digest != manifest["model_sha256"]:
        raise ValueError(f"{model_id} model digest differs from metadata or live selection")
    if metadata.get("package") != model_id.replace("_", " ") or metadata.get("parameter_count") != 60641:
        raise ValueError(f"{model_id} package metadata does not describe the selected model")
    if metadata.get("architecture", {}).get("channels") != [16, 32, 64, 64]:
        raise ValueError(f"{model_id} channel layout differs from the selected depth_4 model")
    if manifest.get("threshold") != metadata.get("validation_threshold"):
        raise ValueError("live threshold differs from the package validation threshold")
    if manifest.get("seed") != metadata.get("training_seed"):
        raise ValueError(f"live seed differs from {model_id} training metadata")
    if metadata.get("stage") != manifest.get("stage", metadata.get("stage")):
        raise ValueError("live stage differs from package metadata")
    if model_id == "CNN_V3.0":
        if (metadata.get("validation_eligible") != manifest.get("validation_eligible")
                or metadata.get("experimental_activation") != manifest.get("experimental_activation")
                or metadata.get("test_access") is not False
                or set(metadata.get("validation_guard_failures", [])) != actual_failures
                or metadata.get("final_test_passed", False) != final_test_passed):
            raise ValueError("CNN V3.0 metadata does not preserve the live selection status")

    params = _json(package / metadata["frontend_config_file"])
    params.pop("_identity_schema", None)
    params["gain_range_db"] = tuple(params["gain_range_db"])
    config = FrontEndConfig(**params)
    config.validate()
    if config.n_mels != 64 or config.n_win != 28 or config.win_samples != 7936:
        raise ValueError(f"{model_id} frontend dimensions differ from the selected input contract")
    if (manifest.get("policy_id") != "threshold-majority-v1"
            or int(manifest.get("smoothing_windows", 0)) != config.smoothing_windows
            or float(manifest.get("interval_seconds", -1)) != config.decision_interval_s):
        raise ValueError(f"live decision settings differ from the {model_id} runtime frontend")
    stats_path = package / metadata["normalization_file"]
    with np.load(stats_path, allow_pickle=False) as archive:
        if set(archive.files) != {"mean", "std"}:
            raise ValueError(f"{model_id} normalization package must contain mean and std only")
        stats = {key: archive[key].astype(np.float32, copy=False) for key in ("mean", "std")}
    if any(value.shape != (64,) or not np.isfinite(value).all() for value in stats.values()):
        raise ValueError(f"{model_id} normalization vectors are malformed")
    if np.any(stats["std"] <= 0):
        raise ValueError(f"{model_id} normalization standard deviations must be positive")
    frontend_check = verify_bundle(package, config)
    if frontend_check.get("hash") != manifest.get("frontend_hash"):
        raise ValueError(f"{model_id} frontend hash differs from the live model selection")
    if any(not value.get("verified") for value in frontend_check["normalized_outputs"].values()):
        raise ValueError(f"{model_id} runtime frontend goldens failed verification")

    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()
    if not hasattr(model, "parameters") or sum(int(p.numel()) for p in model.parameters()) != 60641:
        raise ValueError(f"loaded {model_id} TorchScript model does not contain 60,641 parameters")
    threshold = float(metadata["validation_threshold"])
    if not np.isfinite(threshold):
        raise ValueError(f"{model_id} validation threshold is not finite")
    detector = LoadedDetector(
        config=config,
        stats=stats,
        model=model,
        threshold=threshold,
        seed=int(metadata["training_seed"]),
        artifact=str(Path(manifest["package_path"]) / metadata["model_file"]),
        artifact_sha256=model_digest,
        model_id=model_id,
        model_version=model_version,
        stage=str(metadata.get("stage", "V2")),
        variant="depth_4",
        validation_eligible=bool(manifest.get("validation_eligible", True)),
        experimental_activation=bool(manifest.get("experimental_activation", False)),
        validation_guard_failures=tuple(manifest.get("validation_guard_failures", [])),
        test_access=bool(manifest.get("test_access", False)),
        run_id=metadata.get("run_id"),
        validation_metrics=metadata.get("validation_metrics"),
        final_test_passed=bool(metadata.get("final_test_passed", False)),
    )
    detector.score_window(np.zeros(config.win_samples, dtype=np.float32))
    return detector
