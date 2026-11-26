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
    threshold: float | None
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
            "policy_id": self.policy_id,
            "smoothing_windows": 0 if self.threshold is None else self.config.smoothing_windows,
            "window_samples": self.config.win_samples,
            "cadence_seconds": round(
                self.config.hop
                * round(self.config.decision_interval_s * self.config.sr / self.config.hop)
                / self.config.sr,
                6,
            ),
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "threshold_selected": self.threshold is not None,
            "run_id": "cnn_a0_seed42" if self.model_id == "CNN_V4.0" else None,
        }


def load_detector(project_root: str | Path, manifest_path: str | Path | None = None) -> LoadedDetector:
    """Load and verify the explicitly selected V2 baseline or V3 package."""
    root = Path(project_root).resolve()
    manifest_file = Path(manifest_path) if manifest_path else root / "config/live_detector.json"
    manifest = _json(manifest_file)
    if manifest.get("model_id") == "CNN_V4.0":
        return _load_v4(root, manifest_file, manifest)
    model_id = manifest.get("model_id")
    model_version = manifest.get("model_version")
    if (manifest.get("schema_version") != 2
            or model_id not in {"CNN_V2.0", "CNN_V3.0"}
            or model_version != {"CNN_V2.0": "2.0", "CNN_V3.0": "3.0"}.get(model_id)):
        raise ValueError("live detector must explicitly select CNN V2.0 or CNN V3.0")
    if model_id == "CNN_V3.0":
        expected_failures = {
            "clean_recall_at_least_159_of_185",
            "clean_false_positives_at_most_5_of_1325",
            "each_augmented_far_at_most_1pct",
        }
        actual_failures = set(manifest.get("validation_guard_failures", []))
        if (manifest.get("validation_eligible") is not False
                or manifest.get("experimental_activation") is not True
                or manifest.get("test_access") is not False
                or actual_failures != expected_failures):
            raise ValueError("CNN V3.0 requires an explicit experimental selection with its failed guards and locked test state recorded")
    if manifest.get("score_kind") != "raw_logit" or manifest.get("variant") != "depth_4":
        raise ValueError(f"unsupported {model_id} score space or architecture")

    try:
        package = _inside(root, manifest["package_path"], label="model package")
    except ValueError:
        # The workspace registry binds relocated releases to the exact selected
        # package. Individual package-member boundaries remain unchanged.
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
        package_manifest.get("validation_eligible") is not False
        or package_manifest.get("experimental_activation") is not True
        or package_manifest.get("test_access") is not False
        or set(package_manifest.get("validation_guard_failures", [])) != expected_failures
    ):
        raise ValueError("CNN V3.0 package does not bind its experimental status and failed guards")
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
        if (metadata.get("validation_eligible") is not False
                or metadata.get("experimental_activation") is not True
                or metadata.get("test_access") is not False
                or set(metadata.get("validation_guard_failures", [])) != expected_failures
                or set(metadata.get("validation_guard_failures", [])) != actual_failures):
            raise ValueError("CNN V3.0 metadata does not preserve its validation-ineligible experimental status")

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
    )
    detector.score_window(np.zeros(config.win_samples, dtype=np.float32))
    return detector


def _load_v4(root: Path, manifest_file: Path, selection: dict) -> LoadedDetector:
    """Verify the registered local V4 run without reading any split audio/cache."""
    from .config import load_config, normalization_lineage_hash
    from .models.cnn import build_cnn, parameter_count
    base = root.parents[1]
    if root != base / "workspaces/study-v4":
        raise ValueError("V4 requires its registered study workspace")
    registry = base / "repository/registry"
    binding = _json(registry / "runtime-selections/study-v4-score-only.json")
    if binding["selection_sha256"] != _sha256(manifest_file):
        raise ValueError("V4 runtime selection differs from the registry")
    run_record = _json(registry / "runs/cnn_a0_seed42.json")
    cache_record = _json(registry / "caches/cnn-v4-train-val.json")
    run = base / "runs/study-v4/cnn_a0_seed42"
    if run_record["root"] != run.relative_to(base).as_posix():
        raise ValueError("V4 run is not registered at the expected location")
    freeze = _json(run / "freeze.json")
    provenance = _json(run / "provenance.json")
    checkpoint = run / "cnn_v4_a0_seed42.pt"
    config_path = run / "config.yaml"
    stats_path = base / "storage/caches/corpus-002/processed/spectrograms/norm_stats_global.npz"
    if (selection.get("schema_version") != 3 or selection.get("model_version") != "4.0"
            or selection.get("variant") != "depth_3" or selection.get("stage") != "A0"
            or selection.get("seed") != 42 or selection.get("threshold") is not None
            or selection.get("policy_id") != "raw-score-only-v1"
            or selection.get("test_access") is not False
            or selection.get("experimental_activation") is not True):
        raise ValueError("V4 must preserve its raw-score-only, before-test contract")
    if (freeze["status"] != "FROZEN_BEFORE_TEST" or freeze["training_runs"] != 1
            or freeze["test_audio_opened"] or freeze["test_scored"] or freeze["threshold_selected"]):
        raise ValueError("V4 training freeze is inconsistent")
    digest = _sha256(checkpoint)
    if not digest == selection["model_sha256"] == freeze["checkpoint_sha256"] == run_record["checkpoint_sha256"]:
        raise ValueError("V4 checkpoint checksum mismatch")
    if _sha256(run / "freeze.json") != selection["freeze_sha256"]:
        raise ValueError("V4 freeze checksum mismatch")
    if _sha256(config_path) != provenance["source_sha256"]["run_config"]:
        raise ValueError("V4 training frontend configuration changed")
    if not _sha256(stats_path) == selection["statistics_sha256"] == cache_record["normalization"]["sha256"] == freeze["normalization_record"]["normalization_file_sha256"]:
        raise ValueError("V4 normalization checksum mismatch")
    # Recheck the source implementing the training frontend against its provenance.
    for name, relative in {"features": "features.py", "normalization": "normalization.py",
                           "cnn_model": "models/cnn.py", "config_module": "config.py"}.items():
        if _sha256(root / "src/dronedet" / relative) != provenance["source_sha256"][name]:
            raise ValueError(f"V4 training source changed: {name}")
    config = load_config(config_path)
    if (config.sr, config.n_mels, config.n_win, config.win_samples) != (16000, 64, 28, 7936):
        raise ValueError("V4 frontend geometry changed")
    with np.load(stats_path, allow_pickle=False) as data:
        stats = {key: data[key].astype(np.float32) for key in ("mean", "std")}
    if any(x.shape != (64,) or not np.isfinite(x).all() for x in stats.values()) or np.any(stats["std"] <= 0):
        raise ValueError("V4 normalization arrays are invalid")
    if normalization_lineage_hash(config, stats) != selection["frontend_hash"]:
        raise ValueError("V4 normalized frontend identity changed")
    torch.set_num_threads(1)
    model = build_cnn((1, 64, 28), channels=(16, 32, 64), dropout=0.3)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    if parameter_count(model) != 23585:
        raise ValueError("V4 parameter count changed")
    detector = LoadedDetector(config=config, stats=stats, model=model, threshold=None,
        seed=42, artifact=checkpoint.relative_to(base).as_posix(), artifact_sha256=digest,
        model_id="CNN_V4.0", model_version="4.0", stage="A0", variant="depth_3",
        validation_eligible=False, experimental_activation=True,
        validation_guard_failures=(), test_access=False, policy_id="raw-score-only-v1")
    detector.score_window(np.zeros(config.win_samples, np.float32))
    return detector
