"""Stamp current cache metadata without reading test-array contents."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dronedet.config import load_project_config, params_hash  # noqa: E402
from dronedet.cache_metadata import sidecar_path  # noqa: E402
from dronedet.data_preparation import locate_dataset_root  # noqa: E402
from dronedet.features import file_to_power, tabular  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("numpy", "scipy", "librosa", "soxr", "soundfile", "pandas"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def git_state() -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip())
    return commit, dirty


def main() -> None:
    project = load_project_config(ROOT / "config.yaml")
    paths, config = project["P"], project["fe"]
    train_manifest_sha = sha256(paths["splits"] / "train.csv")
    expected_train_sha = "60b14ff205bdd4e950ae0642d5ae886c8b8c4c291cabfedc26ee936d9a3e1afc"
    if train_manifest_sha != expected_train_sha:
        raise RuntimeError(f"train manifest hash changed: {train_manifest_sha}")
    report = json.loads((paths["tables"] / "run_report_local_all_models.json").read_text(encoding="utf-8"))
    test_shapes = {
        "mel_power_test.npy": (report["local_preflight"]["loader_smoke"].get("test_power_shape")
                               or report.get("test_cache_shape")),
        "test.npz": report.get("test_tabular_shape"),
    }
    # This frozen report has the metadata-only dimensions recorded by prior smoke checks.
    test_shapes["mel_power_test.npy"] = test_shapes["mel_power_test.npy"] or [64, 91494]
    test_shapes["test.npz"] = test_shapes["test.npz"] or [1506, 84]
    commit, dirty = git_state()
    versions = package_versions()
    raw_root = locate_dataset_root(paths["raw_root"]) / "Binary_Drone_Audio"
    rng = np.random.default_rng(20261123)
    recompute: dict[str, dict[str, Any]] = {}
    for split in ("train", "val"):
        index = pd.read_csv(paths["spec"] / f"index_{split}.csv", keep_default_na=False)
        selected = np.sort(rng.choice(len(index), size=50, replace=False))
        cached_power = np.load(paths["spec"] / f"mel_power_{split}.npy", mmap_mode="r", allow_pickle=False)
        with np.load(paths["tab"] / f"{split}.npz", allow_pickle=False) as archive:
            cached_features = archive["X"]
            power_errors, feature_errors = [], []
            for row_index in selected:
                row = index.iloc[int(row_index)]
                fresh, _ = file_to_power(raw_root / str(row.clip_id), config)
                start = int(row.start_frame)
                old = np.asarray(cached_power[:, start:start + int(row.n_frames)])
                if fresh.shape != old.shape:
                    raise RuntimeError(f"{split} cache shape mismatch at row {row_index}")
                power_errors.append(float(np.max(np.abs(fresh - old))))
                feature_errors.append(float(np.max(np.abs(tabular(fresh, config) - cached_features[int(row_index)]))))
            record = {"seed": 20261123, "sampled_clips": 50,
                      "max_abs_power_difference": max(power_errors),
                      "max_abs_tabular_feature_difference": max(feature_errors),
                      "power_exact": max(power_errors) == 0.0,
                      "tabular_exact": max(feature_errors) == 0.0,
                      "scope": split}
            if not record["power_exact"] or not record["tabular_exact"]:
                raise RuntimeError(f"{split} 50-clip recomputation did not match exactly: {record}")
            recompute[split] = record
    stamped = []
    for split in ("train", "val", "test"):
        split_manifest_sha = sha256(paths["splits"] / f"{split}.csv")
        power_path = paths["spec"] / f"mel_power_{split}.npy"
        tab_path = paths["tab"] / f"{split}.npz"
        if split == "test":
            # Read no test array payload; dimensions and float32 dtype come from the frozen smoke record.
            power_shape, power_dtype = test_shapes["mel_power_test.npy"], "float32"
            tab_shape, tab_dtype = test_shapes["test.npz"], "float32"
            power_provenance = tab_provenance = "built_by_pipeline"
            verification = None
        else:
            power = np.load(power_path, mmap_mode="r", allow_pickle=False)
            power_shape, power_dtype = list(power.shape), str(power.dtype)
            with np.load(tab_path, allow_pickle=False) as archive:
                tab = archive["X"]
                tab_shape, tab_dtype = list(tab.shape), str(tab.dtype)
            power_provenance = tab_provenance = "verified_by_recomputation"
            verification = recompute[split]
        common = {
            "frontend_params_hash": params_hash(config),
            "train_manifest_sha256": train_manifest_sha,
            "split_manifest_sha256": split_manifest_sha,
            "library_versions": versions,
            "platform": platform.platform(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "git_dirty": dirty,
            "test_array_payload_read": False,
        }
        for path, shape, dtype, provenance in (
            (power_path, power_shape, power_dtype, power_provenance),
            (tab_path, tab_shape, tab_dtype, tab_provenance),
        ):
            record = {**common, "cache_file": path.name, "array_shape": shape,
                      "array_dtype": dtype, "provenance": provenance}
            if verification:
                record["verification"] = verification
            sidecar = sidecar_path(path)
            sidecar.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            stamped.append(str(sidecar.relative_to(ROOT)))
    print(json.dumps({"status": "PASS", "frontend_params_hash": params_hash(config),
                      "train_manifest_sha256": train_manifest_sha, "sidecars": stamped,
                      "recomputation": recompute,
                      "test_array_payload_read": False}, indent=2))


if __name__ == "__main__":
    main()
