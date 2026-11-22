"""Append-only fit ledger, freeze packages, verdict cards and run reports."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np


LEDGER_FIELDS = ["notebook", "stage", "configuration_id", "seed", "validation_f1",
                 "validation_roc_auc", "threshold", "seconds", "parameter_count",
                 "lineage_hash", "device", "thread_count", "bit_reproducible",
                 "warnings", "code_commit", "artefact_path", "artefact_sha256"]
CARD_FIELDS = ["model", "representation", "best_hyperparameters", "val_metrics", "seeds",
               "train_time_s", "strengths", "weaknesses", "failure_cases_noted",
               "decision_log_ids", "threshold", "threshold_rule", "selection_metric",
               "val_auc", "n_parameters", "grid_searched", "tie_with_default",
               "front_end_hash"]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_code_commit(root: str | Path = ".") -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def package_versions() -> dict[str, str | None]:
    names = ("numpy", "pandas", "scipy", "librosa", "soxr", "soundfile",
             "scikit-learn", "torch", "matplotlib", "PyYAML", "joblib")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _tables(root: str | Path) -> Path:
    path = Path(root) / "results" / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_run(root: str | Path, notebook: int | str, run: dict[str, Any],
               validation_scores: np.ndarray) -> None:
    tables = _tables(root)
    path = tables / f"runs_{notebook}.csv"
    scores_path = tables / f"val_scores_{notebook}.npz"
    exists = path.exists()
    row = {field: run.get(field, "") for field in LEDGER_FIELDS}
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LEDGER_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    run_id = str(run["run_id"])
    score_map: dict[str, np.ndarray] = {}
    if scores_path.exists():
        with np.load(scores_path, allow_pickle=False) as archive:
            score_map = {key: archive[key] for key in archive.files}
    # An orphaned/invalid prior fit is not resumable; refresh only this run's validation vector.
    score_map[run_id] = np.asarray(validation_scores, dtype=np.float64)
    tmp = scores_path.with_name(scores_path.name + ".tmp.npz")
    np.savez_compressed(tmp, **score_map)
    os.replace(tmp, scores_path)


def save_freeze_package(root: str | Path, notebook: int | str, *,
                        estimators: dict[int, Any], scaler: Any | None,
                        thresholds: dict[int, float], validation_f1: dict[int, float],
                        lineage_hash: str, tabular_hashes: dict[str, str] | None = None,
                        metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Persist every seed, scaler and metadata without overwriting a prior freeze."""
    root = Path(root)
    output = root / "results" / "models" / str(notebook)
    expected_files = [output / f"seed_{seed}.joblib" for seed in estimators]
    expected_files += [output / f"seed_{seed}.pt" for seed in estimators]
    if scaler is not None:
        expected_files.append(output / "scaler.joblib")
    if (output / "metadata.json").exists() or any(path.exists() for path in expected_files):
        raise FileExistsError(f"refusing to overwrite existing freeze package: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        import torch
    except ImportError:
        torch = None
    import joblib
    files = {}
    for seed, estimator in estimators.items():
        path = output / f"seed_{seed}.artifact"
        if torch is not None and isinstance(estimator, torch.nn.Module):
            path = output / f"seed_{seed}.pt"
            torch.save({"state_dict": estimator.state_dict(), "lineage_hash": lineage_hash}, path)
        else:
            path = output / f"seed_{seed}.joblib"
            joblib.dump(estimator, path)
        files[path.name] = sha256_file(path)
    if scaler is not None:
        path = output / "scaler.joblib"
        joblib.dump(scaler, path)
        files[path.name] = sha256_file(path)
    best_seed = min(validation_f1, key=lambda seed: (-float(validation_f1[seed]), int(seed)))
    metadata_doc = {"notebook": str(notebook), "seeds": sorted(int(s) for s in estimators),
                    "thresholds": {str(k): float(v) for k, v in thresholds.items()},
                    "validation_f1": {str(k): float(v) for k, v in validation_f1.items()},
                    "designated_best_seed": int(best_seed), "lineage_hash": lineage_hash,
                    "tabular_npz_sha256": tabular_hashes or {},
                    "libraries": package_versions(), "python": sys.version,
                    "platform": platform.platform(), "device": "cpu",
                    "torch_threads": __import__("torch").get_num_threads() if torch else None,
                    "code_commit": current_code_commit(root), "artifacts": files,
                    "created_utc": datetime.now(timezone.utc).isoformat(), **(metadata or {})}
    meta_path = output / "metadata.json"
    meta_path.write_text(json.dumps(metadata_doc, indent=2) + "\n", encoding="utf-8")
    return {"path": str(output), "designated_best_seed": int(best_seed),
            "metadata": metadata_doc}


def write_verdict_card(root: str | Path, notebook: int | str,
                       card: dict[str, Any]) -> Path:
    missing = [field for field in CARD_FIELDS if field not in card]
    if missing:
        raise ValueError(f"verdict card missing required fields: {missing}")
    output = Path(root) / "results" / "cards"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{notebook}_verdict.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite verdict card: {path}")
    path.write_text(json.dumps({field: card[field] for field in CARD_FIELDS}, indent=2) + "\n",
                    encoding="utf-8")
    return path


def write_run_report(root: str | Path, name: str, report: dict[str, Any]) -> tuple[Path, Path]:
    tables = _tables(root)
    json_path = tables / f"run_report_{name}.json"
    md_path = tables / f"run_report_{name}.md"
    if json_path.exists() or md_path.exists():
        raise FileExistsError(f"refusing to overwrite existing run report for {name}")
    doc = {"created_utc": datetime.now(timezone.utc).isoformat(),
           "code_commit": current_code_commit(root), "python": sys.version,
           "platform": platform.platform(), "device": "cpu",
           "torch_threads": __import__("torch").get_num_threads(),
           "libraries": package_versions(), **report}
    body = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    json_path.write_text(body, encoding="utf-8")
    md_path.write_text(f"# Run report: {name}\n\n```json\n{body}```\n", encoding="utf-8")
    return json_path, md_path
