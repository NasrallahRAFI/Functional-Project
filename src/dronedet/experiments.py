"""Reusable local fit/resume and run-record helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.class_weight import compute_sample_weight

from .ledger import append_run, sha256_file
from .metrics import choose_f1_threshold
from .models.classical import parameter_complexity, score


def configuration_id(params: dict[str, Any]) -> str:
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _run_id(config_id: str, seed: int) -> str:
    return f"{config_id}_seed{seed}"


def _artifact_path(root: Path, notebook: int, run_id: str) -> Path:
    return root / "results" / "models" / str(notebook) / f"fit_{run_id}.joblib"


def _load_resume(root: Path, notebook: int, config_id: str, seed: int):
    run_id = _run_id(config_id, seed)
    ledger_path = root / "results" / "tables" / f"runs_{notebook}.csv"
    score_path = root / "results" / "tables" / f"val_scores_{notebook}.npz"
    artifact = _artifact_path(root, notebook, run_id)
    if not ledger_path.is_file() or not score_path.is_file() or not artifact.is_file():
        return None
    with ledger_path.open(encoding="utf-8", newline="") as stream:
        matches = [row for row in csv.DictReader(stream)
                   if row.get("configuration_id") == config_id and row.get("seed") == str(seed)]
    if not matches:
        return None
    row = matches[-1]
    if row.get("artefact_sha256") != sha256_file(artifact):
        return None
    with np.load(score_path, allow_pickle=False) as archive:
        if run_id not in archive.files:
            return None
        values = archive[run_id]
    estimator = joblib.load(artifact)
    return {"estimator": estimator, "scores": values,
            "threshold": float(row["threshold"]), "f1": float(row["validation_f1"]),
            "roc_auc": float(row["validation_roc_auc"]), "seconds": float(row["seconds"]),
            "complexity": int(float(row["parameter_count"])), "warnings": json.loads(row["warnings"] or "[]"),
            "run_id": run_id, "resumed": True}


def fit_classical(root: str | Path, notebook: int, params: dict[str, Any], seed: int,
                  X_train: np.ndarray, y_train: np.ndarray,
                  X_validation: np.ndarray, y_validation: np.ndarray, *,
                  score_kind: str, sample_weight: bool = False,
                  lineage_hash: str, device: str = "cpu", thread_count: int = 1) -> dict[str, Any]:
    """Fit or validate a resumable scikit-learn run and append its validation record."""
    root = Path(root).resolve()
    config_id = configuration_id(params)
    run_id = _run_id(config_id, seed)
    resumed = _load_resume(root, notebook, config_id, seed)
    if resumed is not None:
        return resumed
    from .models.classical import build_model
    estimator = build_model(notebook, params, seed=seed)
    kwargs = {"sample_weight": compute_sample_weight("balanced", y_train)} if sample_weight else {}
    warning_log: list[str] = []
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(X_train, y_train, **kwargs)
    warning_log.extend(f"{w.category.__name__}: {w.message}" for w in caught)
    elapsed = time.perf_counter() - started
    if notebook == 11 and any(isinstance(w.message, ConvergenceWarning) for w in caught):
        first_fit_seconds = elapsed
        warning_log.append(f"initial max_iter=1000 fit seconds={first_fit_seconds:.6f}; retrying with max_iter=10000 after convergence warning")
        from .models.classical import build_model as make_model
        estimator = make_model(11, {**params, "max_iter": 10000}, seed=seed)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as retry_warnings:
            warnings.simplefilter("always")
            estimator.fit(X_train, y_train, **kwargs)
        warning_log.extend(f"retry {w.category.__name__}: {w.message}" for w in retry_warnings)
        elapsed = first_fit_seconds + (time.perf_counter() - started)
    values = np.asarray(score(estimator, X_validation, kind=score_kind), dtype=np.float64).reshape(-1)
    if len(values) != len(y_validation) or not np.isfinite(values).all():
        raise ValueError(f"notebook {notebook} produced invalid validation scores")
    selected = choose_f1_threshold(y_validation, values)
    complexity = parameter_complexity(estimator)
    package = root / "results" / "models" / str(notebook)
    package.mkdir(parents=True, exist_ok=True)
    artifact = _artifact_path(root, notebook, run_id)
    temp = artifact.with_name(artifact.name + ".tmp")
    joblib.dump(estimator, temp)
    os.replace(temp, artifact)
    artifact_hash = sha256_file(artifact)
    row = {"notebook": notebook, "stage": "", "configuration_id": config_id,
           "seed": seed, "validation_f1": selected["f1"], "validation_roc_auc": selected["roc_auc"],
           "threshold": selected["threshold"], "seconds": elapsed,
           "parameter_count": complexity, "lineage_hash": lineage_hash,
           "device": device, "thread_count": thread_count, "bit_reproducible": True,
           "warnings": json.dumps(warning_log),
           "code_commit": __import__("subprocess").check_output(
               ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
           "artefact_path": str(artifact.relative_to(root)), "artefact_sha256": artifact_hash,
           "run_id": run_id}
    append_run(root, notebook, row, values)
    return {"estimator": estimator, "scores": values, **selected, "seconds": elapsed,
            "complexity": complexity, "warnings": warning_log, "run_id": run_id,
            "configuration_id": config_id, "artifact": str(artifact), "resumed": False}
