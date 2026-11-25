"""Run local CNN Stage A (the reusable global_stats A0 seeds).

This intentionally reads only train and validation inputs. The three A0 seeds
are the global_stats seeds reused by B1; no test cache is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dronedet.config import config_hash, legacy_params_hash, load_config, params_hash
from dronedet.features import (get_window, load_cache_index, load_power_cache,
                               to_db)
from dronedet.ledger import append_run, sha256_file, write_run_report
from dronedet.metrics import choose_f1_threshold
from dronedet.models.cnn import assert_parameter_counts, build_cnn, parameter_count
from dronedet.training import train_torch_model


def _root_from_config(config_path: Path) -> Path:
    import yaml
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(raw["paths"]["local"]["project_root"]).resolve()
    if root != Path.cwd().resolve():
        raise RuntimeError(f"config local project_root {root} != current project {Path.cwd()}")
    return root


def _existing_run(root: Path, seed: int, config_id: str) -> dict | None:
    ledger = root / "results" / "tables" / "runs_30.csv"
    archive_path = root / "results" / "tables" / "val_scores_30.npz"
    artifact = root / "results" / "models" / "30" / f"fit_{config_id}_seed{seed}.pt"
    if not (ledger.is_file() and archive_path.is_file() and artifact.is_file()):
        return None
    rows = pd.read_csv(ledger, keep_default_na=False)
    hit = rows[(rows.configuration_id == config_id) & (rows.seed.astype(str) == str(seed))]
    if hit.empty or str(hit.iloc[-1].artefact_sha256) != sha256_file(artifact):
        return None
    run_id = f"{config_id}_seed{seed}"
    with np.load(archive_path, allow_pickle=False) as score_archive:
        if run_id not in score_archive.files:
            return None
        scores = score_archive[run_id].astype(np.float64)
    row = hit.iloc[-1]
    selected = choose_f1_threshold(_Y_VAL, scores)
    return {"scores": scores, "metrics": selected, "seconds": float(row.seconds),
            "parameter_count": int(float(row.parameter_count)), "resumed": True,
            "artifact": str(artifact), "run_id": run_id}


_Y_VAL: np.ndarray


def _normalise_batch(power: np.ndarray, stats: dict[str, np.ndarray], fe) -> np.ndarray:
    """Vectorized equivalent of the global_stats per-window transform."""
    db = to_db(np.asarray(power, dtype=np.float32), fe)
    normalized = (db - stats["mean"][None, :, None]) / stats["std"][None, :, None]
    return normalized[:, None, :, :].astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    root = _root_from_config(args.config.resolve())
    fe = load_config(args.config)
    legacy_params_digest = legacy_params_hash(fe)
    params_digest = params_hash(fe)
    input_shape = (1, fe.n_mels, fe.n_win)
    if fe.mode != "global_stats":
        raise RuntimeError("Stage A requires the frozen global_stats front end")
    expected_threads = os.cpu_count()
    if expected_threads is None or expected_threads != 4:
        raise RuntimeError(f"Stage A expects the recorded four logical CPUs, found {expected_threads}")
    torch.set_num_threads(expected_threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    counts = assert_parameter_counts(input_shape)
    if counts["default"] != 23_585:
        raise AssertionError("A0 parameter count differs from the Stage A contract")

    paths = {"root": root, "spec": root / "data" / "processed" / "spectrograms",
             "splits": root / "data" / "processed" / "splits"}
    train_manifest_path = paths["splits"] / "train.csv"
    val_manifest_path = paths["splits"] / "val.csv"
    train_sha = hashlib.sha256(train_manifest_path.read_bytes()).hexdigest()
    val_sha = hashlib.sha256(val_manifest_path.read_bytes()).hexdigest()
    with (paths["spec"] / "frontend_hash.json").open(encoding="utf-8") as stream:
        frontend_record = json.load(stream)
    if frontend_record.get("params_hash") != legacy_params_digest:
        raise RuntimeError("frozen front-end parameter hash mismatch")
    with np.load(paths["spec"] / "norm_stats_global.npz", allow_pickle=False) as archive:
        stats = {"mean": archive["mean"].astype(np.float32),
                 "std": archive["std"].astype(np.float32)}
        excluded = int(archive["clips_excluded"])
    lineage_digest = config_hash(fe, stats)
    train_index = load_cache_index(paths, "train")
    val_index = load_cache_index(paths, "val")
    train_cache = load_power_cache(paths, "train", mmap_mode="r")
    val_cache = load_power_cache(paths, "val", mmap_mode="r")
    train_manifest = pd.read_csv(train_manifest_path, keep_default_na=False)
    val_manifest = pd.read_csv(val_manifest_path, keep_default_na=False)
    if not np.array_equal(train_manifest.clip_id.to_numpy(), train_index.clip_id.to_numpy()):
        raise RuntimeError("train cache index and manifest order differ")
    if not np.array_equal(val_manifest.clip_id.to_numpy(), val_index.clip_id.to_numpy()):
        raise RuntimeError("validation cache index and manifest order differ")
    y_train = train_manifest.label.to_numpy(dtype=np.float32)
    global _Y_VAL
    _Y_VAL = val_manifest.label.to_numpy(dtype=np.int8)
    validation_power = np.stack([
        get_window(val_cache, row, fe, split="val")
        for row in val_index.itertuples(index=False)
    ]).astype(np.float32)
    validation_x = _normalise_batch(validation_power, stats, fe)
    if validation_x.shape != (len(_Y_VAL), *input_shape) or not np.isfinite(validation_x).all():
        raise RuntimeError("validation tensor has invalid shape or values")

    config_id = hashlib.sha256(json.dumps(
        {"stage": "A", "architecture": "A0", "mode": "global_stats",
         "gain_augmentation": True, "lineage_hash": lineage_digest},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    results: list[dict] = []
    for seed in (42, 43, 44):
        resumed = _existing_run(root, seed, config_id)
        if resumed is not None:
            results.append({"seed": seed, **resumed})
            continue
        python_seed = seed
        random.seed(python_seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        def train_batch(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            batch = np.stack([
                get_window(train_cache, train_index.iloc[int(index)], fe,
                           rng=rng, split="train")
                for index in indices
            ]).astype(np.float32)
            gains_db = rng.uniform(fe.gain_range_db[0], fe.gain_range_db[1], size=len(indices))
            batch *= np.power(10.0, gains_db / 10.0).astype(np.float32)[:, None, None]
            return _normalise_batch(batch, stats, fe)

        model = build_cnn(input_shape)
        if parameter_count(model) != counts["default"]:
            raise AssertionError("runtime A0 model parameter count changed")
        started = time.perf_counter()
        fit = train_torch_model(model, train_batch, validation_x, y_train, _Y_VAL,
                                seed=seed, batch_size=64, learning_rate=1e-3,
                                weight_decay=1e-2, max_epochs=100, patience=8,
                                device="cpu", expected_threads=expected_threads,
                                lineage_hash=lineage_digest)
        elapsed = time.perf_counter() - started
        scores = np.asarray(fit["validation_logits"], dtype=np.float64)
        metrics = choose_f1_threshold(_Y_VAL, scores)
        artifact_dir = root / "results" / "models" / "30"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"{config_id}_seed{seed}"
        artifact = artifact_dir / f"fit_{run_id}.pt"
        temporary = artifact.with_suffix(".pt.tmp")
        torch.save({"state_dict": fit["state_dict"], "lineage_hash": lineage_digest,
                    "seed": seed, "mode": "global_stats", "stage": "A"}, temporary)
        os.replace(temporary, artifact)
        row = {"notebook": 30, "stage": "A", "configuration_id": config_id,
               "seed": seed, "validation_f1": metrics["f1"],
               "validation_roc_auc": metrics["roc_auc"],
               "threshold": metrics["threshold"], "seconds": elapsed,
               "parameter_count": counts["default"], "lineage_hash": lineage_digest,
               "device": "cpu", "thread_count": expected_threads,
               "bit_reproducible": not bool(fit["warnings"]),
               "warnings": json.dumps(fit["warnings"]),
               "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                        cwd=root, text=True).strip(),
               "artefact_path": str(artifact.relative_to(root)),
               "artefact_sha256": sha256_file(artifact), "run_id": run_id}
        append_run(root, 30, row, scores)
        results.append({"seed": seed, "scores": scores, "metrics": metrics,
                        "seconds": elapsed, "parameter_count": counts["default"],
                        "resumed": False, "artifact": str(artifact), "run_id": run_id,
                        "best_epoch": fit["best_epoch"], "warnings": fit["warnings"]})

    train_seconds = [float(row["seconds"]) for row in results]
    projected = {"guaranteed_fits": 69, "maximum_fits": 78,
                 "mean_seconds_per_fit": float(np.mean(train_seconds)),
                 "max_seconds_per_fit": float(np.max(train_seconds)),
                 "guaranteed_hours": float(np.mean(train_seconds) * 69 / 3600),
                 "maximum_hours": float(np.mean(train_seconds) * 78 / 3600),
                 "limit_hours": 24}
    report = {"stage": "A", "status": "STOP_TIME_BUDGET" if projected["guaranteed_hours"] > 24 else "PASS",
              "test_reads": 0, "frontend_params_hash": params_digest,
              "frontend_lineage_hash": lineage_digest,
              "train_manifest_sha256": train_sha, "validation_manifest_sha256": val_sha,
              "train_clips": len(y_train), "validation_clips": len(_Y_VAL),
              "train_stats_excluded_clips": excluded, "input_shape": input_shape,
              "a0_parameter_count": counts["default"], "parameter_counts": counts,
              "seeds": [{"seed": row["seed"], "metrics": row["metrics"],
                         "seconds": row["seconds"], "parameter_count": row["parameter_count"],
                         "resumed": row["resumed"], "artifact": row["artifact"],
                         "run_id": row["run_id"],
                         "best_epoch": row.get("best_epoch"),
                         "warnings": row.get("warnings", [])} for row in results],
              "projected_total": projected,
              "projected_time_stop": projected["guaranteed_hours"] > 24,
              "device": "cpu", "torch_threads": expected_threads,
              "architecture": "A0 Conv-BN-ReLU x3; pool except last; global average; dropout .3; linear logit",
              "training": {"batch_size": 64, "learning_rate": 1e-3,
                           "weight_decay": 1e-2, "max_epochs": 100, "patience": 8,
                           "gain_range_db": list(fe.gain_range_db)}}
    json_path, md_path = write_run_report(root, "30_local_stage_a", report)
    print(json.dumps({"status": report["status"], "a0_parameter_count": counts["default"],
                      "seeds": report["seeds"], "projected_total": projected,
                      "report": str(json_path), "markdown": str(md_path)}, indent=2))
    return 2 if report["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
