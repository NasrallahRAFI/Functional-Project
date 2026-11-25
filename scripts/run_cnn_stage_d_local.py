"""Train and package the registered Stage D clean CNN control locally."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from dronedet.config import (config_hash, legacy_config_hash,
                             legacy_params_hash, load_config, params_hash)
from dronedet.features import centre_window, load_cache_index, load_power_cache
from dronedet.ledger import append_run, sha256_file, write_run_report
from dronedet.metrics import choose_f1_threshold
from dronedet.models.cnn import build_cnn, parameter_count
from dronedet.training import train_torch_model
from scripts.run_cnn_b1_local import SEEDS, _batch_transform

EXPECTED_FRONTEND_HASH = "97544883461de11a1ee51cc17d7028431d6d0ea8f175effc8e63bc18595f28fc"
EXPECTED_STATS_HASH = "327b5e0c54632f3c7e40533d2d0b521faa876cbaf58b8dd8c5bfa57fe9f6b8f0"


def main() -> int:
    if os.cpu_count() != 4:
        raise RuntimeError(f"registered CNN runs expect four logical CPUs; found {os.cpu_count()}")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    config = load_config(ROOT / "config.yaml")
    if config.mode != "global_stats" or legacy_params_hash(config) != EXPECTED_FRONTEND_HASH:
        raise RuntimeError("Stage D front end differs from the frozen front end")
    stage_c_path = ROOT / "results/tables/run_report_30_local_stage_c.json"
    stage_c = json.loads(stage_c_path.read_text(encoding="utf-8"))
    if stage_c.get("status") != "PASS" or stage_c.get("test_reads") != 0:
        raise RuntimeError("Stage D requires passing Stage C with zero test reads")
    winner = stage_c["winner"]
    model_args = {k: tuple(v) if k == "channels" else v
                  for k, v in winner["overrides"].items() if k in {"channels", "kernel_size", "pool", "head"}}
    learning_rate = float(winner["overrides"].get("learning_rate", 1e-3))
    weight_decay = float(winner["overrides"].get("weight_decay", 1e-2))
    frozen = json.loads((ROOT / "results/tables/cnn_frontend_frozen.json").read_text(encoding="utf-8"))
    stats_path = ROOT / "data/processed/spectrograms/norm_stats_global.npz"
    if sha256_file(stats_path) != frozen.get("statistics_sha256"):
        raise RuntimeError("frozen CNN statistics file hash mismatch")
    paths = {"root": ROOT, "spec": ROOT / "data/processed/spectrograms",
             "splits": ROOT / "data/processed/splits"}
    train_manifest = pd.read_csv(paths["splits"] / "train.csv", keep_default_na=False)
    val_manifest = pd.read_csv(paths["splits"] / "val.csv", keep_default_na=False)
    train_index = load_cache_index(paths, "train")
    val_index = load_cache_index(paths, "val")
    train_cache = load_power_cache(paths, "train", mmap_mode="r")
    val_cache = load_power_cache(paths, "val", mmap_mode="r")
    for split, manifest, index in (("train", train_manifest, train_index),
                                   ("val", val_manifest, val_index)):
        if not np.array_equal(manifest.clip_id.to_numpy(), index.clip_id.to_numpy()):
            raise RuntimeError(f"Stage D {split} manifest/cache order mismatch")
    with np.load(stats_path, allow_pickle=False) as archive:
        stats = {"mean": archive["mean"].astype(np.float32),
                 "std": archive["std"].astype(np.float32),
                 "clips_excluded": int(archive["clips_excluded"])}
    lineage_hash = config_hash(config, stats)
    if (legacy_config_hash(config, stats) != EXPECTED_STATS_HASH or
            legacy_config_hash(config, stats) != frozen.get("lineage_hash")):
        raise RuntimeError("Stage D normalization lineage differs from frozen baseline")

    def fixed_inputs(cache, index):
        power = np.stack([centre_window(
            cache[:, int(row.start_frame):int(row.start_frame) + int(row.n_frames)], config)
            for row in index.itertuples(index=False)]).astype(np.float32)
        return _batch_transform(power, config, stats)

    val_X = fixed_inputs(val_cache, val_index)
    y_train = train_manifest.label.to_numpy(dtype=np.float32)
    y_val = val_manifest.label.to_numpy(dtype=np.int8)
    architecture = {"model": {k: list(v) if isinstance(v, tuple) else v
                              for k, v in model_args.items()},
                    "learning_rate": learning_rate, "weight_decay": weight_decay,
                    "batch_size": 64, "max_epochs": 100, "patience": 8,
                    "augmentation": "none", "window": "fixed centered"}
    config_id = hashlib.sha256(json.dumps(
        {"stage": "D", "architecture": architecture,
         "front_end": params_hash(config), "lineage_hash": lineage_hash},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    results, histories = [], {}
    for seed in SEEDS:
        print(json.dumps({"event": "fit_start", "stage": "D", "seed": seed}), flush=True)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = build_cnn((1, config.n_mels, config.n_win), **model_args)
        artifact = ROOT / "results/models/30_cnn_control" / f"seed_{seed}.pt"

        def train_batch(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            powers = np.stack([centre_window(
                train_cache[:, int(row.start_frame):int(row.start_frame) + int(row.n_frames)], config)
                for row in train_index.iloc[indices].itertuples(index=False)]).astype(np.float32)
            return _batch_transform(powers, config, stats)

        fit = train_torch_model(model, train_batch, val_X, y_train, y_val,
                                seed=seed, batch_size=64, learning_rate=learning_rate,
                                weight_decay=weight_decay, max_epochs=100, patience=8,
                                device="cpu", expected_threads=4, lineage_hash=lineage_hash)
        logits = fit["validation_logits"].astype(np.float64)
        selected = choose_f1_threshold(y_val, logits)
        pred = (logits >= float(selected["threshold"])).astype(np.int8)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": fit["state_dict"], "lineage_hash": lineage_hash,
                    "seed": seed, "stage": "D", "architecture": architecture}, artifact)
        run_id = f"{config_id}_seed{seed}"
        row = {"notebook": 30, "stage": "D", "configuration_id": config_id,
               "seed": seed, "validation_f1": selected["f1"],
               "validation_roc_auc": float(roc_auc_score(y_val, logits)),
               "threshold": selected["threshold"], "seconds": fit["training_seconds"],
               "best_epoch": fit["best_epoch"], "parameter_count": parameter_count(fit["model"]),
               "lineage_hash": lineage_hash, "device": "cpu", "thread_count": 4,
               "bit_reproducible": not bool(fit["warnings"]),
               "warnings": json.dumps(fit["warnings"]),
               "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "artefact_path": str(artifact.relative_to(ROOT)),
               "artefact_sha256": sha256_file(artifact), "run_id": run_id}
        append_run(ROOT, 30, row, logits)
        histories[str(seed)] = fit["history"]
        results.append({"seed": seed, "validation_f1": float(selected["f1"]),
                        "validation_roc_auc": row["validation_roc_auc"],
                        "threshold": float(selected["threshold"]),
                        "confusion_matrix": confusion_matrix(y_val, pred, labels=[0, 1]).tolist(),
                        "best_epoch": fit["best_epoch"], "stopping_epoch": len(fit["history"]),
                        "seconds": fit["training_seconds"], "parameter_count": row["parameter_count"],
                        "warnings": fit["warnings"], "run_id": run_id,
                        "artifact": str(artifact.relative_to(ROOT)),
                        "artifact_sha256": row["artefact_sha256"]})
    designated = max(results, key=lambda x: (x["validation_f1"], -x["seed"]))
    package_dir = ROOT / "results/models/30_cnn_control"
    package = {"card_id": "30_cnn_control", "stage": "D", "designated_seed": designated["seed"],
               "selection": "highest D044 validation F1 at its own optimal threshold; ties to lower seed",
               "configuration_id": config_id, "architecture": architecture,
               "frontend_params_hash": params_hash(config), "statistics_lineage_hash": lineage_hash,
               "statistics_sha256": sha256_file(stats_path), "validation_manifest_sha256": sha256_file(paths["splits"] / "val.csv"),
               "seeds": results, "test_reads": 0}
    (package_dir / "model_card.json").write_text(json.dumps(package, indent=2), encoding="utf-8")
    report = {"stage": "D", "status": "PASS", "test_reads": 0,
              "winner_from_stage_c": winner["name"], "configuration_id": config_id,
              "architecture": architecture, "frontend_params_hash": params_hash(config),
              "statistics_lineage_hash": lineage_hash, "fixed_centered_windows": True,
              "gain_augmentation": False, "epochs": histories, "seed_results": results,
              "designated_seed": designated["seed"], "model_card": str((package_dir / "model_card.json").relative_to(ROOT)),
              "training_data": "fixed centered train-cache windows; validation used for early stopping, threshold selection, and seed designation"}
    path, _ = write_run_report(ROOT, "30_local_stage_d", report)
    card = ["# 30_cnn_control", "", "Stage D clean CNN control for the MLP comparison.", "",
            f"- Architecture: `{json.dumps(architecture, sort_keys=True)}`",
            f"- Front end hash: `{params_hash(config)}`", f"- Statistics lineage: `{lineage_hash}`",
            "- Training: fixed centered windows, no gain augmentation; CPU, AdamW, batch size 64, max 100 epochs, patience 8.",
            f"- Designated validation seed: {designated['seed']} (F1 {designated['validation_f1']:.4f}; threshold {designated['threshold']:.6g}).",
            "- The package includes all three seed checkpoints. No test data was read.", ""]
    (ROOT / "results/cards/30_cnn_control.md").write_text("\n".join(card), encoding="utf-8")
    print(json.dumps({"event": "stage_d_complete", "report": str(path),
                      "designated_seed": designated["seed"], "model_card": str(package_dir / "model_card.json")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
