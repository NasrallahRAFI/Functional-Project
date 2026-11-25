"""Run the registered B3 mel-EQ augmentation control on local CPU."""
from __future__ import annotations

import argparse
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dronedet.config import (config_hash, legacy_config_hash,
                             legacy_params_hash, load_config, params_hash)
from dronedet.features import get_window, load_cache_index, load_power_cache
from dronedet.ledger import append_run, sha256_file, write_run_report
from dronedet.models.cnn import build_cnn, parameter_count
from dronedet.normalization import apply_mel_eq_training
from dronedet.training import train_torch_model
from scripts.run_cnn_b1_local import (
    CONDITIONS, FILTERS, SEEDS, _batch_transform, _evaluate, _load_a0_state,
    _validation_windows,
)


ROOT = Path.cwd().resolve()
EXPECTED_FRONTEND_HASH = "97544883461de11a1ee51cc17d7028431d6d0ea8f175effc8e63bc18595f28fc"
EXPECTED_STATS_HASH = "327b5e0c54632f3c7e40533d2d0b521faa876cbaf58b8dd8c5bfa57fe9f6b8f0"


def _train_eq_seed(seed: int, config, stats: dict, train_index: pd.DataFrame,
                   train_cache: np.ndarray, y_train: np.ndarray,
                   val_powers: dict[str, np.ndarray], y_val: np.ndarray,
                   lineage_hash: str, max_db: float,
                   n_control_points: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    augmentation = {"name": "mel_eq", "max_db": float(max_db),
                    "n_control_points": int(n_control_points)}
    params = {"stage": "B3", "architecture": "A0", "mode": "global_stats",
              "gain_augmentation": True, "eq_augmentation": augmentation,
              "lineage_hash": lineage_hash}
    config_id = hashlib.sha256(json.dumps(
        params, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    run_id = f"{config_id}_seed{seed}"
    artifact = ROOT / "results/models/30" / f"fit_{run_id}.pt"
    ledger_path = ROOT / "results/tables/runs_30.csv"
    scores_path = ROOT / "results/tables/val_scores_30.npz"
    model = build_cnn((1, config.n_mels, config.n_win))
    if ledger_path.is_file() and scores_path.is_file() and artifact.is_file():
        ledger = pd.read_csv(ledger_path, keep_default_na=False)
        hit = ledger[(ledger.configuration_id == config_id) &
                     (ledger.seed.astype(str) == str(seed))]
        with np.load(scores_path, allow_pickle=False) as archive:
            has_scores = run_id in archive.files
            saved = archive[run_id].astype(np.float64) if has_scores else None
        if len(hit) and has_scores and hit.iloc[-1].artefact_sha256 == sha256_file(artifact):
            payload = torch.load(artifact, map_location="cpu", weights_only=True)
            if payload.get("lineage_hash") != lineage_hash:
                raise RuntimeError(f"B3 checkpoint lineage mismatch for {run_id}")
            model.load_state_dict(payload["state_dict"])
            result = _evaluate(model, val_powers, y_val, config, stats)
            if not np.allclose(result["validation_scores"]["gain_+0db"], saved,
                               rtol=0, atol=1e-6):
                raise RuntimeError(f"B3 resume score mismatch for {run_id}")
            result.update({"seed": seed, "seconds": float(hit.iloc[-1].seconds),
                           "artifact": str(artifact), "run_id": run_id, "resumed": True,
                           "parameter_count": parameter_count(model)})
            return result

    def train_batch(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        windows = np.stack([
            get_window(train_cache, train_index.iloc[int(i)], config,
                       rng=rng, split="train") for i in indices
        ]).astype(np.float32)
        gains = rng.uniform(config.gain_range_db[0], config.gain_range_db[1],
                            size=len(indices))
        windows *= np.power(10.0, gains / 10.0).astype(np.float32)[:, None, None]
        windows = np.stack([
            apply_mel_eq_training(window, rng, training=True, max_db=max_db,
                                  n_control_points=n_control_points)
            for window in windows
        ]).astype(np.float32)
        return _batch_transform(windows, config, stats)

    val_base = _batch_transform(val_powers["unfiltered"], config, stats)
    fit = train_torch_model(model, train_batch, val_base, y_train, y_val,
                            seed=seed, batch_size=64, learning_rate=1e-3,
                            weight_decay=1e-2, max_epochs=100, patience=8,
                            device="cpu", expected_threads=4, lineage_hash=lineage_hash)
    result = _evaluate(fit["model"], val_powers, y_val, config, stats)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_suffix(".pt.tmp")
    torch.save({"state_dict": fit["state_dict"], "lineage_hash": lineage_hash,
                "seed": seed, "stage": "B3", "augmentation": augmentation}, temporary)
    os.replace(temporary, artifact)
    row = {"notebook": 30, "stage": "B3", "configuration_id": config_id,
           "seed": seed, "validation_f1": result["base_metrics"]["f1"],
           "validation_roc_auc": result["base_metrics"]["roc_auc"],
           "threshold": result["threshold"], "seconds": fit["training_seconds"],
           "parameter_count": parameter_count(fit["model"]),
           "lineage_hash": lineage_hash, "device": "cpu", "thread_count": 4,
           "bit_reproducible": not bool(fit["warnings"]),
           "warnings": json.dumps(fit["warnings"]),
           "code_commit": subprocess.check_output(
               ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "artefact_path": str(artifact.relative_to(ROOT)),
           "artefact_sha256": sha256_file(artifact), "run_id": run_id}
    append_run(ROOT, 30, row, result["validation_scores"]["gain_+0db"])
    result.update({"seed": seed, "seconds": fit["training_seconds"],
                   "artifact": str(artifact), "run_id": run_id, "resumed": False,
                   "parameter_count": parameter_count(fit["model"]),
                   "best_epoch": fit["best_epoch"], "warnings": fit["warnings"]})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-db", type=float, default=3.0,
                        help="maximum mel EQ curve magnitude in dB (closeout-tested: 3)")
    parser.add_argument("--n-control-points", type=int, required=True,
                        help="number of uniformly sampled EQ control points")
    args = parser.parse_args()
    if os.cpu_count() != 4:
        raise RuntimeError(f"registered CNN runs expect four logical CPUs; found {os.cpu_count()}")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    config = load_config(ROOT / "config.yaml")
    if config.mode != "global_stats" or legacy_params_hash(config) != EXPECTED_FRONTEND_HASH:
        raise RuntimeError("B3 must use the tagged project global_stats front end")
    if not np.isfinite(args.max_db) or args.max_db < 0:
        raise ValueError("--max-db must be finite and nonnegative")
    if not 1 <= args.n_control_points <= config.n_mels:
        raise ValueError(f"--n-control-points must be in [1, {config.n_mels}]")
    b2 = json.loads((ROOT / "results/tables/run_report_30_local_b2.json").read_text())
    if b2.get("status") != "PASS" or b2.get("selection", {}).get("winner", {}).get("mode") != "global_stats":
        raise RuntimeError("B3 requires the completed B2 report with global_stats retained")
    paths = {"root": ROOT, "spec": ROOT / "data/processed/spectrograms",
             "splits": ROOT / "data/processed/splits"}
    train_manifest = pd.read_csv(paths["splits"] / "train.csv", keep_default_na=False)
    val_manifest = pd.read_csv(paths["splits"] / "val.csv", keep_default_na=False)
    train_index = load_cache_index(paths, "train")
    val_index = load_cache_index(paths, "val")
    train_cache = load_power_cache(paths, "train", mmap_mode="r")
    val_cache = load_power_cache(paths, "val", mmap_mode="r")
    if not np.array_equal(train_manifest.clip_id.to_numpy(), train_index.clip_id.to_numpy()):
        raise RuntimeError("B3 train manifest/cache order mismatch")
    if not np.array_equal(val_manifest.clip_id.to_numpy(), val_index.clip_id.to_numpy()):
        raise RuntimeError("B3 validation manifest/cache order mismatch")
    eligible = int(train_index.stats_eligible.astype(bool).sum())
    if eligible != 7519:
        raise RuntimeError(f"B3 stats_eligible count mismatch: expected 7519, got {eligible}")
    with np.load(paths["spec"] / "norm_stats_global.npz", allow_pickle=False) as archive:
        stats = {"mean": archive["mean"].astype(np.float32),
                 "std": archive["std"].astype(np.float32),
                 "clips_excluded": int(archive["clips_excluded"])}
    lineage_hash = config_hash(config, stats)
    if legacy_config_hash(config, stats) != EXPECTED_STATS_HASH or stats["clips_excluded"] != 88:
        raise RuntimeError("B3 normalization-statistics lineage does not match the frozen front end")
    yaml_data = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw_root = Path(yaml_data["paths"]["local"]["raw_root"]).resolve()
    val_powers, y_val = _validation_windows(config, val_index, val_cache,
                                             val_manifest, raw_root)
    y_train = train_manifest.label.to_numpy(dtype=np.float32)
    baseline = json.loads((ROOT / "results/tables/run_report_30_local_b1.json").read_text())
    baseline_rows = baseline["mode_fit_results"]["global_stats"]
    baseline_scores = []
    for seed in SEEDS:
        model = _load_a0_state(seed, baseline_rows[0]["run_id"].rsplit("_seed", 1)[0])
        evaluated = _evaluate(model, val_powers, y_val, config, stats)
        baseline_scores.append({"seed": seed, "mode": "no_mel_eq",
                                "threshold": evaluated["threshold"],
                                "base_metrics": evaluated["base_metrics"],
                                "condition_f1": evaluated["condition_f1"],
                                "worst_condition_f1": evaluated["worst_condition_f1"],
                                "run_id": baseline_rows[seed - 42]["run_id"]})
    eq_rows = []
    for seed in SEEDS:
        print(json.dumps({"event": "fit_start", "stage": "B3",
                          "seed": seed, "max_db": args.max_db,
                          "n_control_points": args.n_control_points}), flush=True)
        eq_rows.append(_train_eq_seed(seed, config, stats, train_index, train_cache,
                                      y_train, val_powers, y_val, lineage_hash,
                                      args.max_db, args.n_control_points))
    report = {"stage": "B3", "status": "PASS", "test_reads": 0,
              "frontend_params_hash": params_hash(config),
              "statistics_lineage_hash": lineage_hash,
              "stats_eligible_train_clips": eligible,
              "excluded_train_clips": 88,
              "augmentation": {"name": "mel_eq", "max_db": args.max_db,
                               "n_control_points": args.n_control_points,
                               "operation_order": "random_window, gain, mel_eq in power space, dB normalization",
                               "training_only": True},
              "conditions": list(CONDITIONS),
              "threshold_rule": "per-seed threshold selected at unfiltered 0 dB and held for all conditions",
              "control_role": "reported only; never adopted into front-end freeze",
              "baseline_no_mel_eq": baseline_scores,
              "mel_eq_runs": [{k: v for k, v in r.items() if k != "validation_scores"}
                              for r in eq_rows],
              "device": "cpu", "torch_threads": 4,
              "validation_raw_root_used": str(raw_root)}
    path, _ = write_run_report(ROOT, "30_local_b3", report)
    print(json.dumps({"event": "B3_complete", "report": str(path),
                      "baseline_worst": [r["worst_condition_f1"] for r in baseline_scores],
                      "mel_eq_worst": [r["worst_condition_f1"] for r in eq_rows]},
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
