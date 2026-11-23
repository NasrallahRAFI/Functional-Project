"""Run the registered Stage C CNN architecture and training sweep on local CPU."""
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
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dronedet.config import config_hash, load_config, params_hash
from dronedet.features import get_window, load_cache_index, load_power_cache
from dronedet.ledger import append_run, sha256_file, write_run_report
from dronedet.metrics import choose_f1_threshold
from dronedet.models.cnn import assert_parameter_counts, build_cnn, parameter_count
from dronedet.training import train_torch_model
from scripts.run_cnn_b1_local import SEEDS, _batch_transform, _load_a0_state


ROOT = Path.cwd().resolve()
EXPECTED_FRONTEND_HASH = "97544883461de11a1ee51cc17d7028431d6d0ea8f175effc8e63bc18595f28fc"
EXPECTED_STATS_HASH = "327b5e0c54632f3c7e40533d2d0b521faa876cbaf58b8dd8c5bfa57fe9f6b8f0"
# Reduced Stage C scope: retain the six completed variants and flatten_head;
# omit the six not-yet-started dropout and optimizer variants per user decision.
VARIANTS = [
    {"name": "widths_8_16_32", "factor": "channels", "model": {"channels": (8, 16, 32)}},
    {"name": "widths_32_64_128", "factor": "channels", "model": {"channels": (32, 64, 128)}},
    {"name": "depth_2", "factor": "depth", "model": {"channels": (16, 32)}},
    {"name": "depth_4", "factor": "depth", "model": {"channels": (16, 32, 64, 64)}},
    {"name": "kernel_5", "factor": "first_kernel", "model": {"kernel_size": 5}},
    {"name": "pool_2x1", "factor": "pooling", "model": {"pool": (2, 1)}},
    {"name": "flatten_head", "factor": "head", "model": {"head": "flatten"}},
]


def _fit_seed(name: str, seed: int, model_args: dict, learning_rate: float,
              weight_decay: float, config, stats: dict, train_index: pd.DataFrame,
              train_cache: np.ndarray, y_train: np.ndarray, val_X: np.ndarray,
              y_val: np.ndarray, lineage_hash: str) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_cnn((1, config.n_mels, config.n_win), **model_args)
    model_parameters = {"model": model_args, "learning_rate": learning_rate,
                        "weight_decay": weight_decay, "batch_size": 64,
                        "max_epochs": 100, "patience": 8}
    config_id = hashlib.sha256(json.dumps(
        {"stage": "C", "variant": name, "architecture": model_parameters,
         "front_end": params_hash(config), "lineage_hash": lineage_hash,
         "gain_augmentation": True},
        sort_keys=True, separators=(",", ":"), default=list).encode()).hexdigest()
    run_id = f"{config_id}_seed{seed}"
    artifact = ROOT / "results/models/30" / f"fit_{run_id}.pt"
    ledger_path = ROOT / "results/tables/runs_30.csv"
    score_path = ROOT / "results/tables/val_scores_30.npz"
    if artifact.is_file() and ledger_path.is_file() and score_path.is_file():
        ledger = pd.read_csv(ledger_path, keep_default_na=False)
        hit = ledger[(ledger.configuration_id == config_id) &
                     (ledger.seed.astype(str) == str(seed))]
        with np.load(score_path, allow_pickle=False) as archive:
            has_score = run_id in archive.files
            saved = archive[run_id].astype(np.float64) if has_score else None
        if len(hit) and has_score and hit.iloc[-1].artefact_sha256 == sha256_file(artifact):
            payload = torch.load(artifact, map_location="cpu", weights_only=True)
            if payload.get("lineage_hash") != lineage_hash:
                raise RuntimeError(f"Stage C checkpoint lineage mismatch for {run_id}")
            model.load_state_dict(payload["state_dict"])
            scores = model.eval()
            with torch.no_grad():
                current = model(torch.from_numpy(val_X)).cpu().numpy().astype(np.float64)
            if not np.allclose(current, saved, rtol=0, atol=1e-6):
                raise RuntimeError(f"Stage C resume score mismatch for {run_id}")
            threshold = choose_f1_threshold(y_val, current)
            return {"name": name, "seed": seed, "threshold": threshold["threshold"],
                    "validation_f1": threshold["f1"],
                    "validation_roc_auc": float(roc_auc_score(y_val, current)),
                    "seconds": float(hit.iloc[-1].seconds), "run_id": run_id,
                    "artifact": str(artifact), "resumed": True,
                    "parameter_count": parameter_count(model)}

    def train_batch(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        windows = np.stack([get_window(train_cache, train_index.iloc[int(i)], config,
                                       rng=rng, split="train") for i in indices]).astype(np.float32)
        gains = rng.uniform(config.gain_range_db[0], config.gain_range_db[1], len(indices))
        windows *= np.power(10.0, gains / 10.0).astype(np.float32)[:, None, None]
        return _batch_transform(windows, config, stats)

    fit = train_torch_model(model, train_batch, val_X, y_train, y_val,
                            seed=seed, batch_size=64, learning_rate=learning_rate,
                            weight_decay=weight_decay, max_epochs=100, patience=8,
                            device="cpu", expected_threads=4, lineage_hash=lineage_hash)
    logits = fit["validation_logits"].astype(np.float64)
    selected = choose_f1_threshold(y_val, logits)
    auc = float(roc_auc_score(y_val, logits))
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_suffix(".pt.tmp")
    torch.save({"state_dict": fit["state_dict"], "lineage_hash": lineage_hash,
                "seed": seed, "stage": "C", "variant": name,
                "architecture": model_parameters}, temporary)
    os.replace(temporary, artifact)
    row = {"notebook": 30, "stage": "C", "configuration_id": config_id,
           "seed": seed, "validation_f1": selected["f1"],
           "validation_roc_auc": auc, "threshold": selected["threshold"],
           "seconds": fit["training_seconds"],
           "parameter_count": parameter_count(fit["model"]),
           "lineage_hash": lineage_hash, "device": "cpu", "thread_count": 4,
           "bit_reproducible": not bool(fit["warnings"]),
           "warnings": json.dumps(fit["warnings"]),
           "code_commit": subprocess.check_output(
               ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "artefact_path": str(artifact.relative_to(ROOT)),
           "artefact_sha256": sha256_file(artifact), "run_id": run_id}
    append_run(ROOT, 30, row, logits)
    return {"name": name, "seed": seed, "threshold": selected["threshold"],
            "validation_f1": selected["f1"], "validation_roc_auc": auc,
            "seconds": fit["training_seconds"], "best_epoch": fit["best_epoch"],
            "warnings": fit["warnings"], "run_id": run_id,
            "artifact": str(artifact), "resumed": False,
            "parameter_count": parameter_count(fit["model"])}


def _summarize(name: str, factor: str, overrides: dict,
               rows: list[dict]) -> dict:
    values = [float(row["validation_f1"]) for row in rows]
    return {"name": name, "factor": factor, "overrides": overrides,
            "mean_f1": float(np.mean(values)),
            "sd_f1": float(np.std(values, ddof=1)),
            "per_seed_f1": values,
            "per_seed_auc": [float(row["validation_roc_auc"]) for row in rows],
            "parameter_count": rows[0]["parameter_count"]}


def _passes(candidate: dict, base: dict) -> bool:
    difference = float(candidate["mean_f1"]) - float(base["mean_f1"])
    return difference > float(base["sd_f1"]) and difference >= 0.01


def _compose_accepted_changes(accepted: list[dict], variants: list[dict]) -> tuple[dict, float, float]:
    """Combine accepted Stage C factor changes, including width x depth composition.

    Width and depth both affect the channel tuple in ``DroneCNN``. They are
    separate experimental factors, so their combination scales the selected
    depth pattern by the selected width ratio instead of treating the two
    channel tuples as incompatible overrides.
    """
    selected_specs = [next(spec for spec in variants
                           if spec["name"] == candidate["name"])
                      for candidate in accepted]
    by_factor = {spec["factor"]: spec for spec in selected_specs}
    merged_model: dict = {}
    merged_lr, merged_wd = 1e-3, 1e-2

    width_spec = by_factor.get("channels")
    depth_spec = by_factor.get("depth")
    for spec in selected_specs:
        if "channels" not in spec.get("model", {}):
            merged_model.update(spec.get("model", {}))
        elif spec["factor"] not in {"channels", "depth"}:
            raise RuntimeError(f"unexpected channel override in factor {spec['factor']}")
        if "learning_rate" in spec:
            merged_lr = float(spec["learning_rate"])
        if "weight_decay" in spec:
            merged_wd = float(spec["weight_decay"])

    if width_spec is not None and depth_spec is not None:
        width_scales = {"widths_8_16_32": 0.5, "widths_32_64_128": 2.0}
        depth_patterns = {"depth_2": (16, 32),
                          "depth_4": (16, 32, 64, 64)}
        try:
            scale = width_scales[width_spec["name"]]
            pattern = depth_patterns[depth_spec["name"]]
        except KeyError as exc:
            raise RuntimeError(f"cannot compose width/depth factors for {exc.args[0]}") from exc
        merged_model["channels"] = tuple(int(round(channel * scale))
                                          for channel in pattern)
    elif width_spec is not None:
        merged_model["channels"] = tuple(width_spec["model"]["channels"])
    elif depth_spec is not None:
        merged_model["channels"] = tuple(depth_spec["model"]["channels"])

    return merged_model, merged_lr, merged_wd


def main() -> int:
    if os.cpu_count() != 4:
        raise RuntimeError(f"registered CNN runs expect four logical CPUs; found {os.cpu_count()}")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    config = load_config(ROOT / "config.yaml")
    if config.mode != "global_stats" or params_hash(config) != EXPECTED_FRONTEND_HASH:
        raise RuntimeError("Stage C currently requires the retained default global_stats front end")
    parameter_counts = assert_parameter_counts((1, config.n_mels, config.n_win))
    b2 = json.loads((ROOT / "results/tables/run_report_30_local_b2.json").read_text())
    if b2.get("status") != "PASS" or b2.get("selection", {}).get("accepted"):
        raise RuntimeError("Stage C requires completed B2 with no accepted front-end variant")
    b3_path = ROOT / "results/tables/run_report_30_local_b3.json"
    if not b3_path.is_file():
        raise FileNotFoundError("Stage C requires the completed local B3 control report")
    b3 = json.loads(b3_path.read_text(encoding="utf-8"))
    if b3.get("status") != "PASS" or b3.get("test_reads") != 0:
        raise RuntimeError("Stage C requires a passing B3 control with zero test reads")
    frozen_path = ROOT / "results/tables/cnn_frontend_frozen.json"
    if not frozen_path.is_file():
        raise FileNotFoundError("Stage C requires results/tables/cnn_frontend_frozen.json")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("mode") != "global_stats" or frozen.get("params_hash") != EXPECTED_FRONTEND_HASH:
        raise RuntimeError("frozen CNN front-end record does not match B2 winner")
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
    if not np.array_equal(train_manifest.clip_id.to_numpy(), train_index.clip_id.to_numpy()):
        raise RuntimeError("Stage C train manifest/cache order mismatch")
    if not np.array_equal(val_manifest.clip_id.to_numpy(), val_index.clip_id.to_numpy()):
        raise RuntimeError("Stage C validation manifest/cache order mismatch")
    eligible_count = int(train_index.stats_eligible.astype(bool).sum())
    if eligible_count != 7519:
        raise RuntimeError(f"Stage C stored stats_eligible count mismatch: {eligible_count}")
    with np.load(stats_path, allow_pickle=False) as archive:
        stats = {"mean": archive["mean"].astype(np.float32),
                 "std": archive["std"].astype(np.float32),
                 "clips_excluded": int(archive["clips_excluded"])}
    lineage_hash = config_hash(config, stats)
    if lineage_hash != EXPECTED_STATS_HASH or stats["clips_excluded"] != 88:
        raise RuntimeError("Stage C normalization statistics differ from frozen baseline")
    val_power = np.stack([get_window(val_cache, row, config, split="val")
                          for row in val_index.itertuples(index=False)]).astype(np.float32)
    val_X = _batch_transform(val_power, config, stats)
    y_train = train_manifest.label.to_numpy(dtype=np.float32)
    y_val = val_manifest.label.to_numpy(dtype=np.int8)
    b1 = json.loads((ROOT / "results/tables/run_report_30_local_b1.json").read_text())
    base_rows = b1["mode_fit_results"]["global_stats"]
    base_id = base_rows[0]["run_id"].rsplit("_seed", 1)[0]
    base_seed_rows = []
    for seed in SEEDS:
        model = _load_a0_state(seed, base_id).eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(val_X)).cpu().numpy().astype(np.float64)
        selected = choose_f1_threshold(y_val, logits)
        base_seed_rows.append({"seed": seed, "f1": float(selected["f1"]),
                               "roc_auc": float(roc_auc_score(y_val, logits)),
                               "threshold": float(selected["threshold"])})
    baseline = {"name": "A0_default", "mean_f1": float(np.mean([x["f1"] for x in base_seed_rows])),
                "sd_f1": float(np.std([x["f1"] for x in base_seed_rows], ddof=1)),
                "per_seed_f1": [x["f1"] for x in base_seed_rows],
                "per_seed_auc": [x["roc_auc"] for x in base_seed_rows],
                "per_seed_thresholds": [x["threshold"] for x in base_seed_rows],
                "parameter_count": parameter_count(_load_a0_state(42, base_id))}
    variant_rows, candidates = [], []
    for spec in VARIANTS:
        model_args = spec.get("model", {})
        optimizer = {"learning_rate": float(spec.get("learning_rate", 1e-3)),
                     "weight_decay": float(spec.get("weight_decay", 1e-2))}
        seed_rows = []
        for seed in SEEDS:
            print(json.dumps({"event": "fit_start", "stage": "C",
                              "variant": spec["name"], "seed": seed}), flush=True)
            seed_rows.append(_fit_seed(spec["name"], seed, model_args,
                                       optimizer["learning_rate"], optimizer["weight_decay"],
                                       config, stats, train_index, train_cache,
                                       y_train, val_X, y_val, lineage_hash))
        candidate = _summarize(spec["name"], spec["factor"],
                               {**model_args, **optimizer}, seed_rows)
        candidate["accepted_vs_default"] = _passes(candidate, baseline)
        candidate["seed_runs"] = [{k: v for k, v in row.items()
                                   if k not in {"validation_scores"}} for row in seed_rows]
        candidates.append(candidate)
        variant_rows.append({"spec": spec, "candidate": candidate})

    accepted_by_factor = []
    factors = sorted({spec["factor"] for spec in VARIANTS})
    for factor in factors:
        eligible = [row["candidate"] for row in variant_rows
                    if row["candidate"]["factor"] == factor and
                    row["candidate"]["accepted_vs_default"]]
        if eligible:
            accepted_by_factor.append(max(eligible, key=lambda row: row["mean_f1"]))
    combination = None
    winner = baseline
    if len(accepted_by_factor) == 1:
        winner = accepted_by_factor[0]
    elif len(accepted_by_factor) > 1:
        merged_model, merged_lr, merged_wd = _compose_accepted_changes(
            accepted_by_factor, VARIANTS)
        combo_name = "combination_" + "_".join(x["name"] for x in accepted_by_factor)
        combo_rows = []
        for seed in SEEDS:
            print(json.dumps({"event": "fit_start", "stage": "C",
                              "variant": combo_name, "seed": seed}), flush=True)
            combo_rows.append(_fit_seed(combo_name, seed, merged_model, merged_lr,
                                        merged_wd, config, stats, train_index,
                                        train_cache, y_train, val_X, y_val, lineage_hash))
        combo = _summarize(combo_name, "combination", {**merged_model,
                                "learning_rate": merged_lr, "weight_decay": merged_wd}, combo_rows)
        best_single = max(accepted_by_factor, key=lambda row: row["mean_f1"])
        difference = combo["mean_f1"] - best_single["mean_f1"]
        combo["passes_registered_rule"] = difference > best_single["sd_f1"] and difference >= 0.01
        combination = {"candidate": combo,
                       "accepted_factor_changes": accepted_by_factor,
                       "best_single": best_single["name"],
                       "difference_vs_best_single": difference,
                       "status": "accepted" if combo["passes_registered_rule"] else "rejected"}
        winner = combo if combo["passes_registered_rule"] else best_single

    report = {"stage": "C", "status": "PASS", "test_reads": 0,
              "frontend_params_hash": params_hash(config),
              "statistics_lineage_hash": lineage_hash,
              "stats_eligible_train_clips": eligible_count, "excluded_train_clips": 88,
              "baseline": baseline, "baseline_seed_results": base_seed_rows,
              "rule": "candidate accepted iff mean validation F1 improvement over A0 exceeds A0 ddof=1 SD and is at least 0.01",
              "variants": candidates, "accepted_by_factor": accepted_by_factor,
              "combination": combination, "winner": winner,
              "asserted_parameter_counts": parameter_counts,
              "device": "cpu", "torch_threads": 4,
              "training_data": "stored train cache only; validation used for early stopping and selection"}
    path, _ = write_run_report(ROOT, "30_local_stage_c", report)
    print(json.dumps({"event": "stage_c_complete", "winner": winner["name"],
                      "accepted_by_factor": [x["name"] for x in accepted_by_factor],
                      "report": str(path)}, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
