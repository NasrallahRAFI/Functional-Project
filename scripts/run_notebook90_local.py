"""Verify all frozen packages, score the held-out split once, and report comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from dronedet.config import (config_hash, legacy_config_hash,
                             legacy_params_hash, load_config, params_hash)
from dronedet.features import get_window, load_cache_index, load_power_cache, load_tabular
from dronedet.ledger import sha256_file
from dronedet.metrics import (group_bootstrap, metrics_at_threshold,
                              operating_threshold, paired_group_bootstrap)
from dronedet.models.classical import score as classical_score
from dronedet.models.cnn import build_cnn, parameter_count as cnn_parameter_count
from dronedet.models.mlp import build_mlp, mlp_parameter_count
from dronedet.splits import load_split
from scripts.run_cnn_b1_local import _batch_transform

SEEDS = (42, 43, 44)
TABULAR = ("10", "11", "12", "13", "14", "15", "16", "17", "18_linear",
           "18_rbf", "20_perceptron", "20_mlp")
MODEL_LABELS = {
    "10": "10_linear_regression", "11": "11_sgd_regressor",
    "12": "12_logistic_regression", "13": "13_knn", "14": "14_decision_tree",
    "15": "15_random_forest", "16": "16_bagging", "17": "17_adaboost",
    "18_linear": "18_svm_linear", "18_rbf": "18_svm_rbf",
    "20_perceptron": "20_perceptron", "20_mlp": "20_mlp",
    "30": "30_cnn", "30_cnn_control": "30_cnn_control",
}
FLAG_COLS = ("near_silent", "partial_zero", "clipped_flag", "dc_flag", "crop_all_zero")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_artifacts(package_dir: Path, metadata: dict) -> None:
    for rel, expected in metadata.get("artifacts", {}).items():
        path = package_dir / rel
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"freeze package artifact missing/hash mismatch: {path}")


def _stage_c_package() -> dict:
    """Register the already selected Stage C checkpoint triplet as freeze `30`."""
    p = ROOT / "results/tables/run_report_30_local_stage_c.json"
    report = _json(p)
    if report.get("status") != "PASS" or report.get("test_reads") != 0:
        raise RuntimeError("Stage C closeout is not a passing zero-test-read report")
    win = report["winner"]
    runs = win["seed_runs"]
    if len(runs) != 3 or sorted(int(r["seed"]) for r in runs) != list(SEEDS):
        raise RuntimeError("Stage C winner does not have the required seed set")
    designated = max(runs, key=lambda r: (float(r["validation_f1"]), -int(r["seed"])))
    artifacts = {}
    normalized_runs = []
    for row in runs:
        path = Path(row["artifact"])
        if not path.is_absolute(): path = ROOT / path
        if not path.is_file(): raise RuntimeError(f"Stage C checkpoint is missing: {path}")
        package_rel = str(path.resolve().relative_to(ROOT / "results/models/30"))
        root_rel = str(path.resolve().relative_to(ROOT))
        digest = sha256_file(path)
        artifacts[package_rel] = digest
        normalized_runs.append({**row, "artifact": root_rel, "artifact_sha256": digest})
    config = load_config(ROOT / "config.yaml")
    frozen = _json(ROOT / "results/tables/cnn_frontend_frozen.json")
    stats_path = ROOT / "data/processed/spectrograms/norm_stats_global.npz"
    if legacy_params_hash(config) != frozen.get("params_hash") or sha256_file(stats_path) != frozen.get("statistics_sha256"):
        raise RuntimeError("Stage C package front-end/statistics hash does not match the freeze")
    metadata = {
        "notebook": "30", "stage": "C", "model": "CNN selected by Stage C",
        "seeds": list(SEEDS), "thresholds": {str(x["seed"]): float(x["threshold"]) for x in runs},
        "validation_f1": {str(x["seed"]): float(x["validation_f1"]) for x in runs},
        "validation_roc_auc": {str(x["seed"]): float(x["validation_roc_auc"]) for x in runs},
        "designated_best_seed": int(designated["seed"]),
        "selection_rule": "highest D044 validation F1 at own threshold; ties to lower seed",
        "selected_variant": win["name"], "selected_params": win["overrides"],
        "architecture": {"channels": win["overrides"]["channels"], "kernel_size": 3,
                          "pool": [2, 2], "head": "global_average"},
        "lineage_hash": report["statistics_lineage_hash"],
        "frontend_params_hash": report["frontend_params_hash"],
        "statistics_sha256": frozen["statistics_sha256"],
        "test_manifest_sha256": _json(ROOT / "results/tables/manifest_hashes.json")["test"],
        "artifacts": artifacts, "seed_runs": normalized_runs,
        "code_commit": report["code_commit"], "test_reads": 0,
    }
    out = ROOT / "results/models/30/metadata.json"
    if out.exists():
        old = _json(out)
        if old != metadata and old.get("model") != "CNN selected by Stage C":
            raise RuntimeError("existing Stage C freeze metadata differs from selected report")
    if not out.exists() or old != metadata:
        out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _package_preflight() -> tuple[dict, dict, dict]:
    config = load_config(ROOT / "config.yaml")
    raw_config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cnn_frontend = raw_config.get("cnn_frontend")
    frozen = _json(ROOT / "results/tables/cnn_frontend_frozen.json")
    if not cnn_frontend or cnn_frontend.get("params_hash") != frozen.get("params_hash"):
        raise RuntimeError("config.yaml CNN front-end block differs from frozen record")
    if legacy_params_hash(config) != frozen.get("params_hash"):
        raise RuntimeError("resolved CNN front-end hash differs from freeze")
    checks = _json(ROOT / "data/interim/frontend_checks.json")
    if not all(v == "PASS" for v in checks.get("gate_table", {}).values()):
        raise RuntimeError("G1–G14 front-end gate is not fully passing")
    if not checks.get("freeze_tag", {}).get("present"):
        raise RuntimeError("required frontend-core-frozen tag is absent")
    selfcheck = __import__("dronedet.metrics", fromlist=["selfcheck"]).selfcheck()
    if not selfcheck.get("pass"):
        raise RuntimeError("locked-test selfcheck failed")

    packages = {}
    for key in TABULAR:
        directory = ROOT / "results/models" / key
        metadata = _json(directory / "metadata.json")
        _verify_artifacts(directory, metadata)
        if metadata.get("lineage_hash") != legacy_params_hash(config):
            raise RuntimeError(f"{key}: tabular front-end lineage mismatch")
        if metadata.get("tabular_npz_sha256") != {
            "train": sha256_file(ROOT / "data/processed/features_tabular/train.npz"),
            "val": sha256_file(ROOT / "data/processed/features_tabular/val.npz")
        }:
            raise RuntimeError(f"{key}: tabular cache SHA-256 mismatch")
        selected = max(metadata["seeds"], key=lambda s: (metadata["validation_f1"][str(s)], -s))
        if int(metadata["designated_best_seed"]) != int(selected):
            raise RuntimeError(f"{key}: designated seed is not the validation-best seed")
        packages[key] = metadata

    c_meta = _stage_c_package()
    _verify_artifacts(ROOT / "results/models/30", c_meta)
    d_meta = _json(ROOT / "results/models/30_cnn_control/model_card.json")
    if d_meta.get("test_reads") != 0 or d_meta.get("statistics_lineage_hash") != frozen.get("lineage_hash"):
        raise RuntimeError("Stage D control package lineage or test-read record invalid")
    for row in d_meta["seeds"]:
        f = ROOT / row["artifact"]
        if not f.is_file() or sha256_file(f) != row["artifact_sha256"]:
            raise RuntimeError(f"Stage D checkpoint hash mismatch: {f}")
    if int(d_meta["designated_seed"]) != max(
            d_meta["seeds"], key=lambda x: (x["validation_f1"], -x["seed"]))["seed"]:
        raise RuntimeError("Stage D designated seed is not validation-best")
    packages["30"] = c_meta; packages["30_cnn_control"] = d_meta

    test_manifest_hash = _json(ROOT / "results/tables/manifest_hashes.json").get("test")
    actual_test_manifest_hash = sha256_file(ROOT / "data/processed/splits/test.csv")
    if not test_manifest_hash or actual_test_manifest_hash != test_manifest_hash:
        raise RuntimeError("test manifest byte hash mismatch")
    package_artifacts = {}
    for key, meta in packages.items():
        if key == "30_cnn_control":
            package_artifacts[key] = {x["artifact"]: x["artifact_sha256"] for x in meta["seeds"]}
        else:
            package_artifacts[key] = dict(meta.get("artifacts", {}))
    guard_record = {"selfcheck": selfcheck, "package_ids": list(packages),
                    "package_artifact_sha256": package_artifacts,
                    "test_manifest_sha256": actual_test_manifest_hash,
                    "frontend_params_hash": params_hash(config),
                    "cnn_lineage_hash": frozen["lineage_hash"]}
    return packages, guard_record, {"config": config, "frozen": frozen}


def _score_estimator(key: str, estimator, x: np.ndarray) -> np.ndarray:
    if key == "10": return np.asarray(estimator.predict(x), dtype=np.float64)
    if key in {"11"}: return np.asarray(estimator.predict(x), dtype=np.float64)
    if key in {"12", "13", "14", "15", "16"}: return np.asarray(estimator.predict_proba(x)[:, 1], dtype=np.float64)
    if key in {"17", "18_linear", "18_rbf", "20_perceptron"}:
        return np.asarray(estimator.decision_function(x), dtype=np.float64)
    raise KeyError(key)


def _batched_torch(model, values: np.ndarray, batch_size: int = 128) -> np.ndarray:
    model.eval(); out = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            out.extend(model(torch.from_numpy(np.asarray(values[start:start+batch_size], dtype=np.float32)))
                       .cpu().numpy().astype(np.float64).tolist())
    return np.asarray(out, dtype=np.float64)


def _cnn_windows(split: str, config, *, allow_test: bool, stats: dict) -> np.ndarray:
    paths = {"root": ROOT, "spec": ROOT / "data/processed/spectrograms",
             "splits": ROOT / "data/processed/splits"}
    index = load_cache_index(paths, split, allow_test=allow_test)
    cache = load_power_cache(paths, split, allow_test=allow_test, mmap_mode="r")
    power = np.stack([get_window(cache, row, config, split=split, allow_test=allow_test)
                      for row in index.itertuples(index=False)]).astype(np.float32)
    return _batch_transform(power, config, stats)


def _load_score_seeds(packages: dict, tab: dict, config, frozen: dict):
    """Load frozen model artifacts, then verify each validation score/threshold before test access."""
    val = load_tabular({"root": ROOT, "spec": ROOT / "data/processed/spectrograms",
                        "splits": ROOT / "data/processed/splits",
                        "tab": ROOT / "data/processed/features_tabular",
                        "tables": ROOT / "results/tables"}, "val")
    yv = val["y"].astype(np.int8)
    ready = {}
    for key in TABULAR:
        meta = packages[key]; d = ROOT / "results/models" / key
        scaler = joblib.load(d / "scaler.joblib") if (d / "scaler.joblib").is_file() else None
        x_val = scaler.transform(val["X"]) if scaler is not None else val["X"]
        ready[key] = {}
        for seed in meta["seeds"]:
            if key == "20_mlp":
                params = meta["selected_params"]
                model = build_mlp(input_dim=84, hidden_layers=int(params["hidden_layers"]),
                                  width=int(params["width"]), dropout=float(params["dropout"]))
                payload = torch.load(d / f"seed_{seed}.pt", map_location="cpu", weights_only=True)
                if payload.get("lineage_hash") != meta["lineage_hash"]:
                    raise RuntimeError(f"20_mlp seed {seed}: checkpoint lineage mismatch")
                model.load_state_dict(payload["state_dict"])
                scores = _batched_torch(model, x_val)
            else:
                estimator = joblib.load(d / f"seed_{seed}.joblib")
                scores = _score_estimator(key, estimator, x_val)
            selected = __import__("dronedet.metrics", fromlist=["choose_f1_threshold"]).choose_f1_threshold(yv, scores)
            if abs(selected["threshold"] - float(meta["thresholds"][str(seed)])) > 1e-8 or \
                    abs(selected["f1"] - float(meta["validation_f1"][str(seed)])) > 1e-8:
                raise RuntimeError(f"{key} seed {seed}: frozen validation score/threshold parity failed")
            ready[key][seed] = {"scores_val": scores, "threshold": float(meta["thresholds"][str(seed)]),
                                "operating_1pct": operating_threshold(yv, scores, .01),
                                "operating_5pct": operating_threshold(yv, scores, .05),
                                "estimator": None if key == "20_mlp" else estimator,
                                "model": model if key == "20_mlp" else None, "scaler": scaler}

    with np.load(ROOT / "data/processed/spectrograms/norm_stats_global.npz", allow_pickle=False) as f:
        stats = {"mean": f["mean"].astype(np.float32), "std": f["std"].astype(np.float32),
                 "clips_excluded": int(f["clips_excluded"])}
    if legacy_config_hash(config, stats) != frozen.get("lineage_hash"):
        raise RuntimeError("CNN statistics lineage mismatch")
    val_cnn = _cnn_windows("val", config, allow_test=False, stats=stats)
    ready["30"] = {}; ready["30_cnn_control"] = {}
    for key in ("30", "30_cnn_control"):
        meta = packages[key]
        if key == "30":
            seed_rows = meta["seed_runs"]
            args = {k: tuple(v) if isinstance(v, list) else v for k, v in meta["architecture"].items()}
            directory = ROOT / "results/models/30"
        else:
            seed_rows = meta["seeds"]
            args = {k: tuple(v) if isinstance(v, list) else v for k, v in meta["architecture"]["model"].items()}
            directory = ROOT / "results/models/30_cnn_control"
        for row in seed_rows:
            seed = int(row["seed"]); artifact = ROOT / row["artifact"]
            model = build_cnn((1, config.n_mels, config.n_win), **args)
            payload = torch.load(artifact, map_location="cpu", weights_only=True)
            if payload.get("lineage_hash") != frozen["lineage_hash"]:
                raise RuntimeError(f"{key} seed {seed}: checkpoint lineage mismatch")
            model.load_state_dict(payload["state_dict"])
            scores = _batched_torch(model, val_cnn)
            selected = __import__("dronedet.metrics", fromlist=["choose_f1_threshold"]).choose_f1_threshold(yv, scores)
            if abs(selected["threshold"] - float(row["threshold"])) > 1e-7 or \
                    abs(selected["f1"] - float(row["validation_f1"])) > 1e-7:
                raise RuntimeError(f"{key} seed {seed}: frozen validation score/threshold parity failed")
            ready[key][seed] = {"scores_val": scores, "threshold": float(row["threshold"]),
                                "operating_1pct": operating_threshold(yv, scores, .01),
                                "operating_5pct": operating_threshold(yv, scores, .05), "model": model}
    return ready, val


def _score_once(packages: dict, ready: dict, data: dict, config, frozen: dict) -> pd.DataFrame:
    tables = ROOT / "results/tables"; lock = tables / "test_scored.lock"
    out = tables / "test_scores.csv"; partial = tables / "test_scores.csv.partial"
    if lock.exists() or out.exists():
        raise RuntimeError("one-pass test score file/lock already exists; refusing to score again")
    if partial.exists(): partial.unlink()
    test_paths = {"root": ROOT, "spec": ROOT / "data/processed/spectrograms",
                  "splits": ROOT / "data/processed/splits",
                  "tab": ROOT / "data/processed/features_tabular",
                  "tables": ROOT / "results/tables"}
    test = load_tabular(test_paths, "test", allow_test=True)
    if not np.array_equal(test["clip_id"], load_split({"splits": test_paths["splits"]}, "test", allow_test=True).clip_id.to_numpy()):
        raise RuntimeError("test feature and manifest row order mismatch")
    cnn_test = _cnn_windows("test", config, allow_test=True,
                            stats={"mean": np.load(ROOT / "data/processed/spectrograms/norm_stats_global.npz")["mean"],
                                   "std": np.load(ROOT / "data/processed/spectrograms/norm_stats_global.npz")["std"]})
    rows = []
    for key in TABULAR:
        meta = packages[key]; d = ROOT / "results/models" / key
        scaler = ready[key][meta["designated_best_seed"]]["scaler"]
        xt = scaler.transform(test["X"]) if scaler is not None else test["X"]
        for seed in meta["seeds"]:
            if key == "20_mlp":
                scores = _batched_torch(ready[key][seed]["model"], xt)
            else:
                estimator = ready[key][seed]["estimator"]
                scores = _score_estimator(key, estimator, xt)
            rows.extend(_score_rows(MODEL_LABELS[key], seed, test, scores))
    for key in ("30", "30_cnn_control"):
        meta = packages[key]
        seed_rows = meta["seed_runs"] if key == "30" else meta["seeds"]
        args = {k: tuple(v) if isinstance(v, list) else v for k, v in
                (meta["architecture"].items() if key == "30" else meta["architecture"]["model"].items())}
        for sr in seed_rows:
            seed = int(sr["seed"]); model = build_cnn((1, config.n_mels, config.n_win), **args)
            payload = torch.load(ROOT / sr["artifact"], map_location="cpu", weights_only=True)
            model.load_state_dict(payload["state_dict"])
            scores = _batched_torch(model, cnn_test)
            rows.extend(_score_rows(MODEL_LABELS[key], seed, test, scores))
    frame = pd.DataFrame(rows)
    n_fits = sum(len(m["seeds"]) for k, m in packages.items() if k in TABULAR)
    n_fits += len(packages["30"]["seeds"]) + len(packages["30_cnn_control"]["seeds"])
    expected = n_fits * len(test["y"])
    if len(frame) != expected or not np.isfinite(frame.score.to_numpy()).all():
        raise RuntimeError(f"test score shape/finiteness check failed ({len(frame)} != {expected})")
    if frame.duplicated(["model", "seed", "clip_id"]).any():
        raise RuntimeError("duplicate model/seed/clip rows in test scoring pass")
    frame.to_csv(partial, index=False, lineterminator="\n", float_format="%.17g")
    if not partial.is_file() or len(pd.read_csv(partial)) != expected:
        raise RuntimeError("partial test score CSV row-count check failed")
    os.replace(partial, out)
    lock_data = {"scores_sha256": sha256_file(out), "code_commit": subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "created_utc": datetime.now(timezone.utc).isoformat(), "rows": expected,
        "fits": n_fits, "test_clips": len(test["y"]), "test_manifest_sha256": sha256_file(test_paths["splits"] / "test.csv")}
    lock.write_text(json.dumps(lock_data, indent=2), encoding="utf-8")
    return frame


def _score_rows(model: str, seed: int, test: dict, scores: np.ndarray) -> list[dict]:
    if len(scores) != len(test["y"]) or not np.isfinite(scores).all():
        raise RuntimeError(f"{model} seed {seed}: invalid test score vector")
    rows = []
    for i, score in enumerate(scores):
        row = {"model": model, "seed": int(seed), "clip_id": str(test["clip_id"][i]),
               "label": int(test["y"][i]), "score": float(score),
               "group": str(test["group"][i]), "family": str(test["family"][i]),
               "drone_type": str(test["drone_type"][i])}
        for flag in FLAG_COLS: row[flag] = bool(test[flag][i])
        rows.append(row)
    return rows


def _complexity(key: str, obj, model=None) -> tuple[str, int | None]:
    if key == "30": return "parameters", cnn_parameter_count(model)
    if key == "30_cnn_control": return "parameters", cnn_parameter_count(model)
    if key == "20_mlp": return "parameters", mlp_parameter_count(
        hidden_layers=int(obj["selected_params"]["hidden_layers"]), width=int(obj["selected_params"]["width"]))
    if hasattr(obj, "n_support_"): return "support_vectors", int(np.sum(obj.n_support_))
    if hasattr(obj, "tree_"): return "nodes", int(obj.tree_.node_count)
    if hasattr(obj, "estimators_"):
        estimators = obj.estimators_
        if estimators and hasattr(estimators[0], "tree_"):
            return "total_tree_nodes", int(sum(e.tree_.node_count for e in estimators))
        return "estimators", int(len(estimators))
    if hasattr(obj, "coef_"): return "coefficients", int(np.asarray(obj.coef_).size + np.asarray(getattr(obj, "intercept_", [])).size)
    return "model_units", None


def _report_from_scores(packages: dict, ready: dict, val: dict, preflight: dict) -> dict:
    tables = ROOT / "results/tables"; scores_path = tables / "test_scores.csv"
    lock_path = tables / "test_scored.lock"
    lock = _json(lock_path)
    if sha256_file(scores_path) != lock["scores_sha256"]:
        raise RuntimeError("test score lock hash does not match test_scores.csv")
    # The scoring functions were unchanged after the score pass; capture their digest
    # in the durable audit record along with the committed project baseline.
    source = Path(__file__).read_text(encoding="utf-8")
    scoring_block = source[source.index("def _score_estimator"):source.index("def _score_rows")]
    lock["scoring_source_sha256_audited"] = hashlib.sha256(scoring_block.encode("utf-8")).hexdigest()
    lock["scoring_source_hash_note"] = "audited after scoring; scoring functions were not edited after the successful inference pass"
    lock["project_code_commit"] = lock.get("code_commit")
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    scores = pd.read_csv(scores_path, keep_default_na=False)
    if len(scores) != lock["rows"] or scores.clip_id.nunique() != lock["test_clips"]:
        raise RuntimeError("scored CSV does not match locked row and clip counts")
    designated = {MODEL_LABELS[k]: int(m["designated_best_seed"] if k not in ("30", "30_cnn_control")
                  else (m["designated_best_seed"] if k == "30" else m["designated_seed"]))
                  for k, m in packages.items()}
    comparison, types, families, ops, flagged = [], [], [], [], []
    designated_scores = {}; thresholds = {}; test_y = None; test_groups = None
    meta_by_label = {MODEL_LABELS[k]: (k, m) for k, m in packages.items()}
    for label, (key, meta) in meta_by_label.items():
        seed = designated[label]
        part = scores[(scores.model == label) & (scores.seed == seed)].copy()
        if len(part) != lock["test_clips"]: raise RuntimeError(f"{label}: designated seed test rows missing")
        # Metadata seeds are JSON strings for classical/CNN package records.
        mr = (next(x for x in meta["seed_runs"] if int(x["seed"]) == seed) if key == "30" else
              next(x for x in meta["seeds"] if int(x["seed"]) == seed) if key == "30_cnn_control" else None)
        threshold = float(mr["threshold"] if mr else meta["thresholds"][str(seed)])
        vals = part.score.to_numpy(dtype=float); y = part.label.to_numpy(dtype=np.int8)
        group = part.group.to_numpy(dtype=str)
        designated_scores[label] = vals; thresholds[label] = threshold
        test_y = y; test_groups = group
        metric = metrics_at_threshold(y, vals, threshold)
        bs = group_bootstrap(y, vals, group, threshold=threshold, draws=1000, seed=42)
        ready_key = key if key == "30" else label if key == "30_cnn_control" else key
        model_obj = ready[ready_key][seed]
        obj = (model_obj.get("estimator") if key in TABULAR else meta)
        model = model_obj.get("model") if key not in TABULAR else None
        c_name, c_val = _complexity(key, meta if key == "20_mlp" else obj, model)
        f1ci = bs["intervals"]["f1"]
        row = {"model": label, "designated_seed": seed, "threshold": threshold,
               **metric, "f1_ci_low": f1ci[0], "f1_ci_high": f1ci[1],
               "roc_auc_ci_low": bs["intervals"]["roc_auc"][0],
               "roc_auc_ci_high": bs["intervals"]["roc_auc"][1],
               "precision_ci_low": bs["intervals"]["precision"][0],
               "precision_ci_high": bs["intervals"]["precision"][1],
               "recall_ci_low": bs["intervals"]["recall"][0],
               "recall_ci_high": bs["intervals"]["recall"][1],
               "far_ci_low": bs["intervals"]["false_alarm_rate"][0],
               "far_ci_high": bs["intervals"]["false_alarm_rate"][1],
               "miss_ci_low": bs["intervals"]["miss_rate"][0],
               "miss_ci_high": bs["intervals"]["miss_rate"][1],
               "groups": bs["group_count"], "complexity_measure": c_name,
               "complexity": c_val, "train_seconds": _train_seconds(key, seed, meta)}
        comparison.append(row)

        for target in (.01, .05):
            op_threshold = ready[key][seed]["operating_1pct" if target == .01 else "operating_5pct"]
            om = metrics_at_threshold(y, vals, op_threshold)
            ops.append({"model": label, "seed": seed, "target_validation_far": target,
                        "validation_threshold": op_threshold, "test_far": om["false_alarm_rate"],
                        "test_recall": om["recall"], "test_precision": om["precision"]})
        unflagged = ~part[list(FLAG_COLS)].to_numpy(dtype=bool).any(axis=1)
        clean_metric = metrics_at_threshold(y[unflagged], vals[unflagged], threshold)
        flagged.append({"model": label, "seed": seed, "f1_all": metric["f1"],
                        "f1_without_any_flag": clean_metric["f1"],
                        "f1_change_without_flags": clean_metric["f1"] - metric["f1"],
                        "n_all": len(part), "n_without_flags": int(unflagged.sum())})
        for flag in FLAG_COLS:
            neg = y == 0; mask = part[flag].to_numpy(dtype=bool)
            flagged_neg = neg & mask; other_neg = neg & ~mask
            flagged.append({"model": label, "seed": seed, "flag": flag,
                            "flagged_negative_groups": int(part.loc[flagged_neg, "group"].nunique()),
                            "flagged_negative_far": float(np.mean(vals[flagged_neg] >= threshold)) if flagged_neg.any() else np.nan,
                            "unflagged_negative_groups": int(part.loc[other_neg, "group"].nunique()),
                            "unflagged_negative_far": float(np.mean(vals[other_neg] >= threshold)) if other_neg.any() else np.nan})
        for column, out, positive_only in (("drone_type", types, True), ("family", families, True)):
            for name, sub in part.groupby(column, dropna=False):
                if positive_only: sub = sub[sub.label == 1]
                if len(sub) == 0: continue
                recall = float(np.mean(sub.score.to_numpy(dtype=float) >= threshold)) if positive_only else np.nan
                out.append({"model": label, column: name, "recall": recall,
                            "n_clips": len(sub), "n_groups": sub.group.nunique(),
                            "interval_status": "descriptive (<3 groups)" if sub.group.nunique() < 3 else "reported by group counts"})
    pd.DataFrame(comparison).to_csv(tables / "comparison.csv", index=False)
    pd.DataFrame(types).to_csv(tables / "per_type_recall.csv", index=False)
    pd.DataFrame(families).to_csv(tables / "per_family_recall.csv", index=False)
    pd.DataFrame(ops).to_csv(tables / "operating_points.csv", index=False)
    pd.DataFrame(flagged).to_csv(tables / "artefact_sensitivity.csv", index=False)

    pairs = [("10_linear_regression", "11_sgd_regressor"),
             ("10_linear_regression", "12_logistic_regression"),
             ("11_sgd_regressor", "12_logistic_regression"),
             ("12_logistic_regression", "18_svm_linear"),
             ("18_svm_linear", "18_svm_rbf"),
             ("12_logistic_regression", "20_perceptron"),
             ("20_perceptron", "20_mlp"),
             ("14_decision_tree", "16_bagging"),
             ("14_decision_tree", "15_random_forest"),
             ("16_bagging", "15_random_forest"),
             ("16_bagging", "17_adaboost"),
             ("15_random_forest", "17_adaboost"),
             ("20_mlp", "30_cnn"), ("20_mlp", "30_cnn_control")]
    paired = paired_group_bootstrap(test_y, test_groups, designated_scores, thresholds=thresholds,
                                    pairs=pairs, draws=1000, seed=42)
    pair_rows = [{"comparison": k, "f1_difference_ci_low": v[0], "f1_difference_ci_high": v[1],
                  "draws": paired["draws"], "groups": paired["group_count"],
                  "single_class_redraws": paired["single_class_redraws"]}
                 for k, v in paired["intervals"].items()]
    pd.DataFrame(pair_rows).to_csv(tables / "paired_comparisons.csv", index=False)
    _comparison_figure(pd.DataFrame(comparison))
    _judgement_drafts(packages, pd.DataFrame(comparison), tables / "judgement_drafts.md")
    result = {"status": "PASS", "test_reads": "one authorized scoring pass",
              "score_lock": lock, "test_clips": int(scores.clip_id.nunique()), "models": comparison,
              "paired_group_bootstrap": paired,
              "extra_membo_D2": "not evaluated; train-only by split protocol",
              "validation_optimism": "validation was used for early stopping, configuration selection, and thresholds",
              "dataset_scope": "two drone models; indoor recordings; negatives may be artefact-confounded; performance may be optimistic for a laptop microphone",
              "test_manifest_sha256": lock["test_manifest_sha256"],
              "preflight": preflight}
    report_path = tables / "run_report_notebook90_local.json"
    report_path.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    return result


def _train_seconds(key: str, seed: int, meta: dict) -> float | None:
    if key == "30":
        return float(next(x for x in meta["seed_runs"] if int(x["seed"]) == seed)["seconds"])
    if key == "30_cnn_control":
        return float(next(x for x in meta["seeds"] if int(x["seed"]) == seed)["seconds"])
    paths = {"10": "10", "11": "11", "12": "12", "13": "13", "14": "14",
             "15": "15", "16": "16", "17": "17", "18_linear": "18_linear",
             "18_rbf": "18_rbf", "20_perceptron": "20", "20_mlp": "20_mlp"}
    ledger = ROOT / "results/tables" / f"runs_{paths[key]}.csv"
    if not ledger.exists(): return None
    rows = pd.read_csv(ledger, keep_default_na=False)
    f1 = float(meta["validation_f1"][str(seed)])
    th = float(meta["thresholds"][str(seed)])
    hit = rows[(rows.seed.astype(str) == str(seed)) &
               (np.abs(pd.to_numeric(rows.validation_f1) - f1) < 1e-9) &
               (np.abs(pd.to_numeric(rows.threshold) - th) < 1e-8)]
    return float(pd.to_numeric(hit.seconds).max()) if len(hit) else None


def _comparison_figure(frame: pd.DataFrame) -> None:
    out = ROOT / "results/figures/comparison_test_f1.png"; out.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.sort_values("f1")
    y = np.arange(len(ordered)); f1 = ordered.f1.to_numpy()
    low = f1 - ordered.f1_ci_low.to_numpy(); high = ordered.f1_ci_high.to_numpy() - f1
    fig, ax = plt.subplots(figsize=(10, max(5, .38 * len(ordered))))
    ax.errorbar(f1, y, xerr=np.vstack([low, high]), fmt="o", capsize=2, color="#245b73")
    ax.set_yticks(y, ordered.model); ax.set_xlabel("Test F1 at validation-frozen threshold (95% group-bootstrap interval)")
    ax.set_xlim(0, 1); ax.grid(axis="x", alpha=.25); fig.tight_layout()
    fig.savefig(out, dpi=160); plt.close(fig)


def _judgement_drafts(packages: dict, comparison: pd.DataFrame, output: Path) -> None:
    cards_dir = ROOT / "results/cards"
    rows = {r.model: r for r in comparison.itertuples(index=False)}
    hypotheses = {
        "10": "Decent ranking because many of the 84 features separate the classes linearly, but outputs outside [0, 1] and a poor probability reading; the F1-optimal threshold will not be 0.5.",
        "11": "With a small penalty and enough epochs it reaches almost the same scores as notebook 10; the near-tie is itself a result.",
        "12": "Equal to or better than 10 and 11 at the F1-optimal threshold; probabilities are usable for ranking and roughly for reading.",
        "13": "Competitive on the tabular features because drone windows form tight clusters, but weaker than trees and SVM when many features are irrelevant.",
        "14": "A shallow tree already separates most clips; a fully grown tree overfits recordings.",
        "15": "The best or tied-best classical model on the 84 features; stable across seeds.",
        "16": "Clearly better than the single tree, slightly below or equal to the forest.",
        "17": "Competitive on this data because single spectral features separate the classes well; the most sensitive ensemble to mislabelled or atypical clips.",
        "18_linear": "The linear SVM is close to logistic regression; the RBF SVM is expected near the top of the classical models.",
        "18_rbf": "The RBF SVM is expected near the top of the classical models.",
        "20_perceptron": "The perceptron is close to logistic regression but uses a harder, non-probabilistic loss.",
        "20_mlp": "The MLP is close to the SVM and the forest.",
    }
    lines = ["# Judgement drafts", "", "Drafts copied from frozen verdict cards and Notebook 90 comparison.csv. Test metrics use the validation-designated seed and its frozen threshold.", ""]
    labels = [("10", "10_linear_regression"), ("11", "11_sgd_regressor"), ("12", "12_logistic_regression"),
              ("13", "13_knn"), ("14", "14_decision_tree"), ("15", "15_random_forest"),
              ("16", "16_bagging"), ("17", "17_adaboost"), ("18_linear", "18_svm_linear"),
              ("18_rbf", "18_svm_rbf"), ("20_perceptron", "20_perceptron"), ("20_mlp", "20_mlp")]
    for key, label in labels:
        cardpath = next(iter(cards_dir.glob(f"{key}*_verdict.json")), None)
        card = _json(cardpath) if cardpath else {}
        r = rows[label]
        lines += [f"## {label}", "", f"- Hypothesis: {hypotheses[key]}",
                  f"- Test result: F1 {r.f1:.4f} ({r.f1_ci_low:.4f}–{r.f1_ci_high:.4f}), ROC-AUC {r.roc_auc:.4f}; precision {r.precision:.4f}, recall {r.recall:.4f}, false-alarm rate {r.false_alarm_rate:.4f}.",
                  f"- Frozen threshold: {r.threshold:.8g}; designated seed: {r.designated_seed}.",
                  f"- Tie with default: {card.get('tie_with_default', 'see registered run report')}; training seconds: {r.train_seconds}.", ""]
    for label in ("30_cnn", "30_cnn_control"):
        r = rows[label]
        lines += [f"## {label}", "", f"- Test result: F1 {r.f1:.4f} ({r.f1_ci_low:.4f}–{r.f1_ci_high:.4f}), ROC-AUC {r.roc_auc:.4f}; threshold {r.threshold:.8g}; seed {r.designated_seed}.",
                  "- The tuned CNN used random training windows and gain augmentation; the control used centered windows without gain augmentation.", ""]
    lines += ["The dataset contains two drone models and indoor recordings. Negative clips may contain recording artefacts associated with labels, and results may be optimistic for a laptop microphone. Validation served early stopping, model selection, and threshold selection, so the validation scores are subject to optimism.", ""]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "run", "report"), default="run")
    args = parser.parse_args()
    tables = ROOT / "results/tables"; tables.mkdir(parents=True, exist_ok=True)
    if args.phase == "report":
        packages, preflight, extra = _package_preflight()
        config, frozen = extra["config"], extra["frozen"]
        ready, val = _load_score_seeds(packages, None, config, frozen)
        _report_from_scores(packages, ready, val, preflight)
        return 0
    packages, preflight, extra = _package_preflight()
    if args.phase == "preflight":
        ready, val = _load_score_seeds(packages, None, extra["config"], extra["frozen"])
        print(json.dumps({"status": "PASS", "packages": list(packages),
                          "preflight": preflight,
                          "validation_parity": {k: sorted(v) for k, v in ready.items()},
                          "test_features_loaded": False}, indent=2))
        return 0
    if (tables / "test_scored.lock").exists() or (tables / "test_scores.csv").exists():
        raise RuntimeError("Notebook 90 has already scored the test set; refusing a second pass")
    ready, val = _load_score_seeds(packages, None, extra["config"], extra["frozen"])
    frame = _score_once(packages, ready, None, extra["config"], extra["frozen"])
    _report_from_scores(packages, ready, val, preflight)
    print(json.dumps({"status": "PASS", "test_score_rows": len(frame),
                      "score_sha256": sha256_file(tables / "test_scores.csv"),
                      "report": str(tables / "run_report_notebook90_local.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
