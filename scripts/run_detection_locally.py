"""Run the detection model ladder on the local Windows CPU worktree.

Notebook files are intentionally left unchanged. Model selection uses train and
validation only; test access is isolated to the final comparison phase.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dronedet.bundle import verify_bundle  # noqa: E402
from dronedet.config import cache_content_hash, load_config, load_project_config, params_hash  # noqa: E402
from dronedet.data_preparation import locate_dataset_root  # noqa: E402
from dronedet.features import load_cache_index, load_power_cache, load_tabular  # noqa: E402
from dronedet.ledger import (package_versions, save_freeze_package, sha256_file,
                             write_run_report, write_verdict_card)  # noqa: E402
from dronedet.metrics import (choose_configuration, choose_f1_threshold,
                              group_bootstrap, metrics_at_threshold)  # noqa: E402
from dronedet.models.classical import build_model, classical_grids, score_kind  # noqa: E402
from dronedet.models.mlp import build_mlp, mlp_parameter_count, assert_default_parameter_count  # noqa: E402
from dronedet.experiments import fit_classical  # noqa: E402
from dronedet.splits import load_split  # noqa: E402


HYPOTHESES = {
    10: "Decent ranking (ROC-AUC) because many of the 84 features separate the classes linearly, but outputs outside [0, 1] and a poor probability reading; the F1-optimal threshold will not be 0.5.",
}


def local_runtime() -> dict:
    threads = os.cpu_count() or 1
    if torch.cuda.is_available():
        raise RuntimeError("STOP: this project run is configured for CPU-only execution")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(max(1, min(threads, 4)))
    threadpool_limits(limits=threads)
    if torch.get_num_threads() != threads:
        raise RuntimeError(f"CPU thread count changed during setup: {torch.get_num_threads()} != {threads}")
    return {"device": "cpu", "cpu_count": threads, "torch_threads": torch.get_num_threads(),
            "python": sys.version, "platform": platform.platform(),
            "libraries": package_versions(), "code_commit": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()}


def _frozen_hashes(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            result[key.strip()] = value.strip()
    return result


def _manifest_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase0_local(runtime: dict) -> dict:
    """Local equivalent of C0.5–C0.8; test is hashed/path-checked only."""
    project = load_project_config(ROOT / "config.yaml")
    paths, config = project["P"], project["fe"]
    hashes = {
        "manifest_full": _manifest_sha(paths["interim"] / "manifest_full.csv"),
        "train": _manifest_sha(paths["splits"] / "train.csv"),
        "val": _manifest_sha(paths["splits"] / "val.csv"),
        "test": _manifest_sha(paths["splits"] / "test.csv"),
    }
    frozen = _frozen_hashes(paths["splits"] / "FROZEN.txt")
    recorded = json.loads((paths["tables"] / "manifest_hashes.json").read_text(encoding="utf-8"))
    yaml_hashes = project["split"]["manifest_hashes"]
    for name, actual in hashes.items():
        if not (actual == frozen.get(name) == recorded.get(name) == yaml_hashes.get(name)):
            raise RuntimeError(f"STOP local manifest integrity {name}: actual={actual}")
    current_params_hash = params_hash(config)
    current_cache_hash = cache_content_hash(config)
    front = json.loads((paths["spec"] / "frontend_hash.json").read_text(encoding="utf-8"))
    if current_params_hash != front.get("params_hash"):
        raise RuntimeError("STOP local front-end params hash mismatch")
    content_record = json.loads((paths["spec"] / "cache_content_hash.json").read_text(encoding="utf-8"))
    if current_cache_hash != content_record.get("cache_content_hash"):
        raise RuntimeError("STOP local cache content hash mismatch")
    stats_path = paths["spec"] / "norm_stats_global.npz"
    stats_sha = sha256_file(stats_path)
    stats_expected = front.get("artifacts", {}).get(
        "data/processed/spectrograms/norm_stats_global.npz", {}).get("sha256")
    if stats_sha != stats_expected:
        raise RuntimeError("STOP local normalization-statistics hash mismatch")
    bundle = verify_bundle(paths["golden"] / "frontend_bundle", config)

    raw_root = locate_dataset_root(paths["raw_root"])
    binary_root = raw_root / "Binary_Drone_Audio"
    path_counts = {}
    for split in ("train", "val", "test"):
        manifest = pd.read_csv(paths["splits"] / f"{split}.csv", usecols=["clip_id"], keep_default_na=False)
        missing = []
        for clip_id in manifest.clip_id:
            candidate = (binary_root / str(clip_id).replace("\\", "/")).resolve()
            if binary_root.resolve() not in candidate.parents or not candidate.is_file():
                missing.append(str(clip_id))
        path_counts[split] = {"manifest_rows": len(manifest), "missing": len(missing)}
        if missing:
            raise RuntimeError(f"STOP local raw paths {split}: {missing[:5]}")
    # No test labels, audio, features, flags or scores are loaded here.
    smoke = {}
    for split, expected_rows in (("train", 7607), ("val", 1510)):
        manifest = load_split(paths, split)
        data = load_tabular(paths, split)
        index = load_cache_index(paths, split)
        power = load_power_cache(paths, split)
        if data["X"].shape != (expected_rows, 84) or not np.isfinite(data["X"]).all():
            raise RuntimeError(f"STOP local loader smoke {split}: shape or finiteness mismatch")
        if not np.array_equal(data["clip_id"], manifest.clip_id.to_numpy()) or not np.array_equal(
                data["clip_id"], index.clip_id.to_numpy()):
            raise RuntimeError(f"STOP local loader smoke {split}: row-order mismatch")
        if any(not np.array_equal(data[key].astype(bool), manifest[key].astype(bool).to_numpy())
               for key in ("near_silent", "partial_zero", "clipped_flag", "dc_flag")):
            raise RuntimeError(f"STOP local loader smoke {split}: flag mismatch")
        smoke[split] = {"shape": list(data["X"].shape), "finite": True,
                        "groups": int(manifest.group.nunique()), "index_rows": len(index),
                        "power_cache_shape": list(power.shape)}
    if int(load_tabular(paths, "train")["stats_exclude"].sum()) != 88:
        raise RuntimeError("STOP local train stats exclusion mismatch")
    state = {"status": "PASS", "execution": "local Windows CPU",
             "runtime": runtime, "manifest_hashes": hashes,
             "params_hash": current_params_hash, "cache_content_hash": current_cache_hash,
             "stats_sha256": stats_sha, "bundle_verification": bundle,
             "raw_path_counts": path_counts, "loader_smoke": smoke,
             "train_stats_eligible": 7519, "train_stats_excluded": 88,
             "test_reads": 0, "frontend_core_frozen_tag_present": False,
             "cnn_gate": "still requires verified front-end G1-G14 and tag"}
    interim_path = paths["interim"] / "environment_local.json"
    interim_path.write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    (paths["tables"] / "run_report_phase0_local.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return state


def prepare_tabular(paths: dict):
    train = load_tabular(paths, "train")
    val = load_tabular(paths, "val")
    scaler = StandardScaler().fit(train["X"][train["stats_eligible"]])
    X_train = scaler.transform(train["X"]).astype(np.float32)
    X_val = scaler.transform(val["X"]).astype(np.float32)
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("scaled tabular features contain non-finite values")
    return train, val, scaler, X_train, X_val


def _save_figure(fig, path: Path) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite figure: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return str(path.relative_to(ROOT))


def run_nb10(paths: dict, runtime: dict) -> dict:
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.utils.class_weight import compute_sample_weight

    train, val, scaler, X_train, X_val = prepare_tabular(paths)
    y_train, y_val = train["y"], val["y"]
    fit = fit_classical(ROOT, 10, {}, 42, X_train, y_train, X_val, y_val,
                        score_kind="regression", sample_weight=True,
                        lineage_hash=params_hash(load_config(ROOT / "config.yaml")),
                        device="cpu", thread_count=runtime["torch_threads"])
    scores = fit["scores"]
    selected = choose_f1_threshold(y_val, scores)
    at_half = metrics_at_threshold(y_val, scores, 0.5)
    cond = float(np.linalg.cond(X_train.astype(np.float64)))
    corr = np.asarray([np.corrcoef(X_train[:, j], y_train)[0, 1] for j in range(X_train.shape[1])])
    corr = np.nan_to_num(corr, nan=0.0)
    best_feature = int(np.argmax(np.abs(corr)))
    simple = LinearRegression().fit(X_train[:, [best_feature]], y_train,
                                    sample_weight=compute_sample_weight("balanced", y_train))
    simple_scores = simple.predict(X_val[:, [best_feature]])
    simple_threshold = choose_f1_threshold(y_val, simple_scores)
    pred = scores >= selected["threshold"]
    wrong = np.flatnonzero(pred != y_val)
    wrong = sorted(wrong, key=lambda i: abs(scores[i] - y_val[i]), reverse=True)[:10]
    flags = ["near_silent", "partial_zero", "clipped_flag", "dc_flag", "crop_all_zero"]
    wrong_rows = [{"clip_id": str(val["clip_id"][i]), "label": int(y_val[i]),
                   "score": float(scores[i]), "flags": {f: bool(val.get(f, np.zeros(len(y_val), bool))[i]) for f in flags}}
                  for i in wrong]
    figures = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, name in ((0, "negative"), (1, "drone")):
        ax.hist(scores[y_val == label], bins=40, alpha=0.55, density=True, label=name)
    ax.axvline(0.5, color="black", linestyle="--", label="0.5")
    ax.axvline(selected["threshold"], color="red", linestyle=":", label="validation F1 threshold")
    ax.set(title="Notebook 10: regression score distributions", xlabel="raw regression score", ylabel="density")
    ax.legend(); figures.append(_save_figure(fig, ROOT / "results" / "figures" / "10_score_histogram.png")); plt.close(fig)
    estimator = fit["estimator"]
    order = np.argsort(np.abs(estimator.coef_))[::-1]
    names = np.asarray(train["feature_names"], dtype=str)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(names[order[:20]][::-1], estimator.coef_[order[:20]][::-1])
    ax.set(title="Notebook 10: 20 largest standardized coefficients", xlabel="coefficient")
    figures.append(_save_figure(fig, ROOT / "results" / "figures" / "10_coefficients.png")); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(scores, y_val - scores, s=12, alpha=0.45)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(title="Notebook 10: validation residuals", xlabel="fitted score", ylabel="label - fitted")
    figures.append(_save_figure(fig, ROOT / "results" / "figures" / "10_residuals.png")); plt.close(fig)
    card = {"model": "10_linear_regression", "representation": "84 scaled tabular features",
            "best_hyperparameters": {"fit": "weighted ordinary least squares"},
            "val_metrics": {"at_optimal_threshold": selected, "at_0.5": at_half,
                            "one_feature": simple_threshold},
            "seeds": [42], "train_time_s": fit["seconds"],
            "strengths": [f"validation ROC-AUC={selected['roc_auc']:.6f}"],
            "weaknesses": [f"0.5-threshold F1={at_half['f1']:.6f}; optimal-threshold F1={selected['f1']:.6f}"],
            "failure_cases_noted": wrong_rows, "decision_log_ids": ["D044", "D045", "D047"],
            "threshold": selected["threshold"], "threshold_rule": "validation F1; ties to larger threshold",
            "selection_metric": "validation F1", "val_auc": selected["roc_auc"],
            "n_parameters": int(len(estimator.coef_) + 1), "grid_searched": False,
            "tie_with_default": True, "front_end_hash": params_hash(load_config(ROOT / "config.yaml"))}
    package = save_freeze_package(ROOT, "10", estimators={42: estimator}, scaler=scaler,
        thresholds={42: selected["threshold"]}, validation_f1={42: selected["f1"]},
        lineage_hash=params_hash(load_config(ROOT / "config.yaml")),
        tabular_hashes={"train": str(train["tabular_sha256"]), "val": str(val["tabular_sha256"])},
        metadata={"feature_names": names.tolist(), "one_feature_index": best_feature,
                  "condition_number_scaled_train": cond})
    card_path = write_verdict_card(ROOT, "10", card)
    run_report = {"notebook": 10, "hypothesis": HYPOTHESES[10], "outcome": selected,
                  "threshold_0_5": at_half, "condition_number": cond,
                  "one_feature": {"index": best_feature, "name": str(names[best_feature]),
                                  "correlation": float(corr[best_feature]), "metrics": simple_threshold},
                  "wrongly_scored_validation_clips": wrong_rows, "run_id": fit["run_id"],
                  "artifact": package["path"], "verdict_card": str(card_path.relative_to(ROOT)),
                  "scaler_fit_rows": int(train["stats_eligible"].sum()), "fit_seconds": fit["seconds"],
                  "warnings": fit["warnings"], "figures": figures, "test_reads": 0}
    write_run_report(ROOT, "10", run_report)
    return run_report


def _mean_candidate(params: dict, runs: list[dict], order: int) -> dict:
    f1s = np.asarray([run["f1"] for run in runs], dtype=float)
    return {"params": params, "runs": runs, "mean_f1": float(f1s.mean()),
            "sd_f1": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
            "mean_auc": float(np.mean([run["roc_auc"] for run in runs])),
            "complexity": float(np.mean([run["complexity"] for run in runs])),
            "grid_order": order}


def run_classical_grid(notebook: int, paths: dict, runtime: dict) -> dict:
    """Run one complete, resumable candidate grid using only train/validation."""
    train, val, scaler, X_train_scaled, X_val_scaled = prepare_tabular(paths)
    scaled_models = notebook in (11, 12, 13, 18)
    X_train = X_train_scaled if scaled_models else train["X"]
    X_val = X_val_scaled if scaled_models else val["X"]
    y_train, y_val = train["y"], val["y"]
    spec = classical_grids()[notebook]
    if notebook == 16:
        meta_path = ROOT / "results" / "models" / "14" / "metadata.json"
        if not meta_path.is_file():
            raise RuntimeError("NB16 requires the completed NB14 tree freeze")
        tree_params = json.loads(meta_path.read_text(encoding="utf-8"))["selected_params"]
        spec = {"default": [{"max_samples": 1.0, "tree_params": {}}],
                "grid": ([{"max_samples": v, "tree_params": {}} for v in (0.5, 0.8, 1.0)] +
                         [{"max_samples": v, "tree_params": tree_params} for v in (0.5, 0.8, 1.0)])}
    if notebook == 18:
        gamma_scale = 1.0 / (X_train.shape[1] * float(np.var(X_train)))
        rbf = [{**p, "gamma": float(p["gamma"]) * gamma_scale,
                "gamma_multiplier": float(p["gamma"]), "gamma_scale": gamma_scale}
               for p in spec["rbf"]]
        rbf_default = [{**p, "gamma": float(p["gamma"]) * gamma_scale,
                        "gamma_multiplier": float(p["gamma"]), "gamma_scale": gamma_scale}
                       for p in spec["rbf_default"]]
        sections = (("linear", spec["linear"], spec["linear_default"]),
                    ("rbf", rbf, rbf_default))
    else:
        sections = ((str(notebook), spec["grid"], spec["default"]),)
    all_sections = {}
    for section, grid, defaults in sections:
        seeds = (42, 43, 44) if notebook in (11, 15, 16, 20) else (42,)
        candidates = []
        for order, params in enumerate(grid):
            runs = [fit_classical(ROOT, notebook, params, seed, X_train, y_train,
                X_val, y_val, score_kind=score_kind(notebook, kernel=params.get("kernel")),
                sample_weight=notebook in (10, 11, 17),
                lineage_hash=params_hash(load_config(ROOT / "config.yaml")),
                device="cpu", thread_count=runtime["torch_threads"]) for seed in seeds]
            candidates.append(_mean_candidate(params, runs, order))
        default_params = defaults[0]
        default = next((c for c in candidates if c["params"] == default_params), None)
        if default is None:
            raise RuntimeError(f"notebook {notebook} grid omitted its default {default_params}")
        winner = choose_configuration({**default, "params": default_params},
            [{**c, "params": c["params"]} for c in candidates if c is not default],
            deterministic=not (len(seeds) > 1))
        winning_candidate = next(c for c in candidates if c["params"] == winner["params"])
        estimators = {int(run["run_id"].rsplit("seed", 1)[1]): run["estimator"]
                      for run in winning_candidate["runs"]}
        thresholds = {int(run["run_id"].rsplit("seed", 1)[1]): run["threshold"]
                      for run in winning_candidate["runs"]}
        f1s = {int(run["run_id"].rsplit("seed", 1)[1]): run["f1"]
               for run in winning_candidate["runs"]}
        section_name = (f"{notebook}_{section}" if notebook == 18 else
                        "20_perceptron" if notebook == 20 else str(notebook))
        package = save_freeze_package(ROOT, section_name, estimators=estimators,
            scaler=scaler if scaled_models else None, thresholds=thresholds,
            validation_f1=f1s, lineage_hash=params_hash(load_config(ROOT / "config.yaml")),
            tabular_hashes={"train": str(train["tabular_sha256"]),
                            "val": str(val["tabular_sha256"])},
            metadata={"selected_params": winning_candidate["params"],
                      "all_candidates": [{k: v for k, v in c.items() if k != "runs"}
                                         for c in candidates],
                      "tie_with_default": bool(winner.get("tie_with_default")),
                      "test_reads": 0})
        best_run = max(winning_candidate["runs"], key=lambda run: (run["f1"], -int(run["run_id"].rsplit("seed",1)[1])))
        selected_metrics = metrics_at_threshold(y_val, best_run["scores"], best_run["threshold"])
        card = {"model": f"{notebook}_{section}", "representation": "84 tabular features",
            "best_hyperparameters": winning_candidate["params"], "val_metrics": selected_metrics,
            "seeds": list(estimators), "train_time_s": sum(r["seconds"] for r in winning_candidate["runs"]),
            "strengths": [f"validation ROC-AUC={selected_metrics['roc_auc']:.6f}"],
            "weaknesses": [], "failure_cases_noted": [], "decision_log_ids": ["D057"],
            "threshold": best_run["threshold"], "threshold_rule": "validation F1; ties to larger threshold",
            "selection_metric": "validation F1", "val_auc": selected_metrics["roc_auc"],
            "n_parameters": int(winning_candidate["complexity"]), "grid_searched": True,
            "tie_with_default": bool(winner.get("tie_with_default")),
            "front_end_hash": params_hash(load_config(ROOT / "config.yaml"))}
        card_path = write_verdict_card(ROOT, section_name, card)
        bootstrap = group_bootstrap(y_val, best_run["scores"], val["group"],
                                    threshold=best_run["threshold"], draws=1000, seed=42)
        all_sections[section] = {"default": default["params"], "winner": winning_candidate["params"],
            "tie_with_default": bool(winner.get("tie_with_default")),
            "candidate_summary": [{k: v for k, v in c.items() if k != "runs"} for c in candidates],
            "selected_metrics": selected_metrics, "group_bootstrap": bootstrap,
            "artifact": package["path"], "verdict_card": str(card_path.relative_to(ROOT)),
            "fit_seconds": sum(r["seconds"] for r in winning_candidate["runs"]),
            "warnings": [w for run in winning_candidate["runs"] for w in run["warnings"]],
            "test_reads": 0}
    result = {"notebook": notebook, "sections": all_sections, "test_reads": 0}
    write_run_report(ROOT, "20_perceptron" if notebook == 20 else str(notebook), result)
    return result


def run_nb14(paths: dict, runtime: dict) -> dict:
    from sklearn.tree import DecisionTreeClassifier
    train, val, _, _, _ = prepare_tabular(paths)
    X, y, Xv, yv = train["X"], train["y"], val["X"], val["y"]
    lineage = params_hash(load_config(ROOT / "config.yaml"))
    route_a_params = [{"max_depth": d, "min_samples_leaf": leaf, "criterion": "gini"}
                      for d in (2, 3, 4, 6, 8, None) for leaf in (1, 5, 20)]
    route_a = [fit_classical(ROOT, 14, p, 42, X, y, Xv, yv,
        score_kind="probability", lineage_hash=lineage, thread_count=runtime["torch_threads"])
        for p in route_a_params]
    def best_tree(params, runs):
        return max(zip(params, runs), key=lambda x: (x[1]["f1"], -x[1]["complexity"]))
    a_params, a_run = best_tree(route_a_params, route_a)
    entropy_params = {**a_params, "criterion": "entropy"}
    entropy_run = fit_classical(ROOT, 14, entropy_params, 42, X, y, Xv, yv,
        score_kind="probability", lineage_hash=lineage, thread_count=runtime["torch_threads"])
    default_params = {"max_depth": None, "min_samples_leaf": 1, "criterion": "gini"}
    default = next((r for p, r in zip(route_a_params, route_a) if p == default_params), None)
    if default is None: raise RuntimeError("NB14 default missing from Route A")
    path_model = DecisionTreeClassifier(class_weight="balanced", random_state=42)
    path_model.fit(X, y)
    alphas = np.asarray(path_model.cost_complexity_pruning_path(X, y).ccp_alphas)
    alphas = np.unique(alphas[alphas > 0])
    if len(alphas) > 1: alphas = alphas[:-1]
    if len(alphas) > 20: alphas = np.geomspace(alphas.min(), alphas.max(), 20)
    route_b_params = [{"max_depth": None, "min_samples_leaf": 1, "criterion": "gini", "ccp_alpha": float(a)} for a in alphas]
    route_b = [fit_classical(ROOT, 14, p, 42, X, y, Xv, yv,
        score_kind="probability", lineage_hash=lineage, thread_count=runtime["torch_threads"])
        for p in route_b_params]
    b_params, b_run = best_tree(route_b_params, route_b) if route_b else (None, None)
    deterministic = True
    default_info = {"params": default_params, "mean_f1": default["f1"], "sd_f1": 0.0,
                    "complexity": default["complexity"]}
    a_info = {"params": entropy_params, "mean_f1": entropy_run["f1"], "complexity": entropy_run["complexity"]}
    b_info = ({"params": b_params, "mean_f1": b_run["f1"], "complexity": b_run["complexity"]}
              if b_run else None)
    a_pass = a_info["mean_f1"] - default_info["mean_f1"] >= 0.01
    b_pass = b_info is not None and b_info["mean_f1"] - default_info["mean_f1"] >= 0.01
    chosen = default_info
    if a_pass: chosen = a_info
    if b_pass and (not a_pass or b_info["mean_f1"] >= a_info["mean_f1"] + 0.01
                   or (abs(b_info["mean_f1"] - a_info["mean_f1"]) < 0.01
                       and b_info["complexity"] < a_info["complexity"])):
        chosen = b_info
    if chosen["params"] == default_params: chosen_run = default
    elif chosen["params"] == entropy_params: chosen_run = entropy_run
    else: chosen_run = next(r for p, r in zip(route_b_params, route_b) if p == chosen["params"])
    model = chosen_run["estimator"]
    threshold, f1 = chosen_run["threshold"], chosen_run["f1"]
    pkg = save_freeze_package(ROOT, "14", estimators={42: model}, scaler=None,
        thresholds={42: threshold}, validation_f1={42: f1}, lineage_hash=lineage,
        tabular_hashes={"train": str(train["tabular_sha256"]), "val": str(val["tabular_sha256"])},
        metadata={"selected_params": chosen["params"], "route_a_best": a_info,
                  "route_b_best": b_info, "test_reads": 0})
    met = metrics_at_threshold(yv, chosen_run["scores"], threshold)
    card = {"model": "14_decision_tree", "representation": "84 raw tabular features",
        "best_hyperparameters": chosen["params"], "val_metrics": met, "seeds": [42],
        "train_time_s": chosen_run["seconds"], "strengths": [], "weaknesses": [],
        "failure_cases_noted": [], "decision_log_ids": ["D057"], "threshold": threshold,
        "threshold_rule": "validation F1; ties to larger threshold", "selection_metric": "validation F1",
        "val_auc": met["roc_auc"], "n_parameters": int(model.tree_.node_count), "grid_searched": True,
        "tie_with_default": chosen["params"] == default_params,
        "front_end_hash": lineage}
    card_path = write_verdict_card(ROOT, "14", card)
    result = {"notebook": 14, "default_f1": default["f1"], "route_a_best": a_info,
        "route_b_best": b_info, "selected": chosen, "selected_metrics": met,
        "artifact": pkg["path"], "verdict_card": str(card_path.relative_to(ROOT)), "test_reads": 0}
    write_run_report(ROOT, "14", result)
    return result


def run_nb17(paths: dict, runtime: dict) -> dict:
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    train, val, _, _, _ = prepare_tabular(paths)
    X, y, Xv, yv = train["X"], train["y"], val["X"], val["y"]
    lineage = params_hash(load_config(ROOT / "config.yaml"))
    summary, best = [], None
    staged_default_50 = None
    sample_weight = compute_sample_weight("balanced", y)
    for learning_rate in (0.03, 0.1, 0.3, 1.0):
        model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=800, learning_rate=learning_rate, random_state=42)
        start = time.perf_counter()
        model.fit(X, y, sample_weight=sample_weight)
        elapsed = time.perf_counter() - start
        staged = []
        for round_no, scores in enumerate(model.staged_decision_function(Xv), start=1):
            if learning_rate == 1.0 and round_no == 50:
                staged_default_50 = np.asarray(scores, dtype=np.float64).copy()
            choice = choose_f1_threshold(yv, scores)
            staged.append((choice["f1"], round_no, float(choice["threshold"]),
                           float(choice["roc_auc"]), np.asarray(scores, dtype=np.float64)))
        local = max(staged, key=lambda x: (x[0], -x[1]))
        record = {"learning_rate": learning_rate, "rounds": 800,
                  "best_f1": local[0], "best_round": local[1], "seconds": elapsed}
        summary.append(record)
        if best is None or (local[0], -local[1]) > (best["candidate"][0], -best["candidate"][1]):
            best = {"params": {"learning_rate": learning_rate, "n_estimators": local[1]},
                    "candidate": local, "model_800": model, "elapsed": elapsed}
    default50_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50, learning_rate=1.0, random_state=42).fit(X, y, sample_weight=sample_weight)
    direct50 = default50_model.decision_function(Xv)
    if staged_default_50 is None: raise RuntimeError("NB17 staged default round 50 was not recorded")
    staged50 = staged_default_50
    parity_50 = float(np.max(np.abs(direct50 - staged50)))
    if parity_50 > 1e-9: raise RuntimeError(f"NB17 direct/staged round-50 parity failed: {parity_50}")
    default_f1_50 = choose_f1_threshold(yv, staged_default_50)["f1"]
    if best["candidate"][0] - default_f1_50 < 0.01:
        default_curve = next(s for lr, s in ((1.0, staged_default_50),))
        best = {"params": {"learning_rate": 1.0, "n_estimators": 50},
                "candidate": (default_f1_50, 50,
                    choose_f1_threshold(yv, default_curve)["threshold"],
                    choose_f1_threshold(yv, default_curve)["roc_auc"], default_curve),
                "elapsed": next(r["seconds"] for r in summary if r["learning_rate"] == 1.0)}
    best_round = best["candidate"][1]
    frozen = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=best_round, learning_rate=best["params"]["learning_rate"], random_state=42)
    start = time.perf_counter(); frozen.fit(X, y, sample_weight=sample_weight)
    scores = frozen.decision_function(Xv)
    parity_chosen = float(np.max(np.abs(scores - best["candidate"][4])))
    if parity_chosen > 1e-9: raise RuntimeError(f"NB17 selected staged/refit parity failed: {parity_chosen}")
    selected = choose_f1_threshold(yv, scores)
    pkg = save_freeze_package(ROOT, "17", estimators={42: frozen}, scaler=None,
        thresholds={42: selected["threshold"]}, validation_f1={42: selected["f1"]},
        lineage_hash=lineage, tabular_hashes={"train": str(train["tabular_sha256"]),
        "val": str(val["tabular_sha256"])}, metadata={"selected_params": best["params"],
        "staged_grid": summary, "parity_round_50_max_abs": parity_50,
        "parity_selected_max_abs": parity_chosen, "test_reads": 0})
    met = metrics_at_threshold(yv, scores, selected["threshold"])
    card = {"model": "17_adaboost_stumps", "representation": "84 raw tabular features",
        "best_hyperparameters": best["params"], "val_metrics": met, "seeds": [42],
        "train_time_s": float(time.perf_counter() - start), "strengths": [], "weaknesses": [],
        "failure_cases_noted": [], "decision_log_ids": ["D057"], "threshold": selected["threshold"],
        "threshold_rule": "validation F1; ties to larger threshold", "selection_metric": "validation F1",
        "val_auc": met["roc_auc"], "n_parameters": best_round, "grid_searched": True,
        "tie_with_default": not (selected["f1"] - default_f1_50 >= 0.01),
        "front_end_hash": lineage}
    card_path = write_verdict_card(ROOT, "17", card)
    result = {"notebook": 17, "selected": best["params"], "metrics": met,
        "staged_grid": summary, "default_round_50_f1": default_f1_50,
        "parity_round_50_max_abs": parity_50,
        "parity_selected_max_abs": parity_chosen, "artifact": pkg["path"],
        "verdict_card": str(card_path.relative_to(ROOT)), "test_reads": 0}
    write_run_report(ROOT, "17", result)
    return result


def run_nb20_mlp(paths: dict, runtime: dict) -> dict:
    import csv
    import joblib
    from dronedet.ledger import append_run
    from dronedet.metrics import choose_configuration, group_bootstrap
    from dronedet.training import train_torch_model
    assert_default_parameter_count()
    train, val, scaler, Xt, Xv = prepare_tabular(paths)
    y, yv = train["y"], val["y"]
    lineage = params_hash(load_config(ROOT / "config.yaml"))
    specs = []
    defaults = {"hidden_layers": 2, "width": 64, "dropout": 0.2,
                "learning_rate": 1e-3, "weight_decay": 1e-2}
    for key, values in (("hidden_layers", (1, 2)), ("width", (32, 64, 128)),
                        ("dropout", (0.0, 0.2, 0.5)),
                        ("learning_rate", (1e-3, 3e-3)),
                        ("weight_decay", (1e-3, 1e-2, 1e-1))):
        for value in values:
            p = dict(defaults); p[key] = value
            if p not in specs: specs.append(p)
    if len(specs) != 9: raise AssertionError(f"expected 9 MLP configurations, got {len(specs)}")
    candidates = []
    for order, params in enumerate(specs):
        runs = []
        for seed in (42, 43, 44):
            torch.manual_seed(seed)
            model = build_mlp(hidden_layers=params["hidden_layers"], width=params["width"], dropout=params["dropout"])
            count = mlp_parameter_count(hidden_layers=params["hidden_layers"], width=params["width"])
            batch_fn = lambda idx, rng: Xt[idx]
            fit = train_torch_model(model, batch_fn, Xv, y, yv, seed=seed, batch_size=128,
                learning_rate=params["learning_rate"], weight_decay=params["weight_decay"],
                expected_threads=runtime["torch_threads"], lineage_hash=lineage)
            scores = np.asarray(fit["validation_logits"], dtype=np.float64)
            selected = choose_f1_threshold(yv, scores)
            config_id = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
            run_id = f"{config_id}_seed{seed}"
            model_dir = ROOT / "results" / "models" / "20_mlp"; model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"fit_{run_id}.pt"
            if model_path.exists(): raise FileExistsError(model_path)
            torch.save({"state_dict": fit["state_dict"], "params": params, "seed": seed,
                        "lineage_hash": lineage, "best_epoch": fit["best_epoch"]}, model_path)
            artifact_hash = sha256_file(model_path)
            record = {"notebook": 20, "stage": "mlp", "configuration_id": config_id,
                "seed": seed, "validation_f1": selected["f1"], "validation_roc_auc": selected["roc_auc"],
                "threshold": selected["threshold"], "seconds": fit["training_seconds"],
                "parameter_count": count, "lineage_hash": lineage, "device": "cpu",
                "thread_count": runtime["torch_threads"], "bit_reproducible": True,
                "warnings": json.dumps(fit["warnings"]), "code_commit": runtime["code_commit"],
                "artefact_path": str(model_path.relative_to(ROOT)), "artefact_sha256": artifact_hash,
                "run_id": run_id}
            append_run(ROOT, "20_mlp", record, scores)
            runs.append({**selected, "scores": scores, "seconds": fit["training_seconds"],
                         "complexity": count, "seed": seed, "model": fit["model"],
                         "best_epoch": fit["best_epoch"], "history": fit["history"]})
        vals = [r["f1"] for r in runs]
        candidates.append({"params": params, "runs": runs, "mean_f1": float(np.mean(vals)),
            "sd_f1": float(np.std(vals, ddof=1)), "complexity": count, "grid_order": order})
    default = next(c for c in candidates if c["params"] == defaults)
    winner = choose_configuration(default, [c for c in candidates if c is not default])
    selected = next(c for c in candidates if c["params"] == winner["params"])
    est = {r["seed"]: r["model"] for r in selected["runs"]}
    thresholds = {r["seed"]: r["threshold"] for r in selected["runs"]}
    f1s = {r["seed"]: r["f1"] for r in selected["runs"]}
    pkg = save_freeze_package(ROOT, "20_mlp", estimators=est, scaler=scaler,
        thresholds=thresholds, validation_f1=f1s, lineage_hash=lineage,
        tabular_hashes={"train": str(train["tabular_sha256"]), "val": str(val["tabular_sha256"])},
        metadata={"selected_params": selected["params"], "candidate_means": [
            {k:v for k,v in c.items() if k != "runs"} for c in candidates], "test_reads": 0})
    best = max(selected["runs"], key=lambda r: (r["f1"], -r["seed"]))
    bs = group_bootstrap(yv, best["scores"], val["group"], threshold=best["threshold"], draws=1000, seed=42)
    met = metrics_at_threshold(yv, best["scores"], best["threshold"])
    card = {"model": "20_mlp", "representation": "84 standardized tabular features",
        "best_hyperparameters": selected["params"], "val_metrics": met, "seeds": [42,43,44],
        "train_time_s": sum(r["seconds"] for r in selected["runs"]), "strengths": [], "weaknesses": [],
        "failure_cases_noted": [], "decision_log_ids": ["D057"], "threshold": best["threshold"],
        "threshold_rule": "validation F1; ties to larger threshold", "selection_metric": "validation F1",
        "val_auc": met["roc_auc"], "n_parameters": int(selected["complexity"]), "grid_searched": True,
        "tie_with_default": bool(winner.get("tie_with_default")), "front_end_hash": lineage}
    card_path = write_verdict_card(ROOT, "20_mlp", card)
    result = {"notebook": "20_mlp", "winner": selected["params"], "metrics": met,
        "group_bootstrap": bs, "epochs_by_seed": {r["seed"]: r["best_epoch"] for r in selected["runs"]},
        "candidate_summary": [{k:v for k,v in c.items() if k != "runs"} for c in candidates],
        "artifact": pkg["path"], "verdict_card": str(card_path.relative_to(ROOT)), "test_reads": 0}
    write_run_report(ROOT, "20_mlp", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "10", "11", "12", "13", "14", "15", "16", "17", "18", "20p", "20m"), default="preflight")
    args = parser.parse_args()
    runtime = local_runtime()
    project = load_project_config(ROOT / "config.yaml")
    if args.phase == "preflight":
        print(json.dumps(phase0_local(runtime), indent=2))
    elif args.phase == "10":
        result = run_nb10(project["P"], runtime)
        print(json.dumps(result, indent=2))
    elif args.phase == "14":
        result = run_nb14(project["P"], runtime)
        print(json.dumps(result, indent=2))
    elif args.phase == "20p":
        result = run_classical_grid(20, project["P"], runtime)
        print(json.dumps(result, indent=2))
    elif args.phase == "17":
        result = run_nb17(project["P"], runtime)
        print(json.dumps(result, indent=2))
    elif args.phase == "20m":
        result = run_nb20_mlp(project["P"], runtime)
        print(json.dumps(result, indent=2))
    else:
        result = run_classical_grid(int(args.phase), project["P"], runtime)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
