#!/usr/bin/env python3
"""Build the offline model comparison report from frozen project results.

This reads existing CSV/JSON/NPZ artifacts only. It does not train models or
run inference. The generated HTML embeds aggregated plot data and anonymous
designated-seed score vectors; clip names and group identifiers are omitted.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


MODEL_META = {
    "10_linear_regression": ("Linear regression", "Linear baseline", 42, 1,
        "Unregularized linear regression on the 84-feature vector; no hyperparameter grid. Validation F1 threshold selected from its continuous score.", "OLS, one fitted configuration"),
    "11_sgd_regressor": ("SGD regressor", "Linear baseline", 42, 30,
        "Stochastic gradient descent regressor; 30 settings compared in the ledger. Winner retained from validation selection.", "30 settings across the logged SGD parameter grid"),
    "12_logistic_regression": ("Logistic regression", "Linear classifier", 42, 12,
        "Regularized logistic classifier; C and penalty choices compared. Winner uses C=0.01, L2.", "12 configurations"),
    "13_knn": ("k-nearest neighbours", "Instance-based", 42, 28,
        "Neighbour count, distance weighting, and Minkowski metric compared. Winner: k=9, distance weights, p=1.", "28 configurations"),
    "14_decision_tree": ("Decision tree", "Tree model", 42, 39,
        "Two search routes: bounded depth/leaf/criterion screening and cost-complexity pruning. Selected full Gini tree with ccp_alpha=0.000620737.", "39 configurations across screening and pruning routes"),
    "15_random_forest": ("Random forest", "Ensemble", 43, 8,
        "Eight configurations are recorded in the ledger, including the library default. The default setting was retained: sqrt features, balanced class weights, unrestricted depth, leaf size 1.", "8 logged configurations; default retained"),
    "16_bagging": ("Bagging", "Ensemble", 44, 6,
        "Six configurations compared across sampling and base-tree settings. The default configuration was retained.", "6 configurations; final choice tied the default"),
    "17_adaboost": ("AdaBoost", "Boosted trees", 42, 4,
        "Four learning rates were each evaluated through 800 staged rounds (3,200 learning-rate/round pairs). Best validation checkpoint: learning_rate=1.0, 758 estimators.", "4 learning rates × 800 staged rounds"),
    "18_svm_linear": ("SVM · linear", "Support vector machine", 42, 5,
        "Linear kernel, C in {0.01, 0.1, 1, 10, 100}; winner C=0.01.", "5 linear-kernel settings"),
    "18_svm_rbf": ("SVM · RBF", "Support vector machine", 42, 25,
        "RBF kernel, five C values crossed with five gamma choices (25 settings). Winner is the validation-selected RBF setting.", "25 RBF settings"),
    "20_perceptron": ("Perceptron", "Neural linear model", 43, 1,
        "Default perceptron configuration; no parameter grid. Validation selected its decision threshold.", "One default configuration; no grid"),
    "20_mlp": ("Multilayer perceptron", "Neural network", 43, 10,
        "Nine one-factor settings plus one combined setting, each trained with three seeds. Winner: 2 hidden layers × 64 units, dropout 0.5, learning rate 0.001, weight decay 0.01.", "9 one-factor settings + 1 combination; 3 seeds each"),
    "30_cnn": ("CNN · tuned", "Convolutional network", 44, 7,
        "Frozen Stage C depth_4 winner: 16–32–64–64 channels, 60,641 parameters; test-scored seed 44 and validation-frozen threshold shown. Stage D validation retrains are separate.", "Stage C: 7 run, 6 skipped; Stage D: 3 seeds"),
    "30_cnn_control": ("CNN · clean control", "Convolutional network", 43, 1,
        "Clean Stage D control using the frozen frontend and same 16–32–64–64 architecture, with the control recipe recorded in Stage D.", "One designated control fit scored; 3-seed Stage D run"),
}

MODEL_ORDER = list(MODEL_META)
CARD_GROUPS = [
    ("Linear regression", ["10_linear_regression"]),
    ("SGD regression", ["11_sgd_regressor"]),
    ("Logistic regression", ["12_logistic_regression"]),
    ("k-nearest neighbours", ["13_knn"]),
    ("Decision tree", ["14_decision_tree"]),
    ("Random forest", ["15_random_forest"]),
    ("Bagging", ["16_bagging"]),
    ("AdaBoost", ["17_adaboost"]),
    ("Support vector machines", ["18_svm_linear", "18_svm_rbf"]),
    ("Perceptron & MLP", ["20_perceptron", "20_mlp"]),
    ("Convolutional network", ["30_cnn", "30_cnn_control"]),
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def report_candidates(report: dict) -> list[dict]:
    out: list[dict] = []
    def walk(value):
        if isinstance(value, dict):
            if isinstance(value.get("candidate_summary"), list):
                out.extend(x for x in value["candidate_summary"] if isinstance(x, dict))
            for k, v in value.items():
                if k != "candidate_summary":
                    walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
    walk(report)
    return out


def compact_candidates(items: list[dict], limit=180):
    result = []
    seen = set()
    for item in items:
        params = item.get("params", {})
        val = item.get("mean_f1", item.get("best_f1"))
        signature = json.dumps([params, val], sort_keys=True, default=str)
        if signature in seen:
            continue
        seen.add(signature)
        result.append({"params": params, "validation_f1": val,
                       "sd_f1": item.get("sd_f1"), "round": item.get("best_round"),
                       "complexity": item.get("complexity")})
        if len(result) >= limit:
            break
    return result


def variant_data(root: Path, model: str, val_lookup: pd.DataFrame) -> list[dict]:
    tables = root / "results" / "tables"
    if model.startswith("30_cnn"):
        c = read_json(tables / "run_report_30_local_stage_c.json")
        if model == "30_cnn":
            vals = [{"name": v["name"], "validation_f1": v["mean_f1"],
                     "sd_f1": v.get("sd_f1"), "accepted": bool(v.get("accepted_vs_default")),
                     "seeds": len(v.get("seed_runs", []))} for v in c["variants"]]
            vals.append({"name": "A0 default", "validation_f1": c["baseline"]["mean_f1"],
                         "sd_f1": c["baseline"]["sd_f1"], "accepted": False, "seeds": 3})
            return vals
        return [{"name": "Stage D control", "validation_f1": 0.9382022471910112,
                 "sd_f1": None, "accepted": False, "seeds": 3}]
    if model == "17_adaboost":
        r = read_json(tables / "run_report_17.json")
        return [{"name": f"lr={x['learning_rate']} · best round {x['best_round']}",
                 "validation_f1": x["best_f1"], "sd_f1": None, "round": x["best_round"], "seeds": 1}
                for x in r.get("staged_grid", [])]
    ledger_id = {"10_linear_regression": "10", "11_sgd_regressor": "11",
                 "12_logistic_regression": "12", "13_knn": "13", "14_decision_tree": "14",
                 "15_random_forest": "15", "16_bagging": "16", "18_svm_linear": "18",
                 "18_svm_rbf": "18", "20_perceptron": "20", "20_mlp": "20_mlp"}[model]
    df = pd.read_csv(tables / f"runs_{ledger_id}.csv")
    if "18_svm_" in model:
        wanted = "linear" if model.endswith("linear") else "rbf"
        rep = read_json(tables / "run_report_18.json")
        candidate_scores = {round(float(x["mean_f1"]), 6) for x in rep["sections"][wanted]["candidate_summary"]}
        # The ledger stores opaque config hashes; report candidate means identify the kernel's rows.
        df = df[df.validation_f1.map(lambda x: round(float(x), 6) in candidate_scores)]
    if model == "20_mlp":
        df = df[df.stage.isin(["mlp", "mlp_combination"])]
    grouped = df.groupby("configuration_id", sort=False).agg(
        validation_f1=("validation_f1", "mean"), sd_f1=("validation_f1", "std"), seeds=("seed", "nunique")).reset_index()
    # Use readable hyperparameter labels from the report when available; ledger IDs are still shown in tooltips.
    report_file = {"10_linear_regression":"run_report_10.json", "11_sgd_regressor":"run_report_11.json",
                   "12_logistic_regression":"run_report_12.json", "13_knn":"run_report_13.json",
                   "14_decision_tree":"run_report_14.json", "15_random_forest":"run_report_15.json",
                   "16_bagging":"run_report_16.json", "18_svm_linear":"run_report_18.json",
                   "18_svm_rbf":"run_report_18.json", "20_perceptron":"run_report_20_perceptron.json",
                   "20_mlp":"run_report_20_mlp.json"}[model]
    rep = read_json(tables / report_file)
    cans = compact_candidates(report_candidates(rep))
    params_by_score = {}
    for c in cans:
        if isinstance(c.get("validation_f1"), (float, int)):
            params_by_score.setdefault(round(float(c["validation_f1"]), 6), c.get("params", {}))
    rows = []
    for _, r in grouped.iterrows():
        score = float(r.validation_f1)
        label = json.dumps(params_by_score.get(round(score, 6), {}), ensure_ascii=False, separators=(",", ":"))
        rows.append({"name": label if label != "{}" else str(r.configuration_id)[:12],
                     "id": str(r.configuration_id), "validation_f1": score,
                     "sd_f1": None if pd.isna(r.sd_f1) else float(r.sd_f1), "seeds": int(r.seeds)})
    if model == "10_linear_regression" and not rows:
        rows = [{"name": "OLS default", "validation_f1": float(val_lookup.loc[model, "validation_f1"]), "seeds": 1}]
    return rows


def curve_points(y: np.ndarray, scores: np.ndarray):
    fpr, tpr, _ = roc_curve(y, scores)
    rec, prec, _ = precision_recall_curve(y, scores)
    def shrink(xs, ys, n=120):
        ids = np.unique(np.linspace(0, len(xs)-1, min(n, len(xs))).astype(int))
        return [[round(float(xs[i]), 6), round(float(ys[i]), 6)] for i in ids]
    return {"roc": shrink(fpr, tpr), "pr": shrink(rec[::-1], prec[::-1])}


def build_data(root: Path):
    t = root / "results" / "tables"
    comp = pd.read_csv(t / "comparison.csv").set_index("model")
    scores_all = pd.read_csv(t / "test_scores.csv")
    lock = read_json(t / "test_scored.lock")
    actual_score_hash = hashlib.sha256((t / "test_scores.csv").read_bytes()).hexdigest()
    if actual_score_hash != lock["scores_sha256"]:
        raise ValueError("test_scores.csv does not match test_scored.lock")
    if len(scores_all) != int(lock["rows"]) or scores_all.groupby(["model", "seed"]).ngroups != int(lock["fits"]):
        raise ValueError("test score row or fit counts do not match test_scored.lock")
    # Test CSV row ordering is made common internally; only anonymous vectors are embedded.
    base = scores_all[(scores_all.model == MODEL_ORDER[0]) & (scores_all.seed == MODEL_META[MODEL_ORDER[0]][2])].sort_values("clip_id")
    clip_ids = base.clip_id.tolist()
    y = base.label.to_numpy(dtype=np.int8)
    groups = base.set_index("clip_id").loc[clip_ids].group.to_numpy()
    n_groups = int(pd.Series(groups).nunique())
    model_data = []
    score_vectors = {}
    curves = {}
    run_cache = {}
    report_cache = {}
    def designated_validation(key: str, seed: int) -> float:
        if key == "30_cnn":
            report = read_json(t / "run_report_30_local_stage_c.json")
            row = next(x for x in report["winner"]["seed_runs"] if int(x["seed"]) == seed)
            return float(row["validation_f1"])
        if key == "30_cnn_control":
            report = read_json(t / "run_report_30_local_stage_d.json")
            row = next(x for x in report["seed_results"] if int(x["seed"]) == seed)
            return float(row["validation_f1"])
        ledger_name = {"10_linear_regression":"runs_10.csv", "11_sgd_regressor":"runs_11.csv",
            "12_logistic_regression":"runs_12.csv", "13_knn":"runs_13.csv", "14_decision_tree":"runs_14.csv",
            "15_random_forest":"runs_15.csv", "16_bagging":"runs_16.csv", "17_adaboost":"runs_17.csv",
            "18_svm_linear":"runs_18.csv", "18_svm_rbf":"runs_18.csv", "20_perceptron":"runs_20.csv",
            "20_mlp":"runs_20_mlp.csv"}[key]
        ledger = run_cache.setdefault(ledger_name, pd.read_csv(t / ledger_name))
        rows = ledger[ledger.seed.astype(int) == int(seed)]
        if key in ("10_linear_regression", "20_perceptron"):
            return float(rows.validation_f1.iloc[0])
        if key == "17_adaboost":
            report = read_json(t / "run_report_17.json")
            target = float(max(report["staged_grid"], key=lambda x: x["best_f1"])["best_f1"])
            return float(rows.iloc[(rows.validation_f1 - target).abs().argmin()].validation_f1)
        if key == "14_decision_tree":
            report = read_json(t / "run_report_14.json")
            target = float(report["selected"]["mean_f1"])
        elif key == "20_mlp":
            report = read_json(t / "run_report_20_mlp.json")
            target = float(next(c["mean_f1"] for c in report["candidate_summary"] if c["params"] == report["winner"]))
        else:
            filename, section_key = {"11_sgd_regressor":("run_report_11.json","11"),
                "12_logistic_regression":("run_report_12.json","12"), "13_knn":("run_report_13.json","13"),
                "15_random_forest":("run_report_15.json","15"), "16_bagging":("run_report_16.json","16"),
                "18_svm_linear":("run_report_18.json","linear"), "18_svm_rbf":("run_report_18.json","rbf")}[key]
            if filename not in report_cache:
                report_cache[filename] = read_json(t / filename)
            section = report_cache[filename]["sections"][section_key]
            winner = section["winner"]
            target = float(next(c["mean_f1"] for c in section["candidate_summary"] if c.get("params") == winner))
        group_means = ledger.groupby("configuration_id").validation_f1.mean()
        config_id = (group_means - target).abs().idxmin()
        selected = rows[rows.configuration_id == config_id]
        if selected.empty:
            raise ValueError(f"No designated-seed ledger row for {key}, seed {seed}")
        return float(selected.validation_f1.iloc[0])
    for key in MODEL_ORDER:
        m = comp.loc[key]
        name, family, seed, count, description, search = MODEL_META[key]
        frame = scores_all[(scores_all.model == key) & (scores_all.seed == int(m.designated_seed))].set_index("clip_id").loc[clip_ids]
        if set(frame.label.astype(int)) != set(y.tolist()):
            raise ValueError(f"Labels do not align for {key}")
        vector = frame.score.to_numpy(dtype=float)
        score_vectors[key] = [round(float(v), 8) for v in vector]
        model_data.append({"id": key, "name": name, "family": family, "seed": int(m.designated_seed),
            "threshold": float(m.threshold), "validation_f1": designated_validation(key, int(m.designated_seed)),
            "test_f1": float(m.f1), "f1_low": float(m.f1_ci_low), "f1_high": float(m.f1_ci_high),
            "auc": float(m.roc_auc), "precision": float(m.precision), "recall": float(m.recall),
            "far": float(m.false_alarm_rate), "tn": int(m.tn), "fp": int(m.fp),
            "fn": int(m.fn), "tp": int(m.tp), "params_count": None if pd.isna(m.complexity) else int(m.complexity),
            "complexity": str(m.complexity_measure), "train_seconds": None if pd.isna(m.train_seconds) else float(m.train_seconds),
            "variants_count": count, "search": search, "description": description,
            "variants": variant_data(root, key, comp)})
        if key in ("30_cnn", "30_cnn_control", "18_svm_rbf", "20_mlp", "15_random_forest"):
            curves[key] = curve_points(y, vector)

    # Add three pre-registered paired questions with the same 1,000 shared group draws (seed 42).
    sys.path.insert(0, str(root / "src"))
    from dronedet.metrics import paired_group_bootstrap
    questions = [("30_cnn", "18_svm_rbf"), ("30_cnn", "15_random_forest"), ("30_cnn", "30_cnn_control")]
    lookup = {x["id"]: x for x in model_data}
    raw = {k: scores_all[(scores_all.model == k) & (scores_all.seed == lookup[k]["seed"])].set_index("clip_id").loc[clip_ids].score.to_numpy(float)
           for pair in questions for k in pair}
    thresholds = {k: lookup[k]["threshold"] for pair in questions for k in pair}
    extra = paired_group_bootstrap(y, groups, raw, thresholds=thresholds, pairs=questions, draws=1000, seed=42)
    paired = pd.read_csv(t / "paired_comparisons.csv").to_dict(orient="records")
    # Keep existing pairs and append the CNN checks; avoid duplicate pair labels.
    for pair, interval in extra["intervals"].items():
        paired.append({"comparison": pair, "f1_difference_ci_low": interval[0], "f1_difference_ci_high": interval[1],
                       "draws": 1000, "groups": n_groups, "single_class_redraws": extra["single_class_redraws"]})

    c = read_json(t / "run_report_30_local_stage_c.json")
    d = read_json(t / "run_report_30_local_stage_d.json")
    stage = {
        "baseline_f1": c["baseline"]["mean_f1"], "baseline_sd": c["baseline"]["sd_f1"],
        "threshold": c["baseline"]["mean_f1"] + max(c["baseline"]["sd_f1"], .01),
        "rule": c["rule"], "variants": [{"name": x["name"], "mean": x["mean_f1"], "sd": x.get("sd_f1"),
                        "accepted": x.get("accepted_vs_default", False)} for x in c["variants"]],
        "combination": c.get("combination"), "winner": "depth_4",
        "planned": 13, "run": 7, "skipped": 6,
        "stage_d": [{"seed": int(x["seed"]), "validation_f1": float(x["validation_f1"]),
                     "best_epoch": int(x["best_epoch"]), "stopping_epoch": int(x["stopping_epoch"]),
                     "seconds": float(x["seconds"])} for x in d["seed_results"]],
        "designated_seed": int(d["designated_seed"]), "architecture": d["architecture"],
        "params": int(d["seed_results"][0]["parameter_count"]),
        "stage_d_mean": float(np.mean([x["validation_f1"] for x in d["seed_results"]])),
        "stage_d_sd": float(np.std([x["validation_f1"] for x in d["seed_results"]], ddof=1)),
    }
    a = read_json(t / "run_report_30_local_stage_a.json")
    b1 = read_json(t / "run_report_30_local_b1.json")
    b2 = read_json(t / "run_report_30_local_b2.json")
    b3 = read_json(t / "run_report_30_local_b3.json")
    stage["history"] = {
        "A": f"Frozen log-mel frontend; A0 Conv–BN–ReLU ×3, pooling between blocks, global average pooling, dropout 0.3, one logit ({a['a0_parameter_count']:,} parameters). Batch 64, Adam learning rate 0.001, weight decay 0.01, up to 100 epochs, patience 8.",
        "B1": f"PCEN change screen passed ({b1['pcen_acceptance']['max_change_in_std']:.3f} observed against {b1['pcen_acceptance']['acceptance_limit']:.1f} limit). Compared global_stats, per_window_mean and PCEN; retained global_stats under the robustness rule. Threshold fixed per seed at unfiltered 0 dB across eight gain/filter conditions.",
        "B2": f"Five frontend variants tested (40/96 mel bins; alternate FFT/hop; Slaney scale). None passed the registered robustness rule, so the A0 frontend stayed frozen. The n_mels_40 seed-43 score (0.8960) is distinct from Stage C pool_2x1's three-seed mean (0.8935).",
        "B3": f"Mel equalization was tested as a separate report-only control. Its three 0 dB validation F1 values were {', '.join(f'{z["base_metrics"]["f1"]:.3f}' for z in b3['mel_eq_runs'])}; it was not adopted into the frozen frontend.",
        "C": "Seven of thirteen planned variants ran; six were skipped before starting at the compute stop. Depth_4 won the registered one-factor acceptance rule; the accepted-change combination scored 0.9227 and was rejected against depth_4 at 0.9260.",
        "D": "Stage D retrained depth_4 with fixed centered windows, no gain augmentation, batch 64, max 100 epochs and patience 8. Three seeds used early stopping; seed 43 was designated for the clean control. This validation retrain is distinct from Stage C's frozen test-scored seed 44."
    }
    ops = pd.read_csv(t / "operating_points.csv")
    op = ops[ops.model.isin(["30_cnn", "30_cnn_control", "18_svm_rbf", "20_mlp", "15_random_forest"])].to_dict(orient="records")
    types = pd.read_csv(t / "per_type_recall.csv").to_dict(orient="records")
    families = pd.read_csv(t / "per_family_recall.csv").to_dict(orient="records")
    return {"models": model_data, "scores": score_vectors, "labels": y.tolist(), "curves": curves,
        "paired": paired, "stages": stage, "operating": op, "types": types, "families": families,
        "test": {"clips": int(len(y)), "groups": n_groups, "positive": int(y.sum()), "negative": int((1-y).sum()),
                 "fits": int(lock["fits"]), "score_rows": int(lock["rows"]), "score_hash": lock["scores_sha256"],
                 "test_manifest_hash": lock["test_manifest_sha256"], "scored_at": lock["created_utc"],
                 "score_commit": lock["project_code_commit"]},
        "cards": [{"title": title, "models": ids} for title, ids in CARD_GROUPS],
        "groups": int(n_groups)}


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark"><title>Drone audio · Model record</title>
<style>
:root{--bg:#f1f3ef;--paper:#fbfcf8;--ink:#122326;--muted:#526669;--line:#ccd6d2;--blue:#145b69;--blue2:#237c86;--mint:#b9e86e;--red:#bd553e;--amber:#d39232;--shade:0 18px 46px #172c2c12;--radius:14px;--ease:cubic-bezier(.23,1,.32,1);font-family:"Aptos", "Segoe UI Variable", sans-serif;font-synthesis:none}
[data-theme=dark]{--bg:#101a1c;--paper:#172527;--ink:#edf3ec;--muted:#afc0ba;--line:#34484a;--blue:#9adbe0;--blue2:#67b4bd;--mint:#b9e86e;--red:#f08b72;--amber:#efbc62;--shade:0 18px 46px #0005}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);line-height:1.55}button,input,select{font:inherit}button{cursor:pointer;color:inherit}a{color:inherit}.shell{max-width:1440px;margin:auto;padding:0 38px}.topbar{position:sticky;top:0;z-index:10;background:color-mix(in srgb,var(--bg) 94%,transparent);border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}.topbar .shell{height:58px;display:flex;align-items:center;justify-content:space-between;gap:20px}.brand{font-size:12px;letter-spacing:.1em;text-transform:uppercase;font-weight:750}.nav{display:flex;gap:22px;overflow:auto;white-space:nowrap;font-size:12px;color:var(--muted)}.nav a{text-decoration:none}.nav a:hover{color:var(--ink)}.theme{border:1px solid var(--line);background:var(--paper);border-radius:99px;padding:7px 12px;font-size:12px;white-space:nowrap}
main{overflow:hidden}.hero{padding:76px 0 66px;border-bottom:1px solid var(--line);position:relative}.hero:after{content:"";position:absolute;right:3%;top:44px;width:min(34vw,420px);aspect-ratio:1;border:1px solid var(--line);border-radius:50%;opacity:.5;pointer-events:none}.hero:before{content:"";position:absolute;right:9%;top:88px;width:min(23vw,290px);aspect-ratio:1;border:1px solid var(--line);border-radius:50%;opacity:.55}.hero-grid{position:relative;z-index:1;display:grid;grid-template-columns:minmax(0,1.35fr) minmax(300px,.65fr);gap:7vw;align-items:end}.kicker{font:600 11px/1.3 ui-monospace,Consolas,monospace;color:var(--blue);letter-spacing:.1em;text-transform:uppercase}.hero h1{font-size:clamp(46px,7vw,100px);letter-spacing:-.055em;line-height:.94;font-weight:680;max-width:800px;margin:20px 0 24px}.hero h1 em{color:var(--blue);font-style:normal}.hero-copy{max-width:65ch;font-size:17px;color:var(--muted)}.verdict{border-top:2px solid var(--ink);padding-top:18px;max-width:370px}.verdict strong{display:block;font-size:20px;line-height:1.25;letter-spacing:-.02em}.verdict p{margin:12px 0 0;color:var(--muted);font-size:13px}.numbers{display:flex;gap:32px;flex-wrap:wrap;margin-top:38px}.number{min-width:114px}.number b{display:block;font-size:27px;letter-spacing:-.05em}.number span{font:11px ui-monospace,Consolas,monospace;color:var(--muted)}.narrow-copy{display:none}.section{padding:68px 0;border-bottom:1px solid var(--line)}.section-head{display:flex;align-items:end;justify-content:space-between;gap:24px;margin-bottom:26px}.section h2{font-size:clamp(30px,4vw,52px);line-height:1;letter-spacing:-.045em;margin:0;max-width:18ch}.section-intro{max-width:58ch;color:var(--muted);font-size:14px;margin:0}.eyebrow{display:block;font:600 11px ui-monospace,Consolas,monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--blue);margin-bottom:12px}
.protocol{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid var(--ink);border-bottom:1px solid var(--ink)}.phase{padding:20px 22px 22px 0;position:relative}.phase+.phase{border-left:1px solid var(--line);padding-left:22px}.phase b{font:11px ui-monospace,Consolas,monospace;color:var(--blue)}.phase h3{font-size:19px;margin:9px 0 4px}.phase .count{font-size:28px;font-weight:680;letter-spacing:-.04em}.phase p{font-size:12px;color:var(--muted);margin:4px 0 0;max-width:31ch}.lockline{margin-top:18px;padding:13px 16px;background:var(--paper);border:1px solid var(--line);font:11px ui-monospace,Consolas,monospace;overflow-wrap:anywhere}
.families{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);border:1px solid var(--line)}.family{background:var(--paper);padding:24px;min-width:0}.family.full{grid-column:1/-1}.family-top{display:flex;align-items:baseline;justify-content:space-between;gap:15px;border-bottom:1px solid var(--line);padding-bottom:13px;margin-bottom:15px}.family h3{font-size:22px;letter-spacing:-.03em;margin:0}.family .count{font:11px ui-monospace,Consolas,monospace;color:var(--muted)}.model-row{display:grid;grid-template-columns:minmax(130px,.7fr) 1.1fr 1fr;gap:14px;padding:12px 0;border-bottom:1px solid var(--line);align-items:center}.model-row:last-of-type{border:0}.model-name{font-weight:700;font-size:13px}.model-name small{display:block;font-size:10px;color:var(--muted);font-weight:500}.metric-pair{display:flex;gap:15px;flex-wrap:wrap;font:11px ui-monospace,Consolas,monospace}.metric-pair b{font-size:16px;display:block;font-family:inherit;letter-spacing:-.04em}.mini-bars{display:flex;align-items:end;gap:2px;height:42px;overflow:hidden;border-bottom:1px solid var(--line);padding-top:4px}.mini-bars i{display:block;min-width:3px;flex:1;background:var(--blue2);opacity:.47;border-radius:2px 2px 0 0}.mini-bars i.best{background:var(--mint);opacity:1;outline:1px solid var(--ink)}.model-detail{font-size:12px;color:var(--muted);margin:4px 0}.detail-toggle{border:0;background:none;color:var(--blue);text-decoration:underline;text-underline-offset:3px;padding:4px 0;font-size:11px}.variant-details{display:none;margin:6px 0 10px;max-height:210px;overflow:auto;border-top:1px solid var(--line)}.variant-details.open{display:block}.variant-details table{font:10px ui-monospace,Consolas,monospace;width:100%;border-collapse:collapse}.variant-details td,.variant-details th{padding:6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}.tag{display:inline-block;background:color-mix(in srgb,var(--mint) 45%,var(--paper));border:1px solid var(--line);padding:2px 6px;border-radius:99px;font:10px ui-monospace,Consolas,monospace;color:var(--ink)}
.cnn-grid{display:grid;grid-template-columns:.75fr 1.25fr;gap:30px}.stepper{display:grid;grid-template-columns:1fr 1fr;gap:0;border-top:1px solid var(--line)}.step{padding:14px 14px 14px 0;border-bottom:1px solid var(--line);cursor:pointer}.step.active{color:var(--blue)}.step strong{display:block;font-size:15px}.step small{font:10px ui-monospace,Consolas,monospace;color:var(--muted)}.step-copy{min-height:100px;background:var(--paper);padding:17px;border:1px solid var(--line);font-size:13px;color:var(--muted)}.stage-chart{background:var(--paper);border:1px solid var(--line);padding:20px}.stage-bar{display:grid;grid-template-columns:130px 1fr 55px;gap:10px;align-items:center;margin:12px 0;font-size:11px}.track{height:14px;position:relative;background:color-mix(in srgb,var(--line) 55%,transparent)}.fill{height:100%;background:var(--blue2);transform-origin:left;animation:grow 750ms var(--ease) both}.fill.accepted{background:var(--mint)}.track .threshold{position:absolute;top:-5px;bottom:-5px;width:1px;background:var(--red);z-index:2}.architecture{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-top:26px;padding:20px;background:var(--bg);border:1px solid var(--line);overflow:auto}.layer{flex:1;text-align:center;min-width:70px}.layer i{display:block;height:var(--h);max-width:80px;margin:0 auto 8px;background:var(--blue2);opacity:var(--o);border-radius:4px}.layer strong{font:11px ui-monospace,Consolas,monospace}.layer small{display:block;font-size:10px;color:var(--muted)}.param-note{font:11px ui-monospace,Consolas,monospace;color:var(--muted);margin-top:12px}
.controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.chip,.seg button{background:var(--paper);border:1px solid var(--line);padding:7px 11px;border-radius:99px;font-size:11px}.chip.active,.seg button.active{background:var(--ink);color:var(--paper);border-color:var(--ink)}.seg{display:flex;gap:6px}.leaderboard{border-top:1px solid var(--ink)}.leader{display:grid;grid-template-columns:32px minmax(150px,1.3fr) 2fr 115px 130px;gap:15px;align-items:center;padding:12px 0;border-bottom:1px solid var(--line);font-size:12px}.rank{font:11px ui-monospace,Consolas,monospace;color:var(--muted)}.leader-label{font-weight:680}.barline{height:15px;background:color-mix(in srgb,var(--line) 55%,transparent);position:relative}.barline span{height:100%;display:block;background:var(--blue2)}.barline span.winner{background:var(--mint)}.ci-line{position:absolute;height:2px;top:6px;background:var(--red)}.leader-metric{font:11px ui-monospace,Consolas,monospace}.leader-metric b{font-size:16px}.slope-wrap{overflow:auto;background:var(--paper);padding:15px;border:1px solid var(--line)}.slope{width:100%;height:420px;min-width:700px}.legend{display:flex;gap:16px;color:var(--muted);font-size:11px;margin:8px 0}.legend i{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--blue2);margin-right:5px}.legend i.test{background:var(--red)}
.analysis-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.plot{background:var(--paper);border:1px solid var(--line);padding:18px;min-width:0}.plot h3{font-size:16px;margin:0}.plot p{font-size:11px;color:var(--muted);margin:4px 0 12px}.plot svg{width:100%;height:260px;display:block;overflow:visible}.axes{stroke:var(--line);stroke-width:1}.curve{fill:none;stroke-width:2.3}.curve.primary{stroke:var(--blue2)}.curve.control{stroke:var(--red)}.curve.rbf{stroke:var(--mint)}.curve.mlp{stroke:var(--amber)}.curve.rf{stroke:var(--ink)}.metric-control{display:grid;grid-template-columns:1fr auto;gap:9px;align-items:center;margin:12px 0}.metric-control label{font-size:12px}.metric-control input{grid-column:1/-1;width:100%;accent-color:var(--blue2)}.metric-control select{background:var(--paper);color:var(--ink);border:1px solid var(--line);padding:6px}.confusion{display:grid;grid-template-columns:1fr 1fr;gap:5px;max-width:320px;margin:14px 0}.confusion div{padding:17px;background:color-mix(in srgb,var(--blue2) 12%,var(--paper));text-align:center}.confusion div:nth-child(2),.confusion div:nth-child(3){background:color-mix(in srgb,var(--red) 14%,var(--paper))}.confusion b{font-size:24px;display:block}.confusion small{font:10px ui-monospace,Consolas,monospace;color:var(--muted)}.metric-live{display:flex;gap:18px;flex-wrap:wrap;font:11px ui-monospace,Consolas,monospace}.metric-live b{display:block;font-size:18px;color:var(--ink)}.hist{display:flex;align-items:end;height:200px;gap:1px;border-bottom:1px solid var(--line);padding:5px 0}.hist i{flex:1;min-width:1px;position:relative}.hist i.neg{background:var(--blue2);opacity:.72}.hist i.pos{background:var(--red);opacity:.8}.hist-legend{display:flex;gap:18px;font-size:11px;color:var(--muted);margin-top:8px}.hist-legend span:before{content:"";display:inline-block;width:9px;height:9px;background:var(--blue2);margin-right:6px}.hist-legend span:last-child:before{background:var(--red)}
.data-table{border-collapse:collapse;width:100%;font-size:12px}.data-table th,.data-table td{text-align:left;padding:10px 9px;border-bottom:1px solid var(--line);vertical-align:top}.data-table th{font:10px ui-monospace,Consolas,monospace;color:var(--muted);text-transform:uppercase}.data-table td.num{font:11px ui-monospace,Consolas,monospace}.paired{display:grid;gap:8px}.pair-row{display:grid;grid-template-columns:190px 1fr 130px;align-items:center;gap:12px;font-size:11px}.pair-name{overflow-wrap:anywhere}.forest{height:12px;position:relative;background:linear-gradient(90deg,transparent calc(50% - .5px),var(--ink) calc(50% - .5px),var(--ink) calc(50% + .5px),transparent calc(50% + .5px))}.forest span{position:absolute;height:2px;top:5px;background:var(--blue2)}.forest i{position:absolute;top:2px;width:8px;height:8px;border-radius:50%;background:var(--blue2);transform:translateX(-50%)}.forest .sig{background:var(--red)}.footnotes{display:grid;grid-template-columns:1fr 1fr;gap:26px}.footnotes h3{font-size:16px}.footnotes p,.footnotes li{font-size:12px;color:var(--muted)}.footnotes ul{padding-left:18px}.hash{font:10px ui-monospace,Consolas,monospace;overflow-wrap:anywhere}.footer{padding:28px 0 50px;color:var(--muted);font-size:11px;display:flex;justify-content:space-between;gap:20px}.reveal{opacity:0;transform:translateY(10px);transition:opacity 500ms var(--ease),transform 500ms var(--ease)}.reveal.visible{opacity:1;transform:none}@keyframes grow{from{transform:scaleX(.02)}to{transform:scaleX(1)}}
@media(max-width:900px){.shell{padding:0 22px}.hero-grid{grid-template-columns:1fr;gap:30px}.hero:before,.hero:after{right:-120px}.hero{padding-top:54px}.verdict{max-width:600px}.cnn-grid{grid-template-columns:1fr}.leader{grid-template-columns:25px minmax(135px,1fr) 1.5fr 95px 105px;gap:8px}.section{padding:54px 0}}
@media(max-width:640px){.shell{width:100%;padding:0 16px}.hero-grid{display:block}.hero-grid>*{min-width:0;width:100%}.hero-copy{max-width:none;width:100%}.verdict{max-width:none;width:100%}.topbar .shell{height:52px}.nav{display:none}.hero h1{font-size:clamp(39px,11vw,47px)}.hero h1 em{display:block}.narrow-copy{display:inline!important}.wide-copy{display:none!important}.hero-copy{font-size:15px!important;max-width:320px!important;width:320px!important;overflow-wrap:anywhere}.verdict{max-width:320px!important;width:320px!important;overflow-wrap:anywhere}.numbers{gap:12px}.number{min-width:0;flex:1 1 70px}.number span{display:block;max-width:72px;white-space:normal;font-size:10px}.number b{font-size:23px}.section-head{display:block}.section-intro{margin-top:14px}.protocol{grid-template-columns:1fr 1fr}.phase{padding:15px 10px 15px 0}.phase+.phase{padding-left:12px}.phase:nth-child(3){border-left:0;padding-left:0}.phase:nth-child(n+3){border-top:1px solid var(--line)}.families{grid-template-columns:1fr}.family.full{grid-column:auto}.family{padding:18px 14px}.model-row{grid-template-columns:1fr 1fr;gap:6px}.mini-bars{grid-column:1/-1}.metric-pair{justify-content:end}.stepper{grid-template-columns:1fr 1fr}.stage-bar{grid-template-columns:95px 1fr 42px;font-size:10px}.analysis-grid{grid-template-columns:1fr}.pair-row{grid-template-columns:118px 1fr 105px;gap:7px;font-size:10px}.footnotes{grid-template-columns:1fr}.leader{min-width:700px}.leaderboard{overflow:auto}.footer{display:block}.footer span{display:block;margin:7px 0}.architecture{justify-content:flex-start}.layer{min-width:58px}}
@media(hover:hover) and (pointer:fine){button,.family,.plot{transition:transform 220ms var(--ease),background-color 180ms var(--ease)}.family:hover{transform:translateY(-2px)}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}.reveal{opacity:1;transform:none;transition:opacity 180ms}.fill{animation:none;transform:none}.family:hover{transform:none}}
</style></head>
<body><script id="report-data" type="application/json">__DATA__</script>
<div class="topbar"><div class="shell"><div class="brand">Drone audio · model record</div><nav class="nav" aria-label="Report sections"><a href="#protocol">Protocol</a><a href="#models">Models</a><a href="#cnn">CNN stages</a><a href="#leaderboard">Leaderboard</a><a href="#analysis">Analysis</a><a href="#paired">Paired comparisons</a><a href="#limits">Reproducibility</a></nav><button class="theme" id="theme" aria-label="Toggle colour theme">◐ Theme</button></div></div>
<main><header class="hero"><div class="shell hero-grid"><div><div class="kicker">Final evaluation · Frozen test set</div><h1>Every model.<br><em>One locked</em> score.</h1><p class="hero-copy"><span class="wide-copy">A full record of model search, validation selection and final test performance for drone audio classification. The test set was scored once after the model choices and thresholds were frozen.</span><span class="narrow-copy">All model searches and final test results,<br>together in one offline report.<br>The test was scored once after every<br>model and threshold had been frozen.</span></p><div class="numbers" id="numbers"></div></div><aside class="verdict"><strong><span class="wide-copy">The Stage C depth_4 CNN, seed 44, leads at test F1 0.9314.</span><span class="narrow-copy">Stage C depth_4 CNN, seed 44,<br>leads at test F1 0.9314.</span></strong><p><span class="wide-copy">Stage D then retrained the selected architecture and designated seed 43 on validation; those Stage D retrain scores are shown separately and were not part of the locked test table. Paired group intervals compare the test-scored artifact with RBF SVM, random forest and the clean control.</span><span class="narrow-copy">Stage D retrained the model on validation.<br>Those runs are shown separately from<br>the locked test table. Paired intervals<br>compare the test-scored CNN with RBF<br>SVM, random forest and clean control.</span></p></aside></div></header>
<section class="section" id="protocol"><div class="shell"><div class="section-head"><div><span class="eyebrow">Evaluation contract</span><h2>Selection happened before the test.</h2></div><p class="section-intro">The validation set selected configurations and thresholds. Frozen artifacts were checked before scoring the held test data once.</p></div><div class="protocol" id="protocol-items"></div><div class="lockline" id="lockline"></div></div></section>
<section class="section" id="models"><div class="shell"><div class="section-head"><div><span class="eyebrow">Search record</span><h2>Eleven model families, fourteen final fits.</h2></div><p class="section-intro">Each strip shows the recorded validation F1 for a tried variant. Hover a bar for its score; expand a row for the logged configurations and search notes.</p></div><div class="controls" id="filters" aria-label="Filter model families"></div><div class="families" id="families"></div></div></section>
<section class="section" id="cnn"><div class="shell"><div class="section-head"><div><span class="eyebrow">Deep model record</span><h2>The CNN was chosen in stages.</h2></div><p class="section-intro">Frontend, control, augmentation and architecture decisions were separated. The held test remained untouched throughout these stages.</p></div><div class="cnn-grid"><div><div class="stepper" id="steps"></div><div class="step-copy" id="step-copy" aria-live="polite"></div></div><div><div class="stage-chart" id="stage-chart"></div><div class="architecture"><div class="layer"><i style="--h:44px;--o:.55"></i><strong>16</strong><small>channels</small></div><b>›</b><div class="layer"><i style="--h:58px;--o:.68"></i><strong>32</strong><small>channels</small></div><b>›</b><div class="layer"><i style="--h:74px;--o:.82"></i><strong>64</strong><small>channels</small></div><b>›</b><div class="layer"><i style="--h:74px;--o:1"></i><strong>64</strong><small>channels</small></div><b>›</b><div class="layer"><i style="--h:24px;--o:.4"></i><strong>1</strong><small>logit</small></div></div><div class="param-note" id="cnn-params"></div></div></div></div></section>
<section class="section" id="leaderboard"><div class="shell"><div class="section-head"><div><span class="eyebrow">Final comparison</span><h2>Test performance with group uncertainty.</h2></div><div class="seg" role="group" aria-label="Leaderboard metric"><button class="active" data-mode="test">Test F1</button><button data-mode="validation">Validation F1</button></div></div><div class="controls" id="leader-filters"></div><div class="leaderboard" id="leaderboard-rows"></div><div class="section-head" style="margin:46px 0 15px"><div><h2 style="font-size:28px">Validation to test shift</h2></div><p class="section-intro">Validation values are from each model’s designated seed. Test values use the frozen threshold.</p></div><div class="legend"><span><i></i>Designated-seed validation</span><span><i class="test"></i>Locked test</span></div><div class="slope-wrap"><svg id="slope" class="slope" role="img" aria-label="Validation and test F1 slope chart"></svg></div></div></section>
<section class="section" id="analysis"><div class="shell"><div class="section-head"><div><span class="eyebrow">Decision behaviour</span><h2>Ranking, operating points and errors.</h2></div><p class="section-intro">Curves show the five leading or reference systems. The threshold control recalculates the confusion matrix from the anonymous frozen test scores.</p></div><div class="analysis-grid"><article class="plot"><h3>ROC</h3><p>False positive rate against true positive rate</p><svg id="roc" role="img" aria-label="ROC curves"></svg></article><article class="plot"><h3>Precision–recall</h3><p>Recall against precision</p><svg id="pr" role="img" aria-label="Precision recall curves"></svg></article><article class="plot"><h3>Threshold explorer</h3><div class="metric-control"><label for="model-select">Model</label><select id="model-select"></select><span id="threshold-readout"></span><input id="threshold" type="range" min="0" max="1000" value="500" aria-label="Decision threshold"></div><div class="confusion" id="confusion"></div><div class="metric-live" id="live-metrics"></div></article><article class="plot"><h3>Score distribution</h3><p>Chosen CNN scores, separated by true class</p><div class="hist" id="hist"></div><div class="hist-legend"><span>Non-drone</span><span>Drone</span></div></article><article class="plot"><h3>False-alarm operating points</h3><p>Thresholds chosen on validation at target FAR; realized test performance</p><div id="operating"></div></article><article class="plot"><h3>Drone type and family recall</h3><p>Per-group rates; small counts can move recall sharply</p><label for="recall-select" style="font-size:11px">Model </label><select id="recall-select" class="theme"></select><div id="recall-tables"></div></article><article class="plot" style="grid-column:1/-1"><h3>F1 against ledger complexity and training time</h3><p>Log scale on time. Bubble size uses the ledger complexity field, which differs by estimator.</p><svg id="cost" role="img" aria-label="F1 against parameter count and training time"></svg></article></div></div></section>
<section class="section" id="paired"><div class="shell"><div class="section-head"><div><span class="eyebrow">Matched groups</span><h2>Paired intervals answer the closer question.</h2></div><p class="section-intro">Each interval bootstraps the same held-out groups for both models and reports ΔF1 for the left model minus the right.</p></div><div class="paired" id="paired-rows"></div><p class="section-intro" style="margin-top:18px">Intervals crossing zero do not establish a difference. They also do not prove equivalence. A comparison is stronger evidence when its paired interval excludes zero.</p></div></section>
<section class="section" id="limits"><div class="shell"><div class="section-head"><div><span class="eyebrow">Read the result in context</span><h2>Limits and reproducibility.</h2></div><p class="section-intro">This report preserves the decision trail so that strong scores are read with their limits attached.</p></div><div class="footnotes"><div><h3>Known limits</h3><ul><li>Validation was reused for model and threshold selection, so validation performance is optimistic.</li><li>The drone-positive set covers only two drone models; broad generalization is unproven.</li><li>Some negative clips may be confounded by recording artefacts.</li><li>The CNN and clean control have overlapping individual intervals. Use the paired interval above; no equivalence test was run.</li><li>Group bootstrap intervals describe variation across the 247 represented groups, not new sites or devices.</li></ul></div><div><h3>Locked artifacts</h3><p>Frozen model/feature commits: <code>7a189e5</code> (Stage C) and <code>e5c1228</code> (final scoring runner). Score lock records the manifest and score-file hashes.</p><p id="hashes" class="hash"></p><p>Source files: <code>comparison.csv</code>, <code>paired_comparisons.csv</code>, <code>operating_points.csv</code>, <code>per_type_recall.csv</code>, <code>per_family_recall.csv</code>, <code>runs_*.csv</code>, CNN Stage A–D reports, and <code>test_scores.csv</code>.</p><p>The page embeds aggregate results, curve samples and score vectors without clip names or group IDs. Regenerate with <code>python scripts/build_model_comparison_dashboard.py</code>.</p></div></div></div></section>
</main><footer class="shell footer"><span>Drone audio classification · Frozen model comparison</span><span>Generated from existing result files · no refit or inference performed</span></footer>
<script>
const D=JSON.parse(document.getElementById('report-data').textContent), M=D.models;
const $=s=>document.querySelector(s), fmt=(x,n=3)=>Number(x).toFixed(n), pct=x=>(100*x).toFixed(1)+'%', escapeHtml=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const byId=Object.fromEntries(M.map(m=>[m.id,m])); let familyFilter='all',leaderMode='test';
function makeNums(){const vals=[[14,'frozen models'],[28,'scored fits'],[D.test.clips.toLocaleString(),'test clips'],[D.test.groups,'test groups'],[fmt(byId['30_cnn'].test_f1,4),'Stage C winner · test F1']];$('#numbers').innerHTML=vals.map(v=>`<div class="number"><b data-count="${v[0]}">${v[0]}</b><span>${v[1]}</span></div>`).join('')}
function protocol(){let a=[['01','Train',7607,'training clips · fit and selection only'],['02','Validate & tune',1510,'thresholds and model choices frozen here'],['03','Freeze','hash','artifacts and manifests verified'],['04','Score test once',D.test.clips.toLocaleString(),'held-out clips · 247 groups']];$('#protocol-items').innerHTML=a.map(x=>`<div class="phase"><b>${x[0]} / ${x[1].toUpperCase()}</b><div class="count">${x[2]}</div><p>${x[3]}</p></div>`).join('');$('#lockline').textContent=`Test score SHA-256 ${D.test.score_hash} · test manifest SHA-256 ${D.test.test_manifest_hash} · ${D.test.fits} fit vectors / ${D.test.score_rows.toLocaleString()} rows`}
function makeFilters(){let filters=['all',...D.cards.map(c=>c.title)];$('#filters').innerHTML=filters.map(x=>`<button class="chip ${x==='all'?'active':''}" data-family="${escapeHtml(x)}">${escapeHtml(x==='all'?'All families':x)}</button>`).join('');$('#filters').onclick=e=>{let b=e.target.closest('button');if(!b)return;familyFilter=b.dataset.family;document.querySelectorAll('#filters button').forEach(x=>x.classList.toggle('active',x===b));renderFamilies()}}
function miniBars(m){let v=m.variants||[];if(!v.length)return '';let min=Math.min(...v.map(x=>x.validation_f1)),max=Math.max(...v.map(x=>x.validation_f1));let range=max-min||.01;return `<div class="mini-bars" role="img" aria-label="${v.length} validation variants">${v.map(x=>{let h=18+70*(x.validation_f1-min)/range;return `<i class="${x.validation_f1===max?'best':''}" style="height:${h}%" title="${escapeHtml(x.name)} · F1 ${fmt(x.validation_f1,4)}"></i>`}).join('')}</div>`}
function renderFamilies(){let out='';for(let card of D.cards){if(familyFilter!=='all'&&familyFilter!==card.title)continue;let models=card.models.map(id=>byId[id]);out+=`<article class="family ${card.title==='Convolutional network'?'full':''}"><div class="family-top"><h3>${escapeHtml(card.title)}</h3><span class="count">${models.length} final ${models.length===1?'fit':'fits'}</span></div>`;for(let m of models){let sorted=[...m.variants].sort((a,b)=>a.validation_f1-b.validation_f1),top=Math.max(...sorted.map(v=>v.validation_f1));out+=`<div class="model-row"><div class="model-name">${escapeHtml(m.name)}<small>seed ${m.seed} · threshold ${fmt(m.threshold,4)} · ${m.variants_count} tried</small></div><div class="metric-pair"><span>Validation<b>${fmt(m.validation_f1,4)}</b></span><span>Test F1<b>${fmt(m.test_f1,4)}</b></span><span>95% group CI<b>${fmt(m.f1_low,3)}–${fmt(m.f1_high,3)}</b></span><span>Precision / recall / FAR<b>${fmt(m.precision,3)} / ${fmt(m.recall,3)} / ${fmt(m.far,3)}</b></span></div><div>${miniBars(m)}</div><div class="model-detail" style="grid-column:1/-1">${escapeHtml(m.description)} <span class="tag">${escapeHtml(m.search)}</span></div><button class="detail-toggle" style="grid-column:1/-1" aria-expanded="false">Show ${m.variants.length} recorded variant scores</button><div class="variant-details" style="grid-column:1/-1"><table><thead><tr><th>Setting / ledger key</th><th>Validation F1</th><th>Seeds</th></tr></thead><tbody>${sorted.map(x=>`<tr><td>${escapeHtml(x.name||x.id||'Recorded candidate')}${x.accepted?' · accepted':''}</td><td>${fmt(x.validation_f1,4)} ${x.validation_f1===top?'· best recorded':''}</td><td>${x.seeds||1}</td></tr>`).join('')}</tbody></table></div></div>`}out+='</article>'}$('#families').innerHTML=out;$('#families').onclick=e=>{let b=e.target.closest('.detail-toggle');if(!b)return;let p=b.nextElementSibling;p.classList.toggle('open');b.setAttribute('aria-expanded',p.classList.contains('open'));b.textContent=(p.classList.contains('open')?'Hide':'Show')+' recorded variant scores'}}
function leader(){let list=[...M].filter(m=>familyFilter==='all'||D.cards.find(c=>c.title===familyFilter)?.models.includes(m.id)).sort((a,b)=>leaderMode==='test'?b.test_f1-a.test_f1:b.validation_f1-a.validation_f1),low=leaderMode==='test'?.45:.45,high=leaderMode==='test'?1:.98;$('#leaderboard-rows').innerHTML=list.map((m,i)=>{let val=leaderMode==='test'?m.test_f1:m.validation_f1,lo=leaderMode==='test'?m.f1_low:val,hi=leaderMode==='test'?m.f1_high:val,left=100*(lo-low)/(high-low),width=100*(hi-lo)/(high-low),bar=100*(val-low)/(high-low);return `<div class="leader"><div class="rank">${String(i+1).padStart(2,'0')}</div><div class="leader-label">${escapeHtml(m.name)}<small style="display:block;color:var(--muted);font-weight:400">${escapeHtml(m.family)}</small></div><div class="barline"><span class="${i===0?'winner':''}" style="width:${bar}%"></span>${leaderMode==='test'?`<i class="ci-line" style="left:${left}%;width:${width}%"></i>`:''}</div><div class="leader-metric"><b>${fmt(val,4)}</b><br>${leaderMode==='test'?`${fmt(m.f1_low,3)}–${fmt(m.f1_high,3)}`:`seed ${m.seed}`}</div><div class="leader-metric">AUC ${fmt(m.auc,4)}<br>FAR ${pct(m.far)}</div></div>`}).join('')}
function svgLine(points,x0,y0,w,h,xd,yd){return points.map((p,i)=>(i?'L':'M')+(x0+p[0]*w)+' '+(y0+(1-p[1])*h)).join(' ')}
function drawCurves(){const ids=['30_cnn','30_cnn_control','18_svm_rbf','20_mlp','15_random_forest'],colors=['primary','control','rbf','mlp','rf'];for(let [target,key] of [['roc','roc'],['pr','pr']]){let svg=$('#'+target),x0=40,y0=16,w=svg.clientWidth-56,h=svg.clientHeight-48;let mx=target==='roc'?1:1,my=target==='roc'?1:1;svg.setAttribute('viewBox',`0 0 ${svg.clientWidth} ${svg.clientHeight}`);let s=`<line class="axes" x1="${x0}" y1="${y0+h}" x2="${x0+w}" y2="${y0+h}"/><line class="axes" x1="${x0}" y1="${y0}" x2="${x0}" y2="${y0+h}"/>`;if(target==='roc')s+=`<line class="axes" stroke-dasharray="4 4" x1="${x0}" y1="${y0+h}" x2="${x0+w}" y2="${y0}"/>`;ids.forEach((id,i)=>{let pts=D.curves[id][key];s+=`<path class="curve ${colors[i]}" d="${svgLine(pts,x0,y0,w,h)}"><title>${byId[id].name}</title></path>`});s+=`<text x="${x0+w/2}" y="${y0+h+33}" text-anchor="middle" fill="currentColor" font-size="10">${target==='roc'?'False positive rate':'Recall'}</text><text x="12" y="${y0+h/2}" transform="rotate(-90 12 ${y0+h/2})" text-anchor="middle" fill="currentColor" font-size="10">${target==='roc'?'True positive rate':'Precision'}</text>`;svg.innerHTML=s}}
function drawSlope(){let svg=$('#slope'),w=svg.clientWidth,h=svg.clientHeight;svg.setAttribute('viewBox',`0 0 ${w} ${h}`);let left=45,right=w-35,top=18,bottom=h-30,lo=.45,hi=1,ys=v=>bottom-(v-lo)/(hi-lo)*(bottom-top),s='';for(let i=0;i<=5;i++){let v=lo+i*(hi-lo)/5,y=ys(v);s+=`<line class="axes" x1="${left}" y1="${y}" x2="${right}" y2="${y}"/><text x="${left-8}" y="${y+4}" text-anchor="end" font-size="10" fill="currentColor">${v.toFixed(2)}</text>`}s+=`<text x="${left+50}" y="${top-4}" font-size="11" fill="currentColor">Validation</text><text x="${right-40}" y="${top-4}" font-size="11" fill="currentColor">Test</text>`;M.forEach((m,i)=>{let x1=left+68,x2=right-60,y1=ys(m.validation_f1),y2=ys(m.test_f1);s+=`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="var(--line)"/><circle cx="${x1}" cy="${y1}" r="4" fill="var(--blue2)"/><circle cx="${x2}" cy="${y2}" r="4" fill="var(--red)"/><text x="${x2+9}" y="${y2+4}" font-size="10" fill="currentColor">${escapeHtml(m.name)}</text>`});svg.innerHTML=s}
function thresholdInit(){let sel=$('#model-select');sel.innerHTML=M.map(m=>`<option value="${m.id}" ${m.id==='30_cnn'?'selected':''}>${escapeHtml(m.name)}</option>`).join('');let range=$('#threshold');function update(){let m=byId[sel.value],v=D.scores[m.id],thr=Number(range.value)/1000*(Math.max(...v)-Math.min(...v))+Math.min(...v);$('#threshold-readout').textContent=`threshold ${fmt(thr,4)}`;let tp=0,fp=0,tn=0,fn=0;for(let i=0;i<v.length;i++){let pred=v[i]>=thr;if(pred&&D.labels[i])tp++;else if(pred)fp++;else if(D.labels[i])fn++;else tn++}let precision=tp/(tp+fp||1),recall=tp/(tp+fn||1),f1=2*precision*recall/(precision+recall||1),far=fp/(fp+tn||1);$('#confusion').innerHTML=[[tn,'True negative'],[fp,'False positive'],[fn,'False negative'],[tp,'True positive']].map(x=>`<div><b>${x[0]}</b><small>${x[1]}</small></div>`).join('');$('#live-metrics').innerHTML=[['Precision',precision],['Recall',recall],['F1',f1],['False alarm',far]].map(x=>`<span>${x[0]}<b>${fmt(x[1],3)}</b></span>`).join('')}sel.onchange=update;range.oninput=update;let m=byId[sel.value],v=D.scores[m.id],i=Math.round(1000*(m.threshold-Math.min(...v))/(Math.max(...v)-Math.min(...v)));range.value=Math.max(0,Math.min(1000,i));update()}
function histogram(){let v=D.scores['30_cnn'],bins=36,min=Math.min(...v),max=Math.max(...v),neg=Array(bins).fill(0),pos=Array(bins).fill(0);v.forEach((x,i)=>{let b=Math.min(bins-1,Math.floor((x-min)/(max-min)*bins));(D.labels[i]?pos:neg)[b]++});let mx=Math.max(...neg,...pos);$('#hist').innerHTML=neg.map((n,i)=>`<i class="neg" style="height:${100*n/mx}%" title="Non-drone bin ${i+1}: ${n}"></i><i class="pos" style="height:${100*pos[i]/mx}%" title="Drone bin ${i+1}: ${pos[i]}"></i>`).join('')}
function tables(){let names={'10_linear_regression':'Linear','11_sgd_regressor':'SGD','12_logistic_regression':'Logistic','13_knn':'k-NN','14_decision_tree':'Tree','15_random_forest':'Random forest','16_bagging':'Bagging','17_adaboost':'AdaBoost','18_svm_linear':'Linear SVM','18_svm_rbf':'RBF SVM','20_perceptron':'Perceptron','20_mlp':'MLP','30_cnn':'CNN','30_cnn_control':'CNN control'};let show=['30_cnn','30_cnn_control','18_svm_rbf','20_mlp','15_random_forest'];let rows=D.operating.filter(x=>show.includes(x.model)).sort((a,b)=>Number(a.target_validation_far)-Number(b.target_validation_far));$('#operating').innerHTML=`<table class="data-table"><thead><tr><th>Model · target FAR</th><th>Test FAR</th><th>Recall</th><th>Precision</th></tr></thead><tbody>${rows.map(x=>`<tr><td>${names[x.model]} · ${pct(Number(x.target_validation_far))}</td><td class="num">${pct(Number(x.test_far))}</td><td class="num">${pct(Number(x.test_recall))}</td><td class="num">${pct(Number(x.test_precision))}</td></tr>`).join('')}</tbody></table>`;let select=$('#recall-select');select.innerHTML=M.map(m=>`<option value="${m.id}" ${m.id==='30_cnn'?'selected':''}>${escapeHtml(m.name)}</option>`).join('');function renderRecall(){let id=select.value;function tableRows(data,label,key){let items=data.filter(x=>x.model===id);return `<h4>${label}</h4><table class="data-table"><thead><tr><th>Group</th><th>Clips</th><th>Groups</th><th>Recall</th><th>Note</th></tr></thead><tbody>${items.map(x=>`<tr><td>${escapeHtml(x[key])}</td><td class="num">${x.n_clips}</td><td class="num">${x.n_groups}</td><td class="num">${pct(Number(x.recall))}</td><td>${escapeHtml(x.interval_status||'')}</td></tr>`).join('')}</tbody></table>`}$('#recall-tables').innerHTML=tableRows(D.types,'By drone type','drone_type')+tableRows(D.families,'By family','family')}select.onchange=renderRecall;renderRecall()}
function costPlot(){let svg=$('#cost'),w=svg.clientWidth,h=260;svg.setAttribute('viewBox',`0 0 ${w} ${h}`);let l=48,r=w-18,t=15,b=h-40,usable=b-t,points=M.filter(m=>m.params_count&&m.train_seconds);let xs=points.map(m=>Math.log10(m.train_seconds+0.02)),xmin=Math.min(...xs),xmax=Math.max(...xs),x=m=>l+(Math.log10(m.train_seconds+.02)-xmin)/(xmax-xmin)*(r-l),y=m=>b-(m.test_f1-.65)/.32*usable;let s=`<line class="axes" x1="${l}" y1="${b}" x2="${r}" y2="${b}"/><line class="axes" x1="${l}" y1="${t}" x2="${l}" y2="${b}"/>`;points.forEach(m=>{s+=`<circle cx="${x(m)}" cy="${y(m)}" r="${Math.min(12,5+Math.sqrt(m.params_count)/80)}" fill="var(--blue2)" opacity=".7"><title>${escapeHtml(m.name)} · ${m.params_count} ${escapeHtml(m.complexity)} · ${fmt(m.train_seconds,1)} sec · F1 ${fmt(m.test_f1,3)}</title></circle><text x="${x(m)+6}" y="${y(m)-5}" font-size="9" fill="currentColor">${escapeHtml(m.name)}</text>`});s+=`<text x="${(l+r)/2}" y="${h-5}" text-anchor="middle" font-size="10" fill="currentColor">Training seconds (log scale)</text><text x="12" y="${h/2}" transform="rotate(-90 12 ${h/2})" text-anchor="middle" font-size="10" fill="currentColor">Test F1</text>`;svg.innerHTML=s}
function paired(){let rows=D.paired;let labels=(s)=>s.replaceAll('__minus__',' − ').replaceAll('_',' ');let vals=rows.flatMap(x=>[Number(x.f1_difference_ci_low),Number(x.f1_difference_ci_high)]),lim=Math.max(.2,Math.max(...vals.map(Math.abs)));$('#paired-rows').innerHTML=rows.map(x=>{let lo=Number(x.f1_difference_ci_low),hi=Number(x.f1_difference_ci_high),left=50+50*lo/lim,width=50*(hi-lo)/lim,sig=lo>0||hi<0,mid=50+50*((lo+hi)/2)/lim;return `<div class="pair-row"><div class="pair-name">${escapeHtml(labels(x.comparison))}</div><div class="forest"><span class="${sig?'sig':''}" style="left:${left}%;width:${width}%"></span><i class="${sig?'sig':''}" style="left:${mid}%"></i></div><div class="leader-metric">[${fmt(lo,3)}, ${fmt(hi,3)}]</div></div>`}).join('')}
function stage(){let steps=[['A','Frontend baseline',D.stages.history.A],['B1','Filtered conditions',D.stages.history.B1],['B2','Input variants',D.stages.history.B2],['B3','Augmentation',D.stages.history.B3],['C','Architecture search',D.stages.history.C],['D','Final seed run',D.stages.history.D]];$('#steps').innerHTML=steps.map((x,i)=>`<div class="step ${i===4?'active':''}" role="button" aria-pressed="${i===4}" tabindex="0" data-i="${i}"><strong>${x[0]} · ${x[1]}</strong><small>${i===4?'CENTREPIECE':'STAGE RECORD'}</small></div>`).join('');$('#step-copy').textContent=steps[4][2];$('#steps').onclick=$('#steps').onkeydown=e=>{let b=e.target.closest('.step');if(!b||e.type==='keydown'&&!['Enter',' '].includes(e.key))return;if(e.type==='keydown')e.preventDefault();document.querySelectorAll('.step').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b))});$('#step-copy').textContent=steps[Number(b.dataset.i)][2]};let s=D.stages,limit=s.threshold;$('#stage-chart').innerHTML=`<strong>Stage C · validation F1 across three seeds</strong><p style="font-size:11px;color:var(--muted)">Acceptance line ${fmt(limit,4)} = A0 mean + max(A0 SD, 0.01); candidate means above the line passed.</p>${[{name:'A0 default',mean:s.baseline_f1,accepted:false},...s.variants].map(x=>`<div class="stage-bar"><span>${escapeHtml(x.name)}</span><div class="track"><i class="threshold" style="left:${limit*100}%"></i><div class="fill ${x.accepted?'accepted':''}" style="width:${x.mean*100}%"></div></div><b>${fmt(x.mean,4)}</b></div>`).join('')}<p style="font-size:11px;color:var(--muted)">Combination of accepted changes: ${fmt(s.combination.candidate.mean_f1,4)}; rejected against depth_4 (0.9260). pool_2x1 canonical Stage C mean is 0.8935; 0.8960 is the separate B2 n_mels_40 seed-43 score and must not be substituted. Stage D seeds: ${s.stage_d.map(x=>`${x.seed}: epoch ${x.best_epoch}, F1 ${fmt(x.validation_f1,4)}`).join(' · ')}</p>`;$('#cnn-params').textContent=`Final CNN · ${s.params.toLocaleString()} parameters · channels 16 → 32 → 64 → 64 · batch 64 · max 100 epochs · patience 8 · no augmentation · fixed centered window · Stage D mean val F1 ${fmt(s.stage_d_mean,4)} ± ${fmt(s.stage_d_sd,4)} SD`}
function init(){makeNums();protocol();makeFilters();renderFamilies();$('#leader-filters').innerHTML=$('#filters').innerHTML;$('#leader-filters').onclick=e=>{let b=e.target.closest('button');if(!b)return;familyFilter=b.dataset.family;document.querySelectorAll('#filters button,#leader-filters button').forEach(x=>x.classList.toggle('active',x.dataset.family===familyFilter));renderFamilies();leader()};document.querySelectorAll('.seg button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.seg button').forEach(x=>x.classList.toggle('active',x===b));leaderMode=b.dataset.mode;leader()});leader();stage();tables();paired();thresholdInit();histogram();$('#hashes').textContent=`${D.test.fits} final score fits · ${D.test.groups} groups · ${D.test.score_rows.toLocaleString()} rows · scored ${D.test.scored_at} · scoring commit ${D.test.score_commit}`;let theme=localStorage.getItem('model-report-theme')||'light';document.documentElement.dataset.theme=theme;$('#theme').onclick=()=>{theme=theme==='light'?'dark':'light';document.documentElement.dataset.theme=theme;localStorage.setItem('model-report-theme',theme)};let io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){e.target.classList.add('visible');io.unobserve(e.target)}}),{threshold:.1});document.querySelectorAll('.section,.hero').forEach(x=>{x.classList.add('reveal');io.observe(x)});drawCurves();drawSlope();costPlot();window.addEventListener('resize',()=>{drawCurves();drawSlope();costPlot()})}init();
</script></body></html>'''


def main():
    # Preserve the original build command as a compatibility entry point.
    from build_model_report_site import main as build_site
    build_site()


if __name__ == "__main__":
    main()
