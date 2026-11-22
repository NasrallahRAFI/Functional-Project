"""Validation-only model selection and final comparison metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def _binary(y: Iterable[int], scores: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(y, dtype=np.int8).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if target.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("labels and finite scores must have identical one-dimensional shapes")
    if not np.isin(target, (0, 1)).all():
        raise ValueError("labels must be binary with drone=1")
    return target, values


def predict_at_threshold(scores: Iterable[float], threshold: float) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError("scores and threshold must be finite")
    return (values >= threshold).astype(np.int8)


def choose_f1_threshold(y: Iterable[int], scores: Iterable[float]) -> dict[str, float]:
    """Maximise validation F1 over unique scores; exact ties choose larger threshold."""
    target, values = _binary(y, scores)
    if np.unique(target).size != 2:
        raise ValueError("threshold selection requires both classes")
    best_f1, best_threshold = -1.0, -np.inf
    for threshold in np.unique(values):
        pred = values >= threshold
        tp = int(np.count_nonzero((target == 1) & pred))
        fp = int(np.count_nonzero((target == 0) & pred))
        fn = int(np.count_nonzero((target == 1) & ~pred))
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        if f1 > best_f1 or (f1 == best_f1 and threshold > best_threshold):
            best_f1, best_threshold = float(f1), float(threshold)
    auc = float(roc_auc_score(target, values))
    return {"threshold": best_threshold, "f1": best_f1, "roc_auc": auc}


def metrics_at_threshold(y: Iterable[int], scores: Iterable[float], threshold: float) -> dict[str, Any]:
    target, values = _binary(y, scores)
    pred = predict_at_threshold(values, threshold)
    tn, fp, fn, tp = confusion_matrix(target, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold), "f1": float(f1_score(target, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, values)),
        "precision": float(precision_score(target, pred, zero_division=0)),
        "recall": float(recall_score(target, pred, zero_division=0)),
        "false_alarm_rate": float(fp / (fp + tn)) if fp + tn else float("nan"),
        "miss_rate": float(fn / (fn + tp)) if fn + tp else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def operating_threshold(y: Iterable[int], scores: Iterable[float], target_far: float) -> float:
    """Smallest observed score whose validation false-alarm rate is within target."""
    target, values = _binary(y, scores)
    if not 0 <= target_far <= 1 or not np.any(target == 0):
        raise ValueError("target FAR must be in [0,1] and validation needs negatives")
    candidates = np.unique(values)
    for threshold in candidates:
        pred_negative = values[target == 0] >= threshold
        far = float(np.mean(pred_negative))
        if far <= target_far:
            return float(threshold)
    raise RuntimeError("no observed threshold satisfies the requested FAR")


def replace_rule(default_mean: float, default_sd: float, candidate_mean: float,
                 *, deterministic: bool = False) -> bool:
    margin = 0.0 if deterministic else float(default_sd)
    return candidate_mean - default_mean > margin and candidate_mean - default_mean >= 0.01


def choose_configuration(default: dict[str, Any], candidates: list[dict[str, Any]],
                         *, deterministic: bool = False) -> dict[str, Any]:
    """Apply rule R; candidates carry mean_f1, sd_f1, complexity and grid_order."""
    base = dict(default)
    base.setdefault("grid_order", 0)
    passing = [dict(c) for c in candidates if replace_rule(
        float(default["mean_f1"]), float(default.get("sd_f1", 0.0)),
        float(c["mean_f1"]), deterministic=deterministic)]
    if not passing:
        base["tie_with_default"] = True
        return base
    passing.sort(key=lambda c: (-round(float(c["mean_f1"]), 6),
                                float(c.get("complexity", float("inf"))),
                                int(c.get("grid_order", 0))))
    passing[0]["tie_with_default"] = False
    return passing[0]


_MODE_COMPLEXITY = {"global_stats": 0, "per_window_mean": 1, "pcen": 2}


def select_mode(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the registered B1 0 dB guard, worst-condition score and simplicity tie."""
    if not candidates:
        raise ValueError("select_mode needs at least one candidate")
    best_zero = max(float(c["mean_f1_0db"]) for c in candidates)
    eligible = [dict(c) for c in candidates if float(c["mean_f1_0db"]) >= best_zero - 0.02]
    best_score = max(float(c["score"]) for c in eligible)
    best_row = next(c for c in eligible if float(c["score"]) == best_score)
    margin = max(0.01, float(best_row.get("sd_f1", 0.0)))
    tied = [c for c in eligible if float(c["score"]) >= best_score - margin]
    winner = min(tied, key=lambda c: _MODE_COMPLEXITY[str(c["mode"])])
    return {**winner, "eligible_modes": [c["mode"] for c in eligible],
            "tied_modes": [c["mode"] for c in tied], "tie_margin": margin}


def select_variant(base: dict[str, Any], variants: list[dict[str, Any]]) -> dict[str, Any]:
    """Select B2 variants; combination evaluation remains a separate registered fit."""
    guard = float(base["mean_f1_0db"]) - 0.02
    sd = float(base.get("sd_f1", 0.0))
    accepted = [dict(v) for v in variants
                if float(v["mean_f1_0db"]) >= guard
                and float(v["score"]) - float(base["score"]) > sd
                and float(v["score"]) - float(base["score"]) >= 0.01]
    accepted.sort(key=lambda v: (-float(v["score"]), int(v.get("grid_order", 0))))
    return {"accepted": accepted, "winner": accepted[0] if accepted else dict(base),
            "combination_required": len(accepted) >= 2}


def group_bootstrap(y: Iterable[int], scores: Iterable[float], groups: Iterable[str], *,
                    threshold: float, draws: int = 1000, seed: int = 42) -> dict[str, Any]:
    target, values = _binary(y, scores)
    group = np.asarray(groups).astype(str).reshape(-1)
    if len(group) != len(target) or draws < 1:
        raise ValueError("groups must align with labels and draws must be positive")
    unique, inverse = np.unique(group, return_inverse=True)
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in ("f1", "roc_auc", "precision", "recall", "false_alarm_rate", "miss_rate")}
    redrawn = 0
    while len(samples["f1"]) < draws:
        mult = np.bincount(rng.integers(0, len(unique), size=len(unique)), minlength=len(unique))
        weights = mult[inverse]
        weighted_classes = np.unique(target[weights > 0])
        if weighted_classes.size < 2:
            redrawn += 1
            continue
        pred = predict_at_threshold(values, threshold)
        tn = float(weights[(target == 0) & (pred == 0)].sum())
        fp = float(weights[(target == 0) & (pred == 1)].sum())
        fn = float(weights[(target == 1) & (pred == 0)].sum())
        tp = float(weights[(target == 1) & (pred == 1)].sum())
        samples["f1"].append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        samples["roc_auc"].append(float(roc_auc_score(target, values, sample_weight=weights)))
        samples["precision"].append(tp / (tp + fp) if tp + fp else 0.0)
        samples["recall"].append(tp / (tp + fn) if tp + fn else 0.0)
        samples["false_alarm_rate"].append(fp / (fp + tn) if fp + tn else float("nan"))
        samples["miss_rate"].append(fn / (fn + tp) if fn + tp else float("nan"))
    intervals = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
                 for k, v in samples.items()}
    return {"draws": draws, "seed": seed, "group_count": int(len(unique)),
            "single_class_redraws": redrawn, "intervals": intervals}


def paired_group_bootstrap(y: Iterable[int], groups: Iterable[str],
                           scores_by_model: dict[str, Iterable[float]], *,
                           thresholds: dict[str, float], pairs: list[tuple[str, str]],
                           draws: int = 1000, seed: int = 42) -> dict[str, Any]:
    target = np.asarray(y, dtype=np.int8).reshape(-1)
    group = np.asarray(groups).astype(str).reshape(-1)
    unique, inverse = np.unique(group, return_inverse=True)
    scores = {k: _binary(target, v)[1] for k, v in scores_by_model.items()}
    rng = np.random.default_rng(seed)
    diffs = {f"{a}__minus__{b}": [] for a, b in pairs}
    redrawn = 0
    while len(next(iter(diffs.values()))) < draws:
        weights = np.bincount(rng.integers(0, len(unique), size=len(unique)), minlength=len(unique))[inverse]
        if np.unique(target[weights > 0]).size < 2:
            redrawn += 1
            continue
        metrics = {}
        for model, values in scores.items():
            pred = values >= thresholds[model]
            tp = weights[(target == 1) & pred].sum(); fp = weights[(target == 0) & pred].sum()
            fn = weights[(target == 1) & ~pred].sum()
            metrics[model] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        for a, b in pairs:
            diffs[f"{a}__minus__{b}"].append(float(metrics[a] - metrics[b]))
    return {"draws": draws, "seed": seed, "group_count": int(len(unique),),
            "single_class_redraws": redrawn,
            "intervals": {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                          for k, v in diffs.items()}}


def selfcheck() -> dict[str, Any]:
    """Synthetic protocol checks and locked-test loader guard checks."""
    y = np.asarray([0, 0, 1, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.4, 0.4, 0.9])
    selected = choose_f1_threshold(y, scores)
    brute = []
    for threshold in np.unique(scores):
        pred = scores >= threshold
        tp = np.sum((y == 1) & pred); fp = np.sum((y == 0) & pred); fn = np.sum((y == 1) & ~pred)
        brute.append((2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0, threshold))
    brute_best = max(brute, key=lambda item: (item[0], item[1]))
    if (selected["f1"], selected["threshold"]) != brute_best:
        raise AssertionError("F1 threshold selection differs from brute force")
    from .splits import TestAccessError, load_split
    from .features import get_window, load_cache_index, load_power_cache, load_tabular
    guarded_calls = {
        "manifest": lambda: load_split({}, "test"),
        "tabular": lambda: load_tabular({}, "test"),
        "mel_cache": lambda: load_power_cache({}, "test"),
        "index": lambda: load_cache_index({}, "test"),
        "window": lambda: get_window(None, None, None, split="test"),
    }
    for loader, call in guarded_calls.items():
        try:
            call()
        except TestAccessError:
            continue
        raise AssertionError(f"test guard did not raise for {loader}")
    import json
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    prohibited = re.compile(r"(?:test" + r"\.csv|mel_power" + "_test|features_tabular/" +
                            "test|allow_test\\s*=\\s*" + "True)")
    scanned = []
    violations = []
    source_root = root / "src" / "dronedet"
    for path in source_root.rglob("*.py"):
        scanned.append(str(path.relative_to(root)))
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if prohibited.search(line):
                violations.append({"path": str(path.relative_to(root)), "line": line_no,
                                   "match": prohibited.search(line).group(0)})
    notebook_root = root / "notebooks"
    for number in range(10, 31):
        path = notebook_root / f"{number:02d}_*.ipynb"
        matches = list(notebook_root.glob(f"{number:02d}_*.ipynb"))
        for nb_path in matches:
            if number == 10:
                nb = json.loads(nb_path.read_text(encoding="utf-8"))
                title = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", [])[:1])
                if "Phase 0 Colab bring-up" in title:
                    # C0.6/C0.7 explicitly permit test-manifest byte hashing and path existence checks.
                    scanned.append(f"{nb_path.relative_to(root)} (Phase 0 preflight exception)")
                    continue
            scanned.append(str(nb_path.relative_to(root)))
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            for cell_no, cell in enumerate(nb.get("cells", [])):
                text = "".join(cell.get("source", []))
                if prohibited.search(text):
                    violations.append({"path": str(nb_path.relative_to(root)), "cell": cell_no,
                                       "match": prohibited.search(text).group(0)})
    if violations:
        raise AssertionError(f"test-access source scan found unguarded references: {violations}")
    return {"threshold_selection": selected, "brute_force": brute_best,
            "test_guard_loaders_checked": 5, "test_access_scan": scanned,
            "test_access_violations": violations, "pass": True}
