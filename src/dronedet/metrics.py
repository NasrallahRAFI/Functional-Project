"""Shared binary-classification metrics for the frozen drone-detection splits."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int | list[list[int]]]:
    """Score drone-positive binary labels at an explicit operating threshold.

    ``scores`` are the model's continuous ranking values (regression outputs,
    probabilities, decision values, or logits). The caller owns threshold
    selection; this function never fits or selects one.
    """
    y = np.asarray(y_true)
    score = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or score.ndim != 1 or len(y) != len(score) or not len(y):
        raise ValueError("y_true and scores must be non-empty 1D arrays of equal length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")
    if not np.isfinite(score).all():
        raise ValueError("scores must contain only finite values")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if np.unique(y).size != 2:
        raise ValueError("ROC-AUC requires both classes in y_true")

    y = y.astype(np.int8, copy=False)
    predicted = (score >= threshold).astype(np.int8)
    matrix = confusion_matrix(y, predicted, labels=(0, 1))
    tn, fp, fn, tp = (int(x) for x in matrix.ravel())
    negatives = tn + fp
    positives = tp + fn
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)),
        "false_alarm_rate": float(fp / negatives),
        "miss_rate": float(fn / positives),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def select_f1_threshold(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> float:
    """Choose the validation-score threshold with maximum F1; ties choose higher."""
    y = np.asarray(y_true)
    score = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or score.ndim != 1 or len(y) != len(score) or not len(y):
        raise ValueError("y_true and scores must be non-empty 1D arrays of equal length")
    if not np.isin(y, (0, 1)).all() or np.unique(y).size != 2:
        raise ValueError("threshold selection requires both binary classes")
    if not np.isfinite(score).all():
        raise ValueError("scores must contain only finite values")
    candidates = np.concatenate(([np.nextafter(score.max(), np.inf)], np.unique(score)))
    f1_values = np.asarray([
        f1_score(y, score >= threshold, zero_division=0)
        for threshold in candidates
    ])
    best = f1_values.max()
    return float(candidates[f1_values == best].max())


def miss_rate_at_false_alarm_rate(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    max_false_alarm_rate: float,
) -> tuple[float, float]:
    """Return (miss rate, threshold) at the best achievable FAR not above a cap.

    Among tied recall values, the larger threshold is selected. This is an
    operating-point report only; it does not alter the approved F1 selection.
    """
    y = np.asarray(y_true)
    score = np.asarray(scores, dtype=np.float64)
    if not 0.0 <= max_false_alarm_rate <= 1.0:
        raise ValueError("max_false_alarm_rate must be between 0 and 1")
    if y.ndim != 1 or score.ndim != 1 or len(y) != len(score) or not len(y):
        raise ValueError("y_true and scores must be non-empty 1D arrays of equal length")
    if not np.isin(y, (0, 1)).all() or np.unique(y).size != 2:
        raise ValueError("operating-point calculation requires both binary classes")
    if not np.isfinite(score).all():
        raise ValueError("scores must contain only finite values")

    candidates = np.concatenate(([np.nextafter(score.max(), np.inf)], np.unique(score)))
    negatives = y == 0
    positives = y == 1
    feasible = []
    for threshold in candidates:
        predicted = score >= threshold
        far = float(np.mean(predicted[negatives]))
        if far <= max_false_alarm_rate:
            recall = float(np.mean(predicted[positives]))
            feasible.append((recall, float(threshold)))
    recall, threshold = max(feasible, key=lambda item: (item[0], item[1]))
    return 1.0 - recall, threshold
