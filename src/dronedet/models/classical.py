"""Library-first scikit-learn builders for the detection-model ladder."""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def build_model(notebook: int, params: dict[str, Any] | None = None, *, seed: int = 42):
    p = dict(params or {})
    if notebook == 10:
        return LinearRegression(**p)
    if notebook == 11:
        return SGDRegressor(loss="squared_error", penalty="l2", early_stopping=False,
                            max_iter=int(p.pop("max_iter", 1000)), tol=1e-3,
                            random_state=seed, **p)
    if notebook == 12:
        penalty = p.pop("penalty", "l2")
        kwargs = {"C": float(p.pop("C", 1.0)), "class_weight": "balanced",
                  "max_iter": 1000, "random_state": 42}
        if penalty == "l1":
            return LogisticRegression(solver="liblinear", l1_ratio=1.0, **kwargs, **p)
        return LogisticRegression(solver="lbfgs", **kwargs, **p)
    if notebook == 13:
        return KNeighborsClassifier(n_neighbors=int(p.pop("k", 5)), weights=p.pop("weights", "uniform"),
                                    p=int(p.pop("p", 2)), **p)
    if notebook == 14:
        return DecisionTreeClassifier(class_weight="balanced", random_state=42,
                                      **{"max_depth": None, "min_samples_leaf": 1,
                                         "criterion": "gini", **p})
    if notebook == 15:
        return RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1,
                                      random_state=seed,
                                      **{"max_features": "sqrt", "min_samples_leaf": 1,
                                         "max_depth": None, **p})
    if notebook == 16:
        tree = p.pop("estimator", None)
        if tree is None:
            tree = DecisionTreeClassifier(class_weight="balanced", random_state=seed,
                                          **p.pop("tree_params", {}))
        return BaggingClassifier(estimator=tree, n_estimators=300, max_features=1.0,
                                 bootstrap=True, n_jobs=-1, random_state=seed,
                                 **{"max_samples": 1.0, **p})
    if notebook == 17:
        rounds = int(p.pop("n_estimators", 50))
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                                  n_estimators=rounds, learning_rate=float(p.pop("learning_rate", 1.0)),
                                  random_state=42, **p)
    if notebook == 18:
        return SVC(class_weight="balanced", probability=False, cache_size=1000,
                   random_state=42, **{"kernel": "rbf", "C": 1.0, **p})
    raise ValueError(f"unsupported classical notebook: {notebook}")


def score(estimator, X, *, kind: str):
    if kind == "regression":
        values = estimator.predict(X)
    elif kind == "probability":
        values = estimator.predict_proba(X)[:, 1]
    elif kind == "decision":
        values = estimator.decision_function(X)
    else:
        raise ValueError(f"unknown score kind: {kind}")
    return values


def score_kind(notebook: int, *, kernel: str | None = None) -> str:
    if notebook in (10, 11):
        return "regression"
    if notebook in (12, 13, 14, 15, 16):
        return "probability"
    if notebook == 17:
        return "decision"
    if notebook == 18:
        return "decision"
    raise ValueError(f"unsupported classical notebook: {notebook}")


def parameter_complexity(estimator) -> int:
    if hasattr(estimator, "tree_"):
        return int(estimator.tree_.node_count)
    if hasattr(estimator, "estimators_"):
        total = 0
        for model in estimator.estimators_:
            base = getattr(model, "tree_", None)
            if base is None and hasattr(model, "estimators_"):
                total += sum(int(tree.tree_.node_count) for tree in model.estimators_)
            elif base is not None:
                total += int(base.node_count)
        return total or int(len(estimator.estimators_))
    if hasattr(estimator, "support_"):
        return int(len(estimator.support_))
    return int(getattr(estimator, "n_features_in_", 0))


def classical_grids() -> dict[int, dict[str, list[dict[str, Any]]]]:
    """Return the owner-approved candidates; default candidates are identified explicitly."""
    return {
        10: {"default": [{}], "grid": [{}]},
        11: {"default": [{"alpha": 1e-4, "eta0": 1e-2, "learning_rate": "invscaling"}],
             "grid": [{"alpha": a, "eta0": e, "learning_rate": l}
                      for a in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
                      for e in (1e-3, 1e-2, 1e-1) for l in ("invscaling", "adaptive")]},
        12: {"default": [{"C": 1.0, "penalty": "l2"}],
             "grid": [{"C": c, "penalty": p} for c in (1e-3, 1e-2, 1e-1, 1, 10, 100)
                      for p in ("l2", "l1")]},
        13: {"default": [{"k": 5, "weights": "uniform", "p": 2}],
             "grid": [{"k": k, "weights": w, "p": p} for k in (1, 3, 5, 9, 15, 25, 51)
                      for w in ("uniform", "distance") for p in (1, 2)]},
        14: {"default": [{"max_depth": None, "min_samples_leaf": 1, "criterion": "gini"}],
             "grid": [{"max_depth": d, "min_samples_leaf": m, "criterion": "gini"}
                      for d in (2, 3, 4, 6, 8, None) for m in (1, 5, 20)]},
        15: {"default": [{"max_features": "sqrt", "min_samples_leaf": 1,
                          "max_depth": None, "class_weight": "balanced"}],
             "grid": ([{"max_features": v, "min_samples_leaf": 1, "max_depth": None,
                        "class_weight": "balanced"} for v in ("log2", 0.3)] +
                      [{"max_features": "sqrt", "min_samples_leaf": v, "max_depth": None,
                        "class_weight": "balanced"} for v in (3, 5)] +
                      [{"max_features": "sqrt", "min_samples_leaf": 1, "max_depth": v,
                        "class_weight": "balanced"} for v in (12, 24)] +
                      [{"max_features": "sqrt", "min_samples_leaf": 1, "max_depth": None,
                        "class_weight": "balanced_subsample"}])},
        18: {"linear_default": [{"kernel": "linear", "C": 1.0}],
             "linear": [{"kernel": "linear", "C": c} for c in (0.01, 0.1, 1, 10, 100)],
             "rbf_default": [{"kernel": "rbf", "C": 1.0, "gamma": 1.0}],
             "rbf": [{"kernel": "rbf", "C": c, "gamma": g}
                     for c in (0.01, 0.1, 1, 10, 100) for g in (0.1, 0.3, 1, 3, 10)]},
    }
