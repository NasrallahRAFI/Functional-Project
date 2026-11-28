"""OSQP 1.1.3 structural reference and descriptor-balanced sampler fits."""

from __future__ import annotations

import math
import platform
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import osqp
import scipy
from scipy import sparse


@dataclass(frozen=True)
class ReferenceFit:
    label: int
    clip_ids: tuple[str, ...]
    probabilities: np.ndarray
    clip_cap: float
    effective_size: float
    solver_record: dict[str, Any]


@dataclass(frozen=True)
class BalancedFit:
    clip_ids: tuple[str, ...]
    weights: np.ndarray
    dropped_margins: tuple[str, ...]
    descriptor_shares: dict[str, dict[str, float]]
    effective_sizes: dict[str, float]
    solver_record: dict[str, Any]


def solver_settings(contract: dict[str, Any]) -> dict[str, Any]:
    solver = contract["solver"]
    if solver["name"] != "OSQP" or solver["version"] != "1.1.3" or solver["backend"] != "builtin direct":
        raise ValueError("the approved V5.1 solver profile is OSQP 1.1.3 builtin direct")
    return dict(solver["settings"])


def _osqp_solve(P: sparse.csc_matrix, q: np.ndarray, A: sparse.csc_matrix,
                lower: np.ndarray, upper: np.ndarray, settings: dict[str, Any]):
    problem = osqp.OSQP()
    problem.setup(P=P, q=q, A=A, l=lower, u=upper, **settings)
    return problem.solve(raise_error=False)


def _sorted_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    values = sorted((dict(row) for row in rows), key=lambda row: str(row["clip_id"]))
    ids = [str(row["clip_id"]) for row in values]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate clip_id in a weight-fitting population")
    if any(row.get("label") not in (0, 1) or isinstance(row.get("label"), bool) for row in values):
        raise ValueError("weight-fitting labels must be binary integers")
    return values


def _validate_margin_rows(rows: list[dict[str, Any]], margin_shares: dict[str, float], *, label: int) -> list[str]:
    if not rows:
        raise ValueError(f"class {label} is empty")
    if not margin_shares or any(not math.isfinite(v) or v <= 0 for v in margin_shares.values()):
        raise ValueError("M1 shares must be positive finite values")
    if not math.isclose(math.fsum(margin_shares.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("M1 shares must sum to one")
    key_name = "source" if label == 1 else "negative_stratum"
    present = {str(row[key_name]) for row in rows}
    if present != set(margin_shares):
        raise ValueError(f"M1 margin population mismatch; absent={sorted(set(margin_shares)-present)}, unexpected={sorted(present-set(margin_shares))}")
    return [str(row[key_name]) for row in rows]


def _constraint_matrix(n: int, constraints: list[tuple[str, np.ndarray, float, float]]) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray, list[str]]:
    rr: list[int] = []
    cc: list[int] = []
    dd: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    names: list[str] = []
    for row_index, (name, coefficients, lo, hi) in enumerate(constraints):
        if coefficients.shape != (n,) or not np.isfinite(coefficients).all():
            raise ValueError(f"invalid constraint coefficient vector for {name}")
        nonzero = np.flatnonzero(coefficients)
        rr.extend([row_index] * len(nonzero))
        cc.extend(nonzero.tolist())
        dd.extend(coefficients[nonzero].astype(np.float64).tolist())
        lower.append(float(lo))
        upper.append(float(hi))
        names.append(name)
    matrix = sparse.coo_matrix((np.asarray(dd, dtype=np.float64), (rr, cc)), shape=(len(constraints), n)).tocsc()
    matrix.sort_indices()
    return matrix, np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64), names


def _max_violation(A: sparse.csc_matrix, x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    values = np.asarray(A @ x).reshape(-1)
    below = np.maximum(lo - values, 0.0)
    above = np.maximum(values - hi, 0.0)
    return float(max(np.max(below, initial=0.0), np.max(above, initial=0.0)))


def fit_structural_reference(
    rows: Iterable[dict[str, Any]],
    *,
    label: int,
    margin_shares: dict[str, float],
    source_polarity: dict[str, str],
    contract: dict[str, Any],
) -> ReferenceFit:
    """Minimize sum(q_i²) under fixed M1, source/group/clip and class-only caps."""
    if label not in (0, 1):
        raise ValueError("label must be 0 or 1")
    selected = _sorted_rows(row for row in rows if row.get("label") == label)
    if set(str(row["source"]) for row in selected) - set(source_polarity):
        raise ValueError("source polarity is missing for one or more admitted sources")
    margin_keys = _validate_margin_rows(selected, margin_shares, label=label)
    n = len(selected)
    counts = {key: margin_keys.count(key) for key in margin_shares}
    q0 = np.asarray([float(margin_shares[key]) / counts[key] for key in margin_keys], dtype=np.float64)
    clip_cap = float(contract["sampling"]["clip_cap_reference_class_median_multiple"]) * float(np.median(q0))
    sampling = contract["sampling"]
    source_values = [str(row["source"]) for row in selected]
    group_values = [str(row["group_key"]) for row in selected]

    constraints: list[tuple[str, np.ndarray, float, float]] = []
    for key in sorted(margin_shares):
        vector = np.fromiter((float(item == key) for item in margin_keys), dtype=np.float64, count=n)
        constraints.append((f"M1:{key}", vector, margin_shares[key], margin_shares[key]))
    for source in sorted(set(source_values)):
        vector = np.fromiter((float(item == source) for item in source_values), dtype=np.float64, count=n)
        constraints.append((f"source_cap:{source}", vector, 0.0, float(sampling["source_cap_per_class"])))
    for group in sorted(set(group_values)):
        vector = np.fromiter((float(item == group) for item in group_values), dtype=np.float64, count=n)
        constraints.append((f"group_cap:{group}", vector, 0.0, float(sampling["group_cap_per_class"])))
    if label == 1:
        polarity_key, cap = "positive_only", float(sampling["positive_only_aggregate_cap"])
    else:
        polarity_key, cap = "negative_only", float(sampling["negative_only_aggregate_cap"])
    only_sources = {source for source, polarity in source_polarity.items() if polarity == polarity_key}
    if only_sources:
        vector = np.fromiter((float(source in only_sources) for source in source_values), dtype=np.float64, count=n)
        constraints.append((f"{polarity_key}_aggregate", vector, 0.0, cap))

    # Variables have fixed nonnegative and clip-cap bounds. Constraint ordering
    # is deterministic: M1, source, group, aggregate, then variable bounds.
    A, lo, hi, names = _constraint_matrix(n, constraints)
    eye = sparse.eye(n, format="csc", dtype=np.float64)
    A = sparse.vstack((A, eye), format="csc")
    lo = np.r_[lo, np.zeros(n, dtype=np.float64)]
    hi = np.r_[hi, np.full(n, clip_cap, dtype=np.float64)]
    names.extend(f"clip_cap:{row['clip_id']}" for row in selected)
    settings = solver_settings(contract)
    solution = _osqp_solve(sparse.eye(n, format="csc", dtype=np.float64) * 2.0,
                           np.zeros(n, dtype=np.float64), A, lo, hi, settings)
    status = str(solution.info.status).lower()
    if status != "solved" or solution.x is None:
        raise RuntimeError(f"B0 reference QP stopped: solver status={solution.info.status}")
    probabilities = np.asarray(solution.x, dtype=np.float64)
    if not np.isfinite(probabilities).all() or float(probabilities.min(initial=0.0)) < -1e-10:
        raise RuntimeError("B0 reference QP returned invalid or materially negative probabilities")
    probabilities[(probabilities < 0) & (probabilities >= -1e-10)] = 0.0
    violation = _max_violation(A, probabilities, lo, hi)
    if violation > float(contract["solver"]["max_original_constraint_violation"]):
        raise RuntimeError(f"B0 reference QP original-unit violation {violation:.3g} exceeds limit")
    total = float(probabilities.sum())
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-8):
        raise RuntimeError(f"B0 reference class probability sum is {total:.12g}, expected 1")
    ess = 1.0 / float(np.square(probabilities).sum())
    floor = min(ess, float(sampling["effective_size"]["reference_retention"]) * ess)
    record = {
        "status": str(solution.info.status),
        "solver_version": osqp.__version__,
        "numeric_environment": {"numpy": np.__version__, "scipy": scipy.__version__, "osqp": osqp.__version__, "platform": platform.platform(), "machine": platform.machine()},
        "settings": settings,
        "variable_count": n,
        "constraint_count": len(names),
        "constraint_names_sha256": __import__("hashlib").sha256("\n".join(names).encode("utf-8")).hexdigest(),
        "objective": float(np.square(probabilities).sum()),
        "class_mass": total,
        "clip_cap": clip_cap,
        "effective_size": ess,
        "b2_minimum": floor,
        "max_original_constraint_violation": violation,
        "iterations": int(solution.info.iter),
        "primal_residual": float(solution.info.prim_res),
        "dual_residual": float(solution.info.dual_res),
    }
    return ReferenceFit(label, tuple(str(row["clip_id"]) for row in selected), probabilities, clip_cap, ess, record)


def fit_balanced_weights(
    rows: Iterable[dict[str, Any]],
    *,
    positive_source_shares: dict[str, float],
    negative_stratum_shares: dict[str, float],
    source_polarity: dict[str, str],
    positive_reference: ReferenceFit,
    negative_reference: ReferenceFit,
    contract: dict[str, Any],
) -> BalancedFit:
    """Fit M1 plus M2/M3 descriptor margins, dropping M3 then M2 only on proven infeasibility."""
    ordered = _sorted_rows(rows)
    ids = [str(row["clip_id"]) for row in ordered]
    labels = np.asarray([int(row["label"]) for row in ordered], dtype=np.int8)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("final weighting requires both classes in train")
    if set(str(row["source"]) for row in ordered) - set(source_polarity):
        raise ValueError("source polarity is missing for one or more admitted sources")
    pos_margin = _validate_margin_rows([r for r in ordered if r["label"] == 1], positive_source_shares, label=1)
    neg_margin = _validate_margin_rows([r for r in ordered if r["label"] == 0], negative_stratum_shares, label=0)
    del pos_margin, neg_margin
    sampling = contract["sampling"]
    prior = {1: float(sampling["positive_prior"]), 0: 1.0 - float(sampling["positive_prior"])}
    source = [str(row["source"]) for row in ordered]
    group = [str(row["group_key"]) for row in ordered]
    negative_stratum = [str(row.get("negative_stratum")) if row.get("negative_stratum") is not None else "" for row in ordered]
    cap_by_class = {1: positive_reference.clip_cap * prior[1], 0: negative_reference.clip_cap * prior[0]}

    m1_constraints: list[tuple[str, np.ndarray, float, float]] = []
    for label, shares, keys, margin_name in (
        (1, positive_source_shares, source, "M1:positive_source"),
        (0, negative_stratum_shares, negative_stratum, "M1:negative_stratum"),
    ):
        for key in sorted(shares):
            vector = np.fromiter((float(labels[i] == label and keys[i] == key) for i in range(len(ordered))), dtype=np.float64, count=len(ordered))
            target = prior[label] * float(shares[key])
            m1_constraints.append((f"{margin_name}:{key}", vector, target, target))
        vector = (labels == label).astype(np.float64)
        m1_constraints.append((f"class_sum:{label}", vector, prior[label], prior[label]))

    aggregate_constraints: list[tuple[str, np.ndarray, float, float]] = []
    for label, class_only, cap, cap_name in (
        (1, "positive_only", float(sampling["positive_only_aggregate_cap"]), "positive_only_aggregate"),
        (0, "negative_only", float(sampling["negative_only_aggregate_cap"]), "negative_only_aggregate"),
    ):
        only_sources = {s for s, polarity in source_polarity.items() if polarity == class_only}
        if only_sources:
            vector = np.fromiter((float(labels[i] == label and source[i] in only_sources) for i in range(len(ordered))), dtype=np.float64, count=len(ordered))
            aggregate_constraints.append((cap_name, vector, 0.0, prior[label] * cap))

    structural_constraints: list[tuple[str, np.ndarray, float, float]] = []
    for label in (0, 1):
        for value in sorted({source[i] for i in range(len(ordered)) if labels[i] == label}):
            vector = np.fromiter((float(labels[i] == label and source[i] == value) for i in range(len(ordered))), dtype=np.float64, count=len(ordered))
            structural_constraints.append((f"source_cap:{label}:{value}", vector, 0.0, prior[label] * float(sampling["source_cap_per_class"])))
        for value in sorted({group[i] for i in range(len(ordered)) if labels[i] == label}):
            vector = np.fromiter((float(labels[i] == label and group[i] == value) for i in range(len(ordered))), dtype=np.float64, count=len(ordered))
            structural_constraints.append((f"group_cap:{label}:{value}", vector, 0.0, prior[label] * float(sampling["group_cap_per_class"])))

    descriptor_constraints: dict[str, list[tuple[str, np.ndarray, float, float]]] = {"M2": [], "M3": []}
    for margin, column in (("M2", "rolloff_decile"), ("M3", "rms_decile")):
        raw_values = [row[column] for row in ordered if row.get(column) is not None]
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in raw_values):
            raise ValueError(f"{margin} descriptor deciles must be integer values")
        values = sorted({int(value) for value in raw_values})
        for value in values:
            pos = np.fromiter((float(labels[i] == 1 and ordered[i].get(column) == value) for i in range(len(ordered))), dtype=np.float64, count=len(ordered))
            neg = np.fromiter((float(labels[i] == 0 and ordered[i].get(column) == value) for i in range(len(ordered))), dtype=np.float64, count=len(ordered))
            coefficients = prior[0] * pos - prior[1] * neg
            descriptor_constraints[margin].append((f"{margin}:{value}", coefficients, 0.0, 0.0))

    settings = solver_settings(contract)
    objective_diagonal = np.asarray([2.0 / (prior[int(label)] ** 2) for label in labels], dtype=np.float64)
    P = sparse.diags(objective_diagonal, format="csc")
    q = np.zeros(len(ordered), dtype=np.float64)
    dropped: list[str] = []
    solution = None
    solver_attempts = []
    for removed in (set(), {"M3"}, {"M3", "M2"}):
        active = list(m1_constraints)
        for margin in ("M2", "M3"):
            if margin not in removed:
                active.extend(descriptor_constraints[margin])
        active.extend(structural_constraints)
        active.extend(aggregate_constraints)
        bounds = np.asarray([cap_by_class[int(label)] for label in labels], dtype=np.float64)
        eye = sparse.eye(len(ordered), format="csc", dtype=np.float64)
        A0, lo0, hi0, names = _constraint_matrix(len(ordered), active)
        A = sparse.vstack((A0, eye), format="csc")
        lo = np.r_[lo0, np.zeros(len(ordered), dtype=np.float64)]
        hi = np.r_[hi0, bounds]
        names.extend(f"clip_cap:{ids[i]}" for i in range(len(ids)))
        candidate = _osqp_solve(P, q, A, lo, hi, settings)
        status = str(candidate.info.status).lower()
        solver_attempts.append({"removed": sorted(removed), "status": str(candidate.info.status), "iterations": int(candidate.info.iter)})
        if status == "solved" and candidate.x is not None:
            solution = candidate
            active_A, active_lo, active_hi, active_names = A, lo, hi, names
            dropped = sorted(removed)
            break
        if status != "primal infeasible":
            raise RuntimeError(f"descriptor-weight QP numerical failure; stop without margin relaxation: {candidate.info.status}")
        if removed == {"M3", "M2"}:
            raise RuntimeError("M1-only reference weight fit is infeasible")
    if solution is None:
        raise RuntimeError("descriptor-weight solver produced no usable solution")

    weights = np.asarray(solution.x, dtype=np.float64)
    if not np.isfinite(weights).all() or float(weights.min(initial=0.0)) < -1e-10:
        raise RuntimeError(
            "descriptor-weight QP returned invalid or materially negative weights: "
            f"minimum={float(weights.min(initial=0.0)):.17g}, status={solution.info.status}, "
            f"iterations={solution.info.iter}, primal_residual={solution.info.prim_res:.6g}, "
            f"dual_residual={solution.info.dual_res:.6g}, removed_margins={dropped}"
        )
    weights[(weights < 0) & (weights >= -1e-10)] = 0.0
    violation = _max_violation(active_A, weights, active_lo, active_hi)
    if violation > float(contract["solver"]["max_original_constraint_violation"]):
        raise RuntimeError(f"descriptor-weight QP original-unit violation {violation:.3g} exceeds limit")

    shares: dict[str, dict[str, float]] = {}
    for margin, column in (("rolloff_decile", "rolloff_decile"), ("rms_decile", "rms_decile")):
        summaries = {}
        for numeric_value in sorted({int(row[column]) for row in ordered if row.get(column) is not None}):
            value = str(numeric_value)
            selected_idx = [i for i, row in enumerate(ordered) if row.get(column) == numeric_value]
            positive_mass = float(weights[[i for i in selected_idx if labels[i] == 1]].sum())
            negative_mass = float(weights[[i for i in selected_idx if labels[i] == 0]].sum())
            total_mass = positive_mass + negative_mass
            fraction = positive_mass / total_mass if total_mass > 0 else None
            summaries[value] = fraction
        shares[margin] = summaries

    ess: dict[str, float] = {}
    for label, name in ((1, "positive"), (0, "negative")):
        class_weights = weights[labels == label] / prior[label]
        class_sum = float(class_weights.sum())
        if not math.isclose(class_sum, 1.0, rel_tol=0, abs_tol=1e-8):
            raise RuntimeError(f"final {name} class weights sum to {class_sum:.12g}, expected 1")
        ess[name] = 1.0 / float(np.square(class_weights).sum())
    retention = float(sampling["effective_size"]["reference_retention"])
    reference_floors = {
        "positive": min(positive_reference.effective_size, retention * positive_reference.effective_size),
        "negative": min(negative_reference.effective_size, retention * negative_reference.effective_size),
    }
    descriptor_bounds = tuple(float(value) for value in sampling["descriptor_positive_fraction_bounds"])
    descriptor_balance_pass = bool(shares["rolloff_decile"] and shares["rms_decile"]) and all(
        value is not None and descriptor_bounds[0] <= value <= descriptor_bounds[1]
        for values in shares.values() for value in values.values()
    )

    record = {
        "status": str(solution.info.status),
        "solver_version": osqp.__version__,
        "numeric_environment": {"numpy": np.__version__, "scipy": scipy.__version__, "osqp": osqp.__version__, "platform": platform.platform(), "machine": platform.machine()},
        "settings": settings,
        "variable_count": len(ordered),
        "constraint_count": len(active_names),
        "constraint_names_sha256": __import__("hashlib").sha256("\n".join(active_names).encode("utf-8")).hexdigest(),
        "objective": float(np.sum(np.square(weights[labels == 1] / prior[1])) + np.sum(np.square(weights[labels == 0] / prior[0]))),
        "class_masses": {"positive": float(weights[labels == 1].sum()), "negative": float(weights[labels == 0].sum())},
        "effective_sizes": ess,
        "reference_floors": reference_floors,
        "effective_size_ratios": {key: ess[key] / reference.effective_size for key, reference in (("positive", positive_reference), ("negative", negative_reference))},
        "b2_pass": all(ess[key] + 1e-10 >= reference_floors[key] for key in ess),
        "dropped_margins": dropped,
        "attempts": solver_attempts,
        "descriptor_positive_shares": shares,
        "b1_descriptor_balance_pass": descriptor_balance_pass,
        "max_original_constraint_violation": violation,
        "iterations": int(solution.info.iter),
        "primal_residual": float(solution.info.prim_res),
        "dual_residual": float(solution.info.dual_res),
    }
    return BalancedFit(tuple(ids), weights, tuple(dropped), shares, ess, record)
