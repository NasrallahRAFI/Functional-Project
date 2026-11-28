"""V5.1 group assignment, split gates and reservoir selection.

The module reuses only the unchanged two-stage SGKF numerical helper. It does
not call V4 split orchestration, split_gates, training, or test-loading code.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
import sklearn

from dronedet.splits import two_stage_split

from .randomness import canonical_key, keyed_sha1


@dataclass(frozen=True)
class SplitAssignment:
    rows: tuple[dict[str, Any], ...]
    group_splits: dict[str, str]
    input_order_sha256: str
    sklearn_version: str
    counts: dict[str, dict[str, int]]


def _ordered(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = sorted((dict(row) for row in rows), key=lambda row: str(row["clip_id"]))
    ids = [str(row["clip_id"]) for row in result]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate clip_id before V5.1 split assignment")
    if any(isinstance(row.get("label"), bool) or row.get("label") not in (0, 1) for row in result):
        raise ValueError("V5.1 split labels must be binary integers")
    return result


def _pool_stratum(row: dict[str, Any]) -> str:
    source = str(row["source"])
    label = int(row["label"])
    substratum = str(row["substratum"])
    if not substratum:
        raise ValueError(f"missing split substratum for {row['clip_id']}")
    if source == "mic":
        if label != 1 or substratum != "folder_positive":
            raise ValueError(f"MIC pooled strata must use folder_positive at {row['clip_id']}")
    return f"{source}|{label}|{substratum}"


def _rank_group_records(
    file_rows: Iterable[dict[str, Any]],
    *,
    purpose: str,
    group_field: str,
    key_fields: tuple[str, str],
    fractions: tuple[float, float, float],
) -> dict[str, str]:
    """Rank unique groups, then apply floor(train), floor(val), remainder=test."""
    rows = [dict(row) for row in file_rows]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = str(row[group_field])
        if not group:
            raise ValueError(f"empty group value for {purpose}")
        groups[group].append(row)
    ranked_groups: list[tuple[str, str, str]] = []
    for group, members in groups.items():
        keys = {
            canonical_key([str(member[key_fields[0]]), str(member[key_fields[1]])])
            for member in members
        }
        if len(keys) != 1:
            raise ValueError(f"split group {group!r} maps to multiple rank keys for {purpose}")
        canonical = next(iter(keys))
        rank = keyed_sha1(purpose, [str(members[0][key_fields[0]]), str(members[0][key_fields[1]])])
        ranked_groups.append((rank, canonical, group))
    ranked_groups.sort()
    count = len(ranked_groups)
    n_train = math.floor(fractions[0] * count)
    n_val = math.floor(fractions[1] * count)
    assignment: dict[str, str] = {}
    for index, (_, _, group) in enumerate(ranked_groups):
        if index < n_train:
            split = "train"
        elif index < n_train + n_val:
            split = "val"
        else:
            split = "test"
        assignment[group] = split
    return assignment


def assign_splits(
    candidate_rows: Iterable[dict[str, Any]],
    source_file_rows: Iterable[dict[str, Any]],
    *,
    mic_pool_approved: bool = False,
) -> SplitAssignment:
    """Assign pooled groups and independently ranked MUSAN/Cochl file groups.

    Candidate roles are retained in the input, but only kept base rows
    determine SGKF. Reservoir candidates inherit their source-file/group split.
    MIC is OOD unless O5 is explicitly passed as approved.
    """
    rows = _ordered(candidate_rows)
    files = [dict(row) for row in source_file_rows]
    membership_sources = {"ae", "dd", "itu"} | ({"mic"} if mic_pool_approved else set())
    base_roles = {"base", "base_and_reservoir"}
    pooled = [row for row in rows if row["source"] in membership_sources
              and row.get("kept") is True and row.get("candidate_role") in base_roles]
    if not pooled:
        raise ValueError("no retained pooled base candidates")

    pinned_groups = {str(row["group_key"]) for row in pooled if row.get("pinned") is True}
    group_splits: dict[str, str] = {group: "train" for group in sorted(pinned_groups)}
    free = [row for row in pooled if str(row["group_key"]) not in pinned_groups]
    if free:
        frame = pd.DataFrame([
            {
                "clip_id": str(row["clip_id"]),
                "group": str(row["group_key"]),
                "stratum": _pool_stratum(row),
            }
            for row in free
        ])
        frame["label"] = frame["stratum"]
        assigned = two_stage_split(frame, 51, n_test=7, n_val=6, label_column="label")
        for group, subframe in frame.assign(split=assigned.to_numpy()).groupby("group", sort=True):
            splits = set(subframe["split"].astype(str))
            if len(splits) != 1:
                raise ValueError(f"group crosses SGKF splits: {group}")
            group_splits[str(group)] = next(iter(splits))

    # Every retained pooled row, including unused reservoir candidates, inherits
    # the split of its source-qualified group.
    pooled_groups = {
        str(row["group_key"])
        for row in rows
        if row["source"] in membership_sources
        and row.get("kept") is True
        and row.get("candidate_role") in base_roles
    }
    absent = pooled_groups - set(group_splits)
    if absent:
        raise ValueError(f"pooled candidate groups have no base assignment: {sorted(absent)[:5]}")

    musan_files = [row for row in files if row.get("source") == "musan"]
    musan_groups: dict[str, str] = {}
    by_collection: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in musan_files:
        collection = str(row.get("source_class", "")).casefold()
        if collection not in {"noise", "music", "speech"}:
            raise ValueError(f"invalid MUSAN subcollection for {row.get('relative_path')}")
        by_collection[collection].append(row)
    for collection, members in sorted(by_collection.items()):
        fractions = (0.60, 0.20, 0.20) if collection == "noise" else (0.80, 0.10, 0.10)
        musan_groups.update(_rank_group_records(
            members,
            purpose="split_musan",
            group_field="group_key",
            key_fields=("source_revision", "relative_path"),
            fractions=fractions,
        ))

    cochl_files = [row for row in files if row.get("source") == "cochl"]
    cochl_groups = _rank_group_records(
        cochl_files,
        purpose="split_cochl",
        group_field="group_key",
        # group_key is the canonical, source-qualified UserID value.
        key_fields=("source_revision", "group_key"),
        fractions=(0.60, 0.20, 0.20),
    ) if cochl_files else {}
    loc_files = [row for row in files if row.get("source") == "loc"]
    if loc_files:
        raise ValueError("local-session split requires a separately recorded O1-B/C decision")

    by_source = {"musan": musan_groups, "cochl": cochl_groups}
    output: list[dict[str, Any]] = []
    for row in rows:
        source = str(row["source"])
        group = str(row["group_key"])
        if source in membership_sources:
            split = group_splits.get(group)
        elif source in by_source:
            split = by_source[source].get(group)
        elif source in {"mic", "wang"}:
            split = "ood"
        elif source == "loc":
            split = None
        else:
            raise ValueError(f"no V5.1 split rule for source {source!r}")
        if row.get("kept") is True and split is None:
            raise ValueError(f"retained candidate group has no split: {source}:{group}")
        output.append({**row, "split": split})

    # two_stage_split receives the sorted unpinned candidates only.
    order_payload = "\n".join(str(row["clip_id"]) for row in free)
    order_hash = hashlib.sha256(order_payload.encode("utf-8")).hexdigest()
    counts: dict[str, dict[str, int]] = {}
    for row in output:
        if row.get("kept") is True and row.get("candidate_role") in base_roles and row.get("split") is not None:
            source = str(row["source"])
            key = f"{source}:{row['split']}"
            counts.setdefault(source, {}).setdefault(str(row["split"]), 0)
            counts[source][str(row["split"])] += 1
    return SplitAssignment(tuple(output), group_splits, order_hash, sklearn.__version__, counts)


def check_split_gates(
    assignment: SplitAssignment,
    reservoir_rows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return contract S1–S8 gate results; never call V4 split_gates."""
    rows = [dict(row) for row in assignment.rows]
    base_roles = {"base", "base_and_reservoir"}
    kept = [row for row in rows if row.get("kept") is True and row.get("candidate_role") in base_roles]
    gates: dict[str, dict[str, Any]] = {}

    by_group: dict[str, set[str]] = defaultdict(set)
    by_sample: dict[str, set[str]] = defaultdict(set)
    for row in kept:
        split = str(row["split"])
        by_group[str(row["group_key"])].add(split)
        by_sample[str(row["sample_sha256"])].add(split)
    gates["S1_GROUP_EXCLUSIVITY"] = {
        "state": "PASS" if all(len(value) == 1 for value in by_group.values()) else "FAIL",
        "crossing_groups": sorted(key for key, value in by_group.items() if len(value) != 1),
    }
    gates["S2_SAMPLE_ID_EXCLUSIVITY"] = {
        "state": "PASS" if all(len(value) == 1 for value in by_sample.values()) else "FAIL",
        "crossing_sample_hashes": sorted(key for key, value in by_sample.items() if len(value) != 1),
    }
    pooled = [row for row in kept if row["source"] in {"ae", "dd", "itu"} or
              (row["source"] == "mic" and row.get("mic_pool_approved") is True)]
    for split in ("train", "val", "test"):
        labels = {int(row["label"]) for row in pooled if row["split"] == split}
        gates.setdefault("S3_BOTH_LABELS", {"state": "PASS", "missing": []})
        if labels != {0, 1}:
            gates["S3_BOTH_LABELS"]["state"] = "FAIL"
            gates["S3_BOTH_LABELS"]["missing"].append(split)
    positive_group_counts: dict[str, int] = {}
    for split in ("val", "test"):
        positive_group_counts[split] = len({
            str(row["group_key"]) for row in pooled if row["split"] == split and int(row["label"]) == 1
        })
    gates["S4_POSITIVE_GROUPS"] = {
        "state": "PASS" if all(n >= 25 for n in positive_group_counts.values()) else "FAIL",
        "groups": positive_group_counts,
    }
    source_groups: dict[str, set[str]] = defaultdict(set)
    for row in pooled:
        source_groups[str(row["source"])].add(str(row["group_key"]))
    missing_source_splits = [
        {"source": source, "split": split}
        for source, groups in sorted(source_groups.items()) if len(groups) >= 6
        for split in ("val", "test")
        if not any(row["source"] == source and row["split"] == split for row in pooled)
    ]
    gates["S5_SOURCE_REPRESENTATION"] = {
        "state": "PASS" if not missing_source_splits else "FAIL",
        "missing": missing_source_splits,
    }
    pinned_moved = [
        str(row["clip_id"]) for row in kept
        if row.get("pinned") is True and row["split"] != "train"
    ]
    gates["S6_PINNED_TRAIN"] = {
        "state": "PASS" if not pinned_moved else "FAIL",
        "violations": pinned_moved,
    }
    ood_leaks = [
        str(row["clip_id"]) for row in rows
        if row.get("source") in {"mic", "wang"} and row.get("kept") is True
        and row.get("split") != "ood"
    ]
    non_ood_roles = [
        str(row["clip_id"]) for row in rows
        if row.get("source") in {"mic", "wang"}
        and row.get("candidate_role") in {"reservoir", "reservoir_candidate", "base_and_reservoir"}
    ]
    gates["S7_OOD_BOUNDARY"] = {
        "state": "PASS" if not ood_leaks and not non_ood_roles else "FAIL",
        "wrong_split": ood_leaks,
        "reservoir_roles": non_ood_roles,
    }
    if reservoir_rows is None:
        gates["S8_RESERVOIR_TRAIN_ONLY"] = {"state": "NOT_RUN", "violations": []}
    else:
        train_musan_groups = {
            str(row["group_key"]) for row in rows
            if row.get("source") == "musan" and row.get("split") == "train"
        }
        reservoir = [dict(row) for row in reservoir_rows]
        reservoir_bad = [
            str(row["clip_id"]) for row in reservoir
            if row.get("candidate_role") != "reservoir" or row.get("source") != "musan"
            or row.get("split") != "train" or str(row.get("group_key")) not in train_musan_groups
        ]
        gates["S8_RESERVOIR_TRAIN_ONLY"] = {
            "state": "PASS" if not reservoir_bad else "FAIL",
            "violations": reservoir_bad,
            "rows": len(reservoir),
        }
    return gates


def select_train_reservoir(
    assignment: SplitAssignment,
    *,
    maximum_rows: int = 200_000,
) -> tuple[dict[str, Any], ...]:
    """Select only unused retained candidates from pooled/MUSAN train groups."""
    if maximum_rows != 200_000:
        raise ValueError("V5.1 reservoir cap is fixed at 200,000")
    rows = [dict(row) for row in assignment.rows]
    base = [row for row in rows if row.get("kept") is True and row.get("candidate_role") in {"base", "base_and_reservoir"}]
    base_hashes = {str(row["sample_sha256"]) for row in base}
    candidates = [
        row for row in rows
        if row.get("source") == "musan" and row.get("kept") is True
        and row.get("candidate_role") == "reservoir_candidate"
        and row.get("split") == "train"
    ]
    selected: list[tuple[str, str, dict[str, Any]]] = []
    seen_hashes = set(base_hashes)
    for row in candidates:
        sample_hash = str(row["sample_sha256"])
        if sample_hash in seen_hashes:
            continue
        seen_hashes.add(sample_hash)
        key = [str(row["source_revision"]), str(row["clip_id"])]
        selected.append((keyed_sha1("reservoir", key), canonical_key(key), row))
    selected.sort(key=lambda item: (item[0], item[1], str(item[2]["clip_id"])))
    result = []
    for _, _, row in selected[:maximum_rows]:
        result.append({**row, "candidate_role": "reservoir", "split": "train"})
    return tuple(result)


def apply_cochl_caps(candidate_rows: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Apply the fixed Cochl per-user and post-split caps by contract rank."""
    rows = [dict(row) for row in candidate_rows]
    if any(row.get("source") != "cochl" for row in rows):
        raise ValueError("Cochl caps accept only Cochl candidates")
    for row in rows:
        if row.get("split") not in {"train", "val", "test"}:
            raise ValueError(f"Cochl candidate has no permitted assigned split: {row.get('clip_id')}")

    eligible = [row for row in rows if row.get("kept") is True]
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_user[str(row["group_key"])].append(row)
    rejected: dict[str, str] = {}
    for user, members in by_user.items():
        ranked_members = sorted(
            members,
            key=lambda row: (
                keyed_sha1("cap", ["cochl", f"user:{user}", str(row["clip_id"])]),
                str(row["clip_id"]),
            ),
        )
        rejected.update({str(row["clip_id"]): "cap_user" for row in ranked_members[12:]})

    remaining = [row for row in eligible if str(row["clip_id"]) not in rejected]
    per_split = {"train": 6_000, "val": 1_200, "test": 1_200}
    for split, limit in per_split.items():
        split_rows = [row for row in remaining if row["split"] == split]
        ranked_members = sorted(
            split_rows,
            key=lambda row: (
                keyed_sha1("cap", ["cochl", f"split:{split}", str(row["clip_id"])]),
                str(row["clip_id"]),
            ),
        )
        rejected.update({str(row["clip_id"]): "cap_split" for row in ranked_members[limit:]})

    output = []
    for row in rows:
        if row.get("kept") is True and str(row["clip_id"]) in rejected:
            row["kept"] = False
            row["exclusion_reason"] = rejected[str(row["clip_id"])]
        output.append(row)
    return tuple(sorted(output, key=lambda row: str(row["clip_id"])))
