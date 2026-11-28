"""V5.1 cross-source sample-identity cleaning and legacy-map checks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .randomness import keyed_sha1

_LEGACY_SOURCES = {"ae", "dd"}
_BASE_ROLES = {"base", "base_and_reservoir"}


def verify_legacy_preservation(
    actual_rows: Iterable[dict[str, Any]],
    reference_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Require exact V4 AE/DD clip/group/representative preservation."""
    actual = [dict(row) for row in actual_rows]
    reference = [dict(row) for row in reference_rows]
    actual_ids = [str(row["clip_id"]) for row in actual]
    reference_ids = [str(row["clip_id"]) for row in reference]
    if len(actual_ids) != len(set(actual_ids)) or len(reference_ids) != len(set(reference_ids)):
        raise ValueError("legacy preservation comparison contains duplicate clip IDs")
    fields = ("source", "group_key", "duplicate_representative", "kept", "exclusion_reason", "pinned")
    actual_map = {str(row["clip_id"]): row for row in actual}
    reference_map = {str(row["clip_id"]): row for row in reference}
    if set(actual_map) != set(reference_map):
        raise ValueError(
            "legacy candidate IDs differ from the V4 reference: "
            f"missing={sorted(set(reference_map)-set(actual_map))[:5]}, "
            f"extra={sorted(set(actual_map)-set(reference_map))[:5]}"
        )
    mismatches = []
    for clip_id in sorted(actual_map):
        for field in fields:
            if actual_map[clip_id].get(field) != reference_map[clip_id].get(field):
                mismatches.append({"clip_id": clip_id, "field": field})
    if mismatches:
        raise ValueError(f"legacy V4 map changed: {mismatches[:10]}")
    counts = {
        source: sum(row.get("source") == source and row.get("kept") is True for row in actual)
        for source in sorted(_LEGACY_SOURCES)
    }
    if counts != {"ae": 10_623, "dd": 890}:
        raise ValueError(f"legacy retained counts differ from the approved V4 contract: {counts}")
    return {"state": "PASS", "candidate_rows": len(actual), "kept_by_source": counts}


def clean_new_source_identities(
    candidate_rows: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Drop new-source exact zeros, stop on cross-source/class hashes, and log aliases.

    AE/DD rows are rejected here so their exact V4 representatives cannot be
    silently replaced by a generic deduplication rule.
    """
    rows = [dict(row) for row in candidate_rows]
    if any(row.get("source") in _LEGACY_SOURCES for row in rows):
        raise ValueError("AE/DD must use verify_legacy_preservation, not generic new-source cleaning")
    ids = [str(row["clip_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate clip_id in P2 new-source candidates")

    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    output = []
    for row in rows:
        if row.get("kept") is not True:
            output.append(row)
            continue
        if row.get("exact_zero") is True:
            row["kept"] = False
            row["exclusion_reason"] = "new_source_exact_zero"
            output.append(row)
            continue
        if not row.get("sample_sha256") or not row.get("source") or row.get("label") not in (0, 1):
            raise ValueError(f"missing identity/label on candidate {row.get('clip_id')}")
        eligible[str(row["sample_sha256"])].append(row)

    for sample_hash, aliases in eligible.items():
        sources = {str(row["source"]) for row in aliases}
        classes = {
            (str(row["source"]), int(row["label"]), str(row["label_basis"]), str(row["substratum"]))
            for row in aliases
        }
        if len(sources) > 1 or len(classes) > 1:
            raise ValueError(
                f"cross-source/class sample identity collision {sample_hash}: "
                f"sources={sorted(sources)}, classes={sorted(classes)}"
            )
        if len(aliases) == 1:
            continue
        ranked = sorted(aliases, key=_representative_rank)
        representative = str(ranked[0]["clip_id"])
        for alias in ranked[1:]:
            alias["kept"] = False
            alias["exclusion_reason"] = "exact_sample_duplicate"
            alias["duplicate_representative"] = representative
        ranked[0]["duplicate_representative"] = representative

    output.extend(row for aliases in eligible.values() for row in aliases)
    return tuple(sorted(output, key=lambda row: str(row["clip_id"])))


def _representative_rank(row: dict[str, Any]) -> tuple[str, str, str]:
    role = str(row["candidate_role"])
    if role not in _BASE_ROLES | {"reservoir_candidate"}:
        raise ValueError(f"invalid P2 candidate role: {role}")
    priority = "0" if role in _BASE_ROLES else "1"
    relative_path = row.get("relative_path", row.get("_relative_path"))
    if relative_path is None:
        raise ValueError(f"candidate lacks its transient source-relative path: {row['clip_id']}")
    digest = keyed_sha1("file_order", [str(row["source_revision"]), str(relative_path)])
    return priority, digest, str(row["clip_id"])
