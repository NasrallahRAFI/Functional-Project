"""P3 split, cap, gate, reservoir and S9 orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .freeze import freeze_split_attempt
from .identity import clip_id as build_clip_id, safe_legacy_filename, source_file_id as build_source_file_id
from .splitting import (
    SplitAssignment,
    apply_cochl_caps,
    assign_splits,
    check_split_gates,
    select_train_reservoir,
)

_BASE_ROLES = {"base", "base_and_reservoir"}


def _validate_source_references(
    candidate_rows: list[dict[str, Any]],
    source_file_rows: list[dict[str, Any]],
    bindings: dict[str, Any],
) -> None:
    if bindings.get("study_id") != "study-v5-1":
        raise ValueError("P3 freeze bindings must identify study-v5-1")
    known_sources = {str(row["source"]) for row in source_file_rows}
    ledger_sources = {str(row["source"]) for row in candidate_rows}
    bound_sources = set(bindings["source_list"])
    observed_sources = known_sources | ledger_sources
    if not observed_sources.issubset(bound_sources):
        raise ValueError("P3 candidates include a source absent from the frozen source list")
    inventory: dict[str, dict[str, Any]] = {}
    for row in source_file_rows:
        file_id = str(row["source_file_id"])
        if file_id in inventory:
            raise ValueError(f"duplicate source_file_id in P3 inventory: {file_id}")
        if build_source_file_id(str(row["source_revision"]), str(row["relative_path"])) != file_id:
            raise ValueError(f"source_file_id disagrees with revision/path: {row['relative_path']}")
        inventory[file_id] = row

    for row in candidate_rows:
        file_id = str(row["source_file_id"])
        source = inventory.get(file_id)
        if source is None:
            raise ValueError(f"candidate source_file_id is absent from inventory: {row['clip_id']}")
        for field in ("source", "source_revision", "file_sha256"):
            if row[field] != source[field]:
                raise ValueError(f"candidate/inventory mismatch at {row['clip_id']} field {field}")
        if row["source"] in {"ae", "dd"}:
            if row["legacy_clip_id"] != row["clip_id"]:
                raise ValueError(f"AE/DD legacy clip ID was not preserved: {row['clip_id']}")
        else:
            if row["legacy_clip_id"] is not None:
                raise ValueError(f"new source has a non-null legacy_clip_id: {row['clip_id']}")
            expected_id = build_clip_id(
                str(row["source"]), file_id,
                int(row["start_sample_16k"]), int(row["n_samples"]),
            )
            if str(row["clip_id"]) != expected_id:
                raise ValueError(f"new clip_id disagrees with file/start/length identity: {row['clip_id']}")


def run_p3(
    output_directory: str | Path,
    *,
    candidate_rows: Iterable[dict[str, Any]],
    source_file_rows: Iterable[dict[str, Any]],
    quarantine_rows: Iterable[dict[str, Any]],
    bindings: dict[str, Any],
) -> dict[str, Any]:
    """Build and freeze one complete P3 attempt after P2 has passed.

    Inputs are in-memory closed-schema P2 artifacts. Any blocking S1–S8 gate
    fails before the output attempt directory is created.
    """
    candidate_rows = [dict(row) for row in candidate_rows]
    source_file_rows = [dict(row) for row in source_file_rows]
    quarantine_rows = [dict(row) for row in quarantine_rows]
    _validate_source_references(candidate_rows, source_file_rows, bindings)
    assignment = assign_splits(candidate_rows, source_file_rows)
    rows = [dict(row) for row in assignment.rows]
    for row in rows:
        row["derived_relpath"] = f"{row['source']}/{safe_legacy_filename(str(row['clip_id']))}.wav"
    assignment = replace(assignment, rows=tuple(rows))

    cochl_indices = [index for index, row in enumerate(rows) if row.get("source") == "cochl"]
    if cochl_indices:
        capped = apply_cochl_caps(rows[index] for index in cochl_indices)
        capped_by_id = {str(row["clip_id"]): row for row in capped}
        rows = [capped_by_id.get(str(row["clip_id"]), row) for row in rows]
        assignment = replace(assignment, rows=tuple(rows))

    reservoir = select_train_reservoir(assignment)
    gates = check_split_gates(assignment, reservoir)
    failures = {name: value for name, value in gates.items() if value["state"] != "PASS"}
    if failures:
        raise ValueError(f"P3 blocking gate failure; no freeze written: {failures}")

    memberships = {
        split: [
            row for row in rows
            if row.get("kept") is True
            and row.get("candidate_role") in _BASE_ROLES
            and row.get("split") == split
        ]
        for split in ("train", "val", "test", "ood")
    }
    frozen = freeze_split_attempt(
        output_directory,
        datasets={
            **memberships,
            "reservoir_train": reservoir,
            "quarantine": quarantine_rows,
            "ledger": rows,
        },
        bindings=bindings,
    )
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        if row.get("kept") is True and row.get("candidate_role") in _BASE_ROLES and row.get("split") is not None:
            source, split = str(row["source"]), str(row["split"])
            counts.setdefault(source, {}).setdefault(split, 0)
            counts[source][split] += 1
    return {
        "membership_sha256": frozen["membership_sha256"],
        "frozen_json_sha256": frozen["frozen_json_sha256"],
        "input_order_sha256": assignment.input_order_sha256,
        "sklearn_version": assignment.sklearn_version,
        "gates": gates,
        "counts": counts,
        "artifact_sha256": frozen["artifact_sha256"],
    }
