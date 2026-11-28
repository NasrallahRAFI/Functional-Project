"""One-pass V5.1 mechanical processing and P2 identity reconciliation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from dronedet.config import FrontEndConfig
from dronedet.features import load_wave

from .cleaning import clean_new_source_identities, verify_legacy_preservation
from .artifacts import canonical_json_bytes, contract_columns, write_csv
from .identity import (
    clip_id as build_clip_id,
    source_file_id as build_source_file_id,
    standardized_sample_sha256,
)
from .mechanical import mechanical_flags

_LEDGER_COLUMNS = (
    "clip_id", "legacy_clip_id", "source_file_id", "source", "source_revision",
    "file_sha256", "recording_id", "group_key", "substratum", "negative_stratum",
    "label", "label_basis", "label_weak", "start_sample_16k", "n_samples",
    "sample_rate", "sample_sha256", "exact_zero", "silent", "near_silent",
    "partial_zero", "clipped_flag", "dc_flag", "crop_all_zero", "stats_eligible",
    "candidate_role", "kept", "exclusion_reason", "duplicate_representative", "pinned",
)
_MECHANICAL_FIELDS = (
    "exact_zero", "silent", "near_silent", "partial_zero", "clipped_flag", "dc_flag",
    "crop_all_zero", "stats_eligible",
)
_LEGACY_SOURCES = {"ae", "dd"}
_REQUIRED_SOURCE_REVISIONS = {
    "ae": "al-emadi-r001",
    "dd": "thesis-r001",
    "itu": "itu-aris-r001",
    "musan": "musan-r001",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _check_cross_source_identities(rows: list[dict[str, Any]]) -> dict[str, int]:
    identities: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["kept"] is True:
            identities.setdefault(str(row["sample_sha256"]), []).append(row)
    collisions = []
    for digest, members in identities.items():
        sources = {str(row["source"]) for row in members}
        labels = {int(row["label"]) for row in members}
        if len(sources) > 1 or len(labels) > 1:
            collisions.append({"sample_sha256": digest, "sources": sorted(sources), "labels": sorted(labels)})
    if collisions:
        raise ValueError(f"cross-source/class standardized sample collision: {collisions[:10]}")
    return {"retained_unique_sample_hashes": len(identities)}


def run_p2(
    candidate_specs: Iterable[dict[str, Any]],
    source_inventory_rows: Iterable[dict[str, Any]],
    *,
    config: FrontEndConfig,
    legacy_reference_rows: Iterable[dict[str, Any]],
    loader: Callable[[str | Path, FrontEndConfig], np.ndarray] = load_wave,
) -> dict[str, Any]:
    """Build the closed mechanical ledger from approved P1 candidate specs.

    ``candidate_specs`` is an attempt-local, non-Git input that adds only
    ``audio_path`` and ``relative_path`` to the eventual ledger identity fields.
    Each registered physical file is byte-verified and decoded exactly once.
    No waveform, descriptor or path is included in the returned artifacts.
    """
    specs = [dict(row) for row in candidate_specs]
    source_rows = [dict(row) for row in source_inventory_rows]
    legacy_reference = [dict(row) for row in legacy_reference_rows]
    spec_sources = {str(row["source"]) for row in specs}
    inventory_sources = {str(row["source"]) for row in source_rows}
    missing_inventory = set(_REQUIRED_SOURCE_REVISIONS) - inventory_sources
    missing_candidates = set(_REQUIRED_SOURCE_REVISIONS) - spec_sources
    if missing_inventory or missing_candidates:
        raise ValueError(
            "P2 cannot begin until the complete default source set is present; "
            f"missing inventory={sorted(missing_inventory)}, candidates={sorted(missing_candidates)}"
        )
    wrong_revisions = {
        source: sorted({str(row["source_revision"]) for row in source_rows if row["source"] == source})
        for source, expected in _REQUIRED_SOURCE_REVISIONS.items()
        if {str(row["source_revision"]) for row in source_rows if row["source"] == source} != {expected}
    }
    if wrong_revisions:
        raise ValueError(f"P2 source revisions differ from approved registrations: {wrong_revisions}")
    ids = [str(row["clip_id"]) for row in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("P2 candidate specs contain duplicate clip_id values")

    inventory: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        file_id = str(row["source_file_id"])
        if file_id in inventory:
            raise ValueError(f"P2 inventory has duplicate source_file_id {file_id}")
        if build_source_file_id(str(row["source_revision"]), str(row["relative_path"])) != file_id:
            raise ValueError(f"P2 inventory file ID disagrees with revision/path: {row['relative_path']}")
        inventory[file_id] = row
    grouped: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        file_id = str(spec["source_file_id"])
        source = inventory.get(file_id)
        if source is None:
            raise ValueError(f"P2 candidate has no inventoried source file: {spec['clip_id']}")
        for field in ("source", "source_revision", "file_sha256"):
            if spec[field] != source[field]:
                raise ValueError(f"P2 candidate/inventory mismatch at {spec['clip_id']} field {field}")
        if spec["relative_path"] != source["relative_path"]:
            raise ValueError(f"P2 candidate path differs from inventory at {spec['clip_id']}")
        if spec["source"] in _LEGACY_SOURCES:
            if spec["legacy_clip_id"] != spec["clip_id"]:
                raise ValueError(f"AE/DD legacy clip ID mismatch: {spec['clip_id']}")
        else:
            expected_id = build_clip_id(
                str(spec["source"]), file_id,
                int(spec["start_sample_16k"]), int(spec["n_samples"]),
            )
            if str(spec["clip_id"]) != expected_id or spec["legacy_clip_id"] is not None:
                raise ValueError(f"new-source candidate ID mismatch: {spec['clip_id']}")
        grouped.setdefault(file_id, []).append(spec)

    ledger: list[dict[str, Any]] = []
    for file_id in sorted(grouped):
        members = grouped[file_id]
        expected_file_hash = str(inventory[file_id]["file_sha256"])
        paths = {str(spec["audio_path"]) for spec in members}
        relative_paths = {str(spec["relative_path"]) for spec in members}
        if len(paths) != 1 or len(relative_paths) != 1:
            raise ValueError(f"one source_file_id maps to multiple P2 paths: {file_id}")
        path = Path(next(iter(paths)))
        if _sha256_file(path) != expected_file_hash:
            raise ValueError(f"registered source bytes changed before P2: {file_id}")
        audio = np.asarray(loader(path, config), dtype=np.float32)
        if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
            raise ValueError(f"P2 loader returned invalid standardized audio for {file_id}")

        for spec in sorted(members, key=lambda row: str(row["clip_id"])):
            start = int(spec["start_sample_16k"])
            count = int(spec["n_samples"])
            if start < 0 or count <= 0 or start + count > audio.size:
                raise ValueError(f"P2 candidate window is outside standardized file: {spec['clip_id']}")
            samples = np.ascontiguousarray(audio[start:start + count], dtype=np.float32)
            flags = mechanical_flags(samples, config)
            generated_fields = set(_MECHANICAL_FIELDS) | {"sample_sha256", "sample_rate"}
            row = {key: spec[key] for key in _LEDGER_COLUMNS if key not in generated_fields}
            row.update({key: flags[key] for key in _MECHANICAL_FIELDS})
            row["sample_rate"] = int(config.sr)
            row["sample_sha256"] = standardized_sample_sha256(samples)
            row["_relative_path"] = str(spec["relative_path"])
            if flags["too_short"]:
                row["kept"] = False
                row["exclusion_reason"] = "too_short"
            if row["source"] not in _LEGACY_SOURCES and count != 16_000:
                raise ValueError(f"new-source P2 candidate is not exactly one second: {row['clip_id']}")
            if row["source"] in _LEGACY_SOURCES and row["legacy_clip_id"] != row["clip_id"]:
                raise ValueError(f"AE/DD legacy clip ID mismatch: {row['clip_id']}")
            if row["source"] not in _LEGACY_SOURCES and row["legacy_clip_id"] is not None:
                raise ValueError(f"new-source candidate has legacy clip ID: {row['clip_id']}")
            ledger.append(row)
            del samples, flags, row
        del audio

    legacy = [row for row in ledger if row["source"] in _LEGACY_SOURCES]
    if not legacy:
        raise ValueError("P2 produced no AE/DD legacy rows")
    legacy_result = verify_legacy_preservation(legacy, legacy_reference)
    new_source = [row for row in ledger if row["source"] not in _LEGACY_SOURCES]
    cleaned_new = clean_new_source_identities(new_source)
    combined = sorted([*legacy, *cleaned_new], key=lambda row: str(row["clip_id"]))
    identity_summary = _check_cross_source_identities(combined)
    if len({str(row["clip_id"]) for row in combined}) != len(combined):
        raise ValueError("P2 ledger contains duplicate clip IDs after reconciliation")
    closed = [{column: row[column] for column in _LEDGER_COLUMNS} for row in combined]
    return {
        "mechanical_ledger": tuple(closed),
        "summary": {
            "state": "PASS",
            "candidate_rows": len(closed),
            "kept_rows": sum(row["kept"] is True for row in closed),
            "source_rows": {
                source: sum(row["source"] == source for row in closed)
                for source in sorted({row["source"] for row in closed})
            },
            **identity_summary,
            "gates": {
                "P-AE/R3": "PASS" if legacy_result["state"] == "PASS" else "FAIL",
                "P2_SAMPLE_IDENTITY": "PASS",
            },
        },
    }


def write_p2_ledger(output_directory: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    """Persist one completed P2 ledger under a fresh attempt-local directory."""
    if set(result) != {"mechanical_ledger", "summary"}:
        raise ValueError("P2 result has an unexpected structure")
    target = Path(output_directory)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite P2 attempt directory: {target}")
    rows = list(result["mechanical_ledger"])
    target.mkdir(parents=True, exist_ok=False)
    ledger_hash = write_csv(
        target / "mechanical_ledger.csv",
        contract_columns("mechanical_ledger.csv"),
        rows,
    )
    summary = {
        "state": "PASS",
        "mechanical_ledger_sha256": ledger_hash,
        **result["summary"],
    }
    summary["gates"] = {**summary.get("gates", {}), "M-P": "PASS"}
    (target / "p2_summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    return summary
