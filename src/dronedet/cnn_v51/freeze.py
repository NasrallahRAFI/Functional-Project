"""Closed-schema S9 split serialization and immutable membership binding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .artifacts import canonical_json_bytes, contract_columns, write_csv

PLAN_SHA256 = "6a27c99f960e4e0520709216cf204e5890eca6e30ebbb438d7934f300af9f5a1"
CONTRACT_SHA256 = "6911c92b34696f351e4033b21af9ff74f0e6c49627b4987ebae87431d8fe0de4"
_FILES = {
    "train": ("train.csv", "train.csv"),
    "val": ("val.csv", "val.csv"),
    "test": ("test.csv", "test.csv"),
    "ood": ("ood.csv", "ood.csv"),
    "reservoir_train": ("reservoir_train.csv", "reservoir_train.csv"),
    "quarantine": ("quarantine_manifest.csv", "quarantine_manifest.csv"),
    "ledger": ("mechanical_ledger.csv", "mechanical_ledger.csv"),
}
_BINDING_FIELDS = {
    "study_id", "source_inventory_sha256", "plan_sha256", "contract_sha256",
    "dependencies", "seed", "group_id_grid_rules", "source_list", "L",
    "transform_hash", "b2_reference_floor_policy",
}


def _rows(value: Iterable[dict[str, Any]], artifact: str) -> list[dict[str, Any]]:
    columns = contract_columns(artifact)
    materialized = []
    for index, row in enumerate(value):
        missing = set(columns) - set(row)
        if missing:
            raise ValueError(f"{artifact} row {index} missing columns: {sorted(missing)}")
        # The contract is closed; explicitly project away transient implementation fields.
        materialized.append({key: row[key] for key in columns})
    return materialized


def _validate_membership(datasets: dict[str, list[dict[str, Any]]]) -> None:
    membership = [row for name in ("train", "val", "test", "ood") for row in datasets[name]]
    ids = [str(row["clip_id"]) for row in membership]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate clip_id across frozen memberships")
    for split in ("train", "val", "test", "ood"):
        for row in datasets[split]:
            if row["split"] != split or row["kept"] is not True:
                raise ValueError(f"{split}.csv contains a row with wrong split or not-kept state")
    for split in ("train", "val", "test"):
        for row in datasets[split]:
            if row["source"] in {"mic", "wang"}:
                raise ValueError(f"OOD source leaked into {split}: {row['clip_id']}")
    if any(row["source"] not in {"mic", "wang"} for row in datasets["ood"]):
        raise ValueError("ood.csv contains a source not designated OOD by the approved defaults")

    groups: dict[str, set[str]] = {}
    samples: dict[str, set[str]] = {}
    for row in membership:
        groups.setdefault(str(row["group_key"]), set()).add(str(row["split"]))
        samples.setdefault(str(row["sample_sha256"]), set()).add(str(row["split"]))
    crossing_groups = sorted(key for key, values in groups.items() if len(values) != 1)
    crossing_samples = sorted(key for key, values in samples.items() if len(values) != 1)
    if crossing_groups or crossing_samples:
        raise ValueError(
            "frozen group/sample identity crosses splits: "
            f"groups={crossing_groups[:5]}, sample_hashes={crossing_samples[:5]}"
        )

    pooled = [row for row in membership if row["source"] in {"ae", "dd", "itu"}]
    for split in ("train", "val", "test"):
        labels = {int(row["label"]) for row in pooled if row["split"] == split}
        if labels != {0, 1}:
            raise ValueError(f"S3 requires both labels in pooled {split}")
    for split in ("val", "test"):
        positives = {
            str(row["group_key"]) for row in pooled
            if row["split"] == split and int(row["label"]) == 1
        }
        if len(positives) < 25:
            raise ValueError(f"S4 requires at least 25 positive groups in {split}; found {len(positives)}")
    pooled_groups: dict[str, set[str]] = {}
    for row in pooled:
        pooled_groups.setdefault(str(row["source"]), set()).add(str(row["group_key"]))
    for source, source_groups in pooled_groups.items():
        if len(source_groups) >= 6:
            for split in ("val", "test"):
                if not any(row["source"] == source and row["split"] == split for row in pooled):
                    raise ValueError(f"S5 requires {source} representation in {split}")
    moved_pins = [row["clip_id"] for row in membership if row["pinned"] is True and row["split"] != "train"]
    if moved_pins:
        raise ValueError(f"S6 pinned groups are not train-only: {moved_pins[:5]}")

    base_hashes = {str(row["sample_sha256"]) for row in membership}
    if len(datasets["reservoir_train"]) > 200_000:
        raise ValueError("train reservoir exceeds the fixed 200,000-row cap")
    reservoir_ids: set[str] = set()
    for row in datasets["reservoir_train"]:
        if row["split"] != "train" or row["candidate_role"] != "reservoir" or row["source"] != "musan":
            raise ValueError(f"invalid reservoir row: {row['clip_id']}")
        sample_hash = str(row["sample_sha256"])
        if sample_hash in base_hashes or sample_hash in reservoir_ids:
            raise ValueError(f"reservoir duplicates a base or prior reservoir identity: {row['clip_id']}")
        reservoir_ids.add(sample_hash)

    ledger_by_id = {str(row["clip_id"]): row for row in datasets["ledger"]}
    if len(ledger_by_id) != len(datasets["ledger"]):
        raise ValueError("mechanical ledger contains duplicate clip_id values")
    for row in membership:
        source_row = ledger_by_id.get(str(row["clip_id"]))
        if source_row is None:
            raise ValueError(f"frozen membership row is missing from mechanical ledger: {row['clip_id']}")
        for field in ("source_file_id", "sample_sha256", "group_key", "label", "candidate_role", "kept"):
            if source_row[field] != row[field]:
                raise ValueError(f"frozen membership differs from ledger at {row['clip_id']} field {field}")


def compare_reproductions(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Evaluate R-DET without accepting divergent frozen memberships."""
    first_hash = str(first.get("membership_sha256", ""))
    second_hash = str(second.get("membership_sha256", ""))
    if len(first_hash) != 64 or len(second_hash) != 64:
        raise ValueError("both reproduction records must contain a SHA-256 membership hash")
    equal = first_hash == second_hash
    return {
        "state": "PASS" if equal else "FAIL",
        "first_membership_sha256": first_hash,
        "second_membership_sha256": second_hash,
        "equal": equal,
    }


def _validate_bindings(bindings: dict[str, Any]) -> None:
    if set(bindings) != _BINDING_FIELDS:
        raise ValueError(
            f"freeze binding fields differ: missing={sorted(_BINDING_FIELDS-set(bindings))}, "
            f"extra={sorted(set(bindings)-_BINDING_FIELDS)}"
        )
    if bindings["plan_sha256"] != PLAN_SHA256 or bindings["contract_sha256"] != CONTRACT_SHA256:
        raise ValueError("freeze must bind the approved plan and contract hashes")
    if bindings["seed"] != 51:
        raise ValueError("freeze seed must be 51")
    if not isinstance(bindings["source_inventory_sha256"], dict) or not bindings["source_inventory_sha256"]:
        raise ValueError("freeze must bind at least one registered source inventory hash")
    for identity, digest in bindings["source_inventory_sha256"].items():
        value = str(digest)
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"invalid source inventory SHA-256 for {identity}")
    if not isinstance(bindings["dependencies"], dict) or not bindings["dependencies"]:
        raise ValueError("freeze must bind the pinned dependency identities")
    sources = bindings["source_list"]
    if not isinstance(sources, list) or not sources or sources != sorted(set(sources)):
        raise ValueError("freeze source_list must be a non-empty sorted unique list")
    if not isinstance(bindings["group_id_grid_rules"], dict) or not bindings["group_id_grid_rules"]:
        raise ValueError("freeze must bind group, ID and grid rules")
    if not isinstance(bindings["b2_reference_floor_policy"], dict) or not bindings["b2_reference_floor_policy"]:
        raise ValueError("freeze must bind the approved B2 reference/floor policy")
    if bindings["L"] != 0.5:
        raise ValueError("freeze must preserve the historical L=0.5 identity")
    transform = str(bindings["transform_hash"])
    if transform != "bb7a21a1e5fb317c43e1e214bc04af6e46e04d75053383d1c62ef6849c9466c7":
        raise ValueError("freeze transform hash differs from the approved processing identity")
    canonical_json_bytes(bindings)


def freeze_split_attempt(
    output_directory: str | Path,
    *,
    datasets: dict[str, Iterable[dict[str, Any]]],
    bindings: dict[str, Any],
) -> dict[str, Any]:
    """Write a new S9 attempt once, then bind every membership artifact by hash.

    The caller must supply the already gated and capped full ledger. Existing
    attempt directories are never overwritten. No timestamps or machine paths
    are written into the membership hash domain.
    """
    if set(datasets) != set(_FILES):
        raise ValueError(f"S9 datasets must be exactly {sorted(_FILES)}")
    _validate_bindings(bindings)
    materialized = {
        key: _rows(datasets[key], schema_name)
        for key, (filename, schema_name) in _FILES.items()
    }
    _validate_membership(materialized)

    target = Path(output_directory)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite frozen attempt directory: {target}")
    target.mkdir(parents=True, exist_ok=False)
    file_hashes: dict[str, str] = {}
    for key, (filename, schema_name) in _FILES.items():
        file_hashes[filename] = write_csv(
            target / filename,
            contract_columns(schema_name),
            materialized[key],
            sort_by="clip_id" if schema_name != "quarantine_manifest.csv" else "source_file_id",
        )

    membership_domain = {
        name: digest for name, digest in sorted(file_hashes.items())
    }
    frozen = {
        **bindings,
        "artifact_sha256": dict(sorted(file_hashes.items())),
        "membership_sha256": hashlib.sha256(canonical_json_bytes(membership_domain)).hexdigest(),
        "membership_hash_domain": membership_domain,
        "test_cache_created": False,
    }
    frozen_path = target / "FROZEN.json"
    frozen_path.write_bytes(canonical_json_bytes(frozen) + b"\n")
    frozen["frozen_json_sha256"] = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    return frozen
