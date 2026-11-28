"""P4 train/validation descriptors, B0 reference, balanced weights and gates."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from dronedet.config import FrontEndConfig
from dronedet.features import load_wave

from .artifacts import CONTRACT_PATH, canonical_json_bytes, contract_columns, write_csv
from .descriptors import apply_decile_edges, fit_decile_edges, v51_descriptor
from .freeze import CONTRACT_SHA256
from .identity import source_file_id as build_source_file_id, standardized_sample_sha256
from .weights import fit_balanced_weights, fit_structural_reference

_DESCRIPTOR_FIELDS = (
    "v51_rms_dbfs_nonzero", "v51_rolloff99_hz", "v51_noise_floor_db",
    "v51_tilt_db_per_band", "band_100_4000_dbfs", "all_zero_descriptor", "rms_missing",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rederive_train_val_descriptors(
    train_manifest_rows: Iterable[dict[str, Any]],
    val_manifest_rows: Iterable[dict[str, Any]],
    source_inventory_rows: Iterable[dict[str, Any]],
    *,
    sealed_source_file_ids: Iterable[str],
    train_manifest_sha256: str,
    val_manifest_sha256: str,
    config: FrontEndConfig,
    path_resolver: Callable[[dict[str, Any]], str | Path],
    loader: Callable[[str | Path, FrontEndConfig], np.ndarray] = load_wave,
) -> tuple[dict[str, Any], ...]:
    """Re-derive only train/validation samples and verify frozen sample hashes."""
    manifests = [dict(row) for row in train_manifest_rows] + [dict(row) for row in val_manifest_rows]
    if not manifests:
        raise ValueError("P4 requires frozen train/validation members")
    if any(row.get("split") not in {"train", "val"} for row in manifests):
        raise PermissionError("P4 waveform access is limited to frozen train and validation rows")
    if any(row.get("kept") is not True for row in manifests):
        raise ValueError("P4 re-derivation accepts only retained frozen rows")
    if len({str(row["clip_id"]) for row in manifests}) != len(manifests):
        raise ValueError("P4 train/validation membership contains duplicate clip IDs")
    for identity_field in ("group_key", "sample_sha256"):
        split_map: dict[str, set[str]] = {}
        for row in manifests:
            split_map.setdefault(str(row[identity_field]), set()).add(str(row["split"]))
        crossing = sorted(key for key, splits in split_map.items() if len(splits) > 1)
        if crossing:
            raise ValueError(f"P4 {identity_field} crosses train/validation: {crossing[:5]}")
    inventories = [dict(row) for row in source_inventory_rows]
    inventory = {str(row["source_file_id"]): row for row in inventories}
    if len(inventory) != len(inventories):
        raise ValueError("P4 inventory contains duplicate source_file_id values")
    by_file: dict[str, list[dict[str, Any]]] = {}
    for row in manifests:
        file_id = str(row["source_file_id"])
        source = inventory.get(file_id)
        if source is None:
            raise ValueError(f"P4 manifest source file is absent from inventory: {row['clip_id']}")
        for field in ("source", "source_revision", "file_sha256"):
            if row[field] != source[field]:
                raise ValueError(f"P4 manifest/inventory mismatch at {row['clip_id']} field {field}")
        if build_source_file_id(str(source["source_revision"]), str(source["relative_path"])) != file_id:
            raise ValueError(f"P4 source_file_id disagrees with inventory path: {source['relative_path']}")
        by_file.setdefault(file_id, []).append(row)
    sealed_ids = {str(file_id) for file_id in sealed_source_file_ids}
    overlap = set(by_file) & sealed_ids
    if overlap:
        raise PermissionError(
            "P4 refuses to open any source file represented in sealed test/OOD membership: "
            f"{sorted(overlap)[:5]}"
        )

    manifest_hashes = {"train": train_manifest_sha256, "val": val_manifest_sha256}
    for split, digest in manifest_hashes.items():
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"P4 {split} manifest hash must be lowercase SHA-256")
    output: list[dict[str, Any]] = []
    for file_id in sorted(by_file):
        members = by_file[file_id]
        source = inventory[file_id]
        path = Path(path_resolver(source))
        if _sha256_file(path) != str(source["file_sha256"]):
            raise ValueError(f"P4 registered source file bytes changed: {file_id}")
        audio = np.asarray(loader(path, config), dtype=np.float32)
        if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
            raise ValueError(f"P4 loader returned invalid standardized audio: {file_id}")
        for row in sorted(members, key=lambda item: str(item["clip_id"])):
            start = int(row["start_sample_16k"])
            length = int(row["n_samples"])
            if row["sample_rate"] != config.sr or start < 0 or length <= 0 or start + length > audio.size:
                raise ValueError(f"P4 frozen waveform coordinates are invalid: {row['clip_id']}")
            samples = np.ascontiguousarray(audio[start:start + length], dtype=np.float32)
            observed = standardized_sample_sha256(samples)
            if observed != row["sample_sha256"]:
                raise ValueError(f"P4 R-LED sample identity mismatch: {row['clip_id']}")
            output.append({
                "clip_id": str(row["clip_id"]),
                "manifest_sha256": manifest_hashes[str(row["split"])],
                "split": str(row["split"]),
                **v51_descriptor(samples, config),
            })
            del samples
        del audio
    if len(output) != len(manifests):
        raise AssertionError("P4 descriptor count differs from frozen train/validation membership")
    return tuple(sorted(output, key=lambda row: str(row["clip_id"])))


def _merge_manifest_descriptors(
    manifest_rows: Iterable[dict[str, Any]],
    descriptor_rows: Iterable[dict[str, Any]],
    *,
    split: str,
    manifest_sha256: str,
) -> list[dict[str, Any]]:
    manifests = [dict(row) for row in manifest_rows]
    descriptors = [dict(row) for row in descriptor_rows]
    if any(row.get("split") != split or row.get("kept") is not True for row in manifests):
        raise ValueError(f"P4 accepts only retained {split} manifest rows")
    if any(row.get("split") != split for row in descriptors):
        raise ValueError(f"P4 descriptor rows contain data outside {split}")
    manifest_ids = [str(row["clip_id"]) for row in manifests]
    descriptor_ids = [str(row["clip_id"]) for row in descriptors]
    if len(manifest_ids) != len(set(manifest_ids)) or len(descriptor_ids) != len(set(descriptor_ids)):
        raise ValueError(f"duplicate clip_id in P4 {split} inputs")
    if set(manifest_ids) != set(descriptor_ids):
        raise ValueError(f"P4 {split} descriptor coverage differs from frozen membership")
    if len(manifest_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in manifest_sha256):
        raise ValueError(f"P4 {split} manifest hash must be lowercase SHA-256")
    descriptor_map = {str(row["clip_id"]): row for row in descriptors}
    output = []
    for manifest in manifests:
        desc = descriptor_map[str(manifest["clip_id"])]
        if desc.get("manifest_sha256") not in (None, manifest_sha256):
            raise ValueError(f"P4 {split} descriptor is bound to a different manifest: {manifest['clip_id']}")
        missing = set(_DESCRIPTOR_FIELDS) - set(desc)
        if missing:
            raise ValueError(f"P4 descriptor missing named fields at {manifest['clip_id']}: {sorted(missing)}")
        output.append({**manifest, **{key: desc[key] for key in _DESCRIPTOR_FIELDS}, "split": split})
    return sorted(output, key=lambda row: str(row["clip_id"]))


def _weak_positive_flags(rows: list[dict[str, Any]]) -> None:
    train = [row for row in rows if row["split"] == "train"]
    background_references: dict[str, list[float]] = {"dd": [], "itu": []}
    for row in train:
        source_class = str(row.get("substratum", "")).strip().casefold()
        if row["label"] == 0 and row["source"] in background_references and source_class == "background":
            background_references[str(row["source"])].append(float(row["band_100_4000_dbfs"]))

    mic_recording_values: dict[str, list[float]] = {}
    for row in train:
        if row["source"] == "mic" and row["label"] == 1:
            mic_recording_values.setdefault(str(row["recording_id"]), []).append(float(row["band_100_4000_dbfs"]))
    mic_thresholds = {
        recording: float(np.quantile(values, 0.10, method="linear"))
        for recording, values in mic_recording_values.items()
    }

    for row in rows:
        if int(row["label"]) == 0:
            row["weak_positive"] = False
            row["weak_positive_reason"] = None
            continue
        source = str(row["source"])
        band = float(row["band_100_4000_dbfs"])
        if source == "ae":
            row["weak_positive"] = None
            row["weak_positive_reason"] = "source_rule_unknown"
        elif source in background_references:
            reference_values = background_references[source]
            if not reference_values:
                row["weak_positive"] = None
                row["weak_positive_reason"] = "no_train_background_reference"
            else:
                reference = float(np.median(np.asarray(reference_values, dtype=np.float64)))
                row["weak_positive"] = bool(abs(band - reference) <= 3.0)
                row["weak_positive_reason"] = None
        elif source == "mic":
            recording = str(row["recording_id"])
            if recording not in mic_thresholds:
                row["weak_positive"] = None
                row["weak_positive_reason"] = "no_train_recording_reference"
            else:
                row["weak_positive"] = bool(abs(band - mic_thresholds[recording]) <= 3.0)
                row["weak_positive_reason"] = None
        else:
            raise ValueError(f"no approved weak-positive rule for pooled positive source {source}")


def _source_polarity(admitted_sources: Iterable[str]) -> dict[str, str]:
    """Use admitted-source label design, never a possibly incomplete split."""
    classes = {
        "ae": "mixed", "dd": "mixed", "itu": "mixed",
        "mic": "positive_only", "wang": "positive_only",
        "musan": "negative_only", "cochl": "negative_only", "loc": "negative_only",
    }
    sources = {str(source) for source in admitted_sources}
    unknown = sources - set(classes)
    if unknown:
        raise ValueError(f"unknown admitted source classes: {sorted(unknown)}")
    return {source: classes[source] for source in sorted(sources)}


def _negative_shares(contract: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, float]:
    sampling = contract["sampling"]
    if any(row["source"] == "cochl" for row in rows):
        return {key: float(value) for key, value in sampling["negative_strata_with_cochl"].items()}
    return {key: float(value) for key, value in sampling["negative_strata_default_no_cochl"].items()}


def _b5_gaps(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("v51_rms_dbfs_nonzero", "v51_rolloff99_hz", "v51_noise_floor_db")
    thresholds = {
        "v51_rms_dbfs_nonzero": 6.0,
        "v51_rolloff99_hz": 1_000.0,
        "v51_noise_floor_db": 6.0,
    }
    result = {}
    for source in sorted({str(row["source"]) for row in train_rows}):
        source_rows = [row for row in train_rows if row["source"] == source]
        labels = {int(row["label"]) for row in source_rows}
        metrics_result = {}
        for metric in metrics:
            if labels != {0, 1}:
                metrics_result[metric] = {"gap": None, "threshold": thresholds[metric], "exceeds": None}
                continue
            positive = np.asarray([float(row[metric]) for row in source_rows if row["label"] == 1], dtype=np.float64)
            negative = np.asarray([float(row[metric]) for row in source_rows if row["label"] == 0], dtype=np.float64)
            gap = abs(float(np.median(positive)) - float(np.median(negative)))
            metrics_result[metric] = {
                "gap": gap,
                "threshold": thresholds[metric],
                "exceeds": bool(gap > thresholds[metric]),
            }
        result[source] = metrics_result
    return {"scope": "unweighted_frozen_train_only", "sources": result}


def fit_p4(
    *,
    train_manifest_rows: Iterable[dict[str, Any]],
    val_manifest_rows: Iterable[dict[str, Any]],
    train_descriptor_rows: Iterable[dict[str, Any]],
    val_descriptor_rows: Iterable[dict[str, Any]],
    train_manifest_sha256: str,
    val_manifest_sha256: str,
    frozen_membership_sha256: str,
    admitted_sources: Iterable[str],
    r_led_verified: bool,
    contract: dict[str, Any],
    mic_pool_approved: bool = False,
    mic_distance_results: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fit P4 outputs from only the frozen base train/validation populations."""
    if len(frozen_membership_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in frozen_membership_sha256):
        raise ValueError("P4 must bind the frozen membership SHA-256")
    if hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() != CONTRACT_SHA256:
        raise ValueError("P4 contract object differs from the approved frozen contract")
    if canonical_json_bytes(contract) != canonical_json_bytes(json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))):
        raise ValueError("P4 contract object differs from the approved frozen contract")
    train = _merge_manifest_descriptors(
        train_manifest_rows, train_descriptor_rows, split="train", manifest_sha256=train_manifest_sha256,
    )
    val = _merge_manifest_descriptors(
        val_manifest_rows, val_descriptor_rows, split="val", manifest_sha256=val_manifest_sha256,
    )
    if any(row["source"] in {"wang", "musan", "loc"} and row["label"] == 1 for row in train + val):
        raise ValueError("P4 input violates the approved pooled-source labels")
    rows = train + val
    _weak_positive_flags(rows)
    edge_record = fit_decile_edges(train, train_manifest_sha256=train_manifest_sha256)
    rows = list(apply_decile_edges(rows, edge_record))
    row_by_id = {str(row["clip_id"]): row for row in rows}
    train = [row_by_id[str(row["clip_id"])] for row in train]
    val = [row_by_id[str(row["clip_id"])] for row in val]

    train_sources = {str(row["source"]) for row in train}
    train_val_sources = train_sources | {str(row["source"]) for row in val}
    admitted_sources = {str(source) for source in admitted_sources}
    if not train_val_sources.issubset(admitted_sources):
        raise ValueError("P4 manifests contain a source outside the admitted source list")
    if "mic" in train_val_sources and not mic_pool_approved:
        raise ValueError("MIC train/validation rows require a recorded pre-P1 O5 pool approval")
    mic_ids = {str(row["clip_id"]) for row in train + val if row["source"] == "mic"}
    if mic_pool_approved and mic_ids:
        if mic_distance_results is None or set(mic_distance_results) != mic_ids:
            raise ValueError("pooled MIC rows require complete precomputed distance outcomes")
        for clip_id, result in mic_distance_results.items():
            if set(result) != {"dist_value", "dist_bin"} or result["dist_bin"] not in {"near", "mid", "far", "unknown"}:
                raise ValueError(f"invalid MIC distance outcome for {clip_id}")
            value = result["dist_value"]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            ):
                raise ValueError(f"non-finite MIC distance for {clip_id}")
    positive_shares = contract["sampling"][
        "positive_shares_MIC_pool" if mic_pool_approved else "positive_shares_default"
    ]
    positive_shares = {key: float(value) for key, value in positive_shares.items()}
    negative_shares = _negative_shares(contract, train)
    polarity = _source_polarity(admitted_sources)
    positive_reference = fit_structural_reference(
        train, label=1, margin_shares=positive_shares,
        source_polarity=polarity, contract=contract,
    )
    negative_reference = fit_structural_reference(
        train, label=0, margin_shares=negative_shares,
        source_polarity=polarity, contract=contract,
    )
    balanced = fit_balanced_weights(
        train,
        positive_source_shares=positive_shares,
        negative_stratum_shares=negative_shares,
        source_polarity=polarity,
        positive_reference=positive_reference,
        negative_reference=negative_reference,
        contract=contract,
    )
    train_weights = {clip: float(weight) for clip, weight in zip(balanced.clip_ids, balanced.weights)}
    descriptors = []
    for row in sorted(rows, key=lambda value: str(value["clip_id"])):
        descriptors.append({
            "clip_id": str(row["clip_id"]),
            "manifest_sha256": train_manifest_sha256 if row["split"] == "train" else val_manifest_sha256,
            "split": str(row["split"]),
            **{field: row[field] for field in _DESCRIPTOR_FIELDS},
            "weak_positive": row["weak_positive"],
            "weak_positive_reason": row["weak_positive_reason"],
            "dist_value": mic_distance_results[str(row["clip_id"])]["dist_value"]
            if row["source"] == "mic" and mic_pool_approved else None,
            "dist_bin": mic_distance_results[str(row["clip_id"])]["dist_bin"]
            if row["source"] == "mic" and mic_pool_approved else "unknown",
            "rms_decile": int(row["rms_decile"]),
            "rolloff_decile": int(row["rolloff_decile"]),
            "w_train": train_weights[str(row["clip_id"])] if row["split"] == "train" else None,
        })

    b1 = bool(balanced.solver_record["b1_descriptor_balance_pass"])
    b2 = bool(balanced.solver_record["b2_pass"])
    gap_report = {
        "frozen_membership_sha256": frozen_membership_sha256,
        "train_manifest_sha256": train_manifest_sha256,
        "contract_sha256": CONTRACT_SHA256,
        **_b5_gaps(train),
    }
    reference_record = {
        "study_id": "study-v5-1",
        "frozen_membership_sha256": frozen_membership_sha256,
        "train_manifest_sha256": train_manifest_sha256,
        "contract_sha256": CONTRACT_SHA256,
        "positive": {
            "clip_ids": list(positive_reference.clip_ids),
            "q_ref": [float(value) for value in positive_reference.probabilities],
            "clip_cap": positive_reference.clip_cap,
            "effective_size": positive_reference.effective_size,
            "solver": positive_reference.solver_record,
        },
        "negative": {
            "clip_ids": list(negative_reference.clip_ids),
            "q_ref": [float(value) for value in negative_reference.probabilities],
            "clip_cap": negative_reference.clip_cap,
            "effective_size": negative_reference.effective_size,
            "solver": negative_reference.solver_record,
        },
    }
    fit_record = {
        "study_id": "study-v5-1",
        "frozen_membership_sha256": frozen_membership_sha256,
        "train_manifest_sha256": train_manifest_sha256,
        "val_manifest_sha256": val_manifest_sha256,
        "contract_sha256": CONTRACT_SHA256,
        "clip_ids": list(balanced.clip_ids),
        "w_train": [float(value) for value in balanced.weights],
        "solver": balanced.solver_record,
    }
    gates = {
        "R-LED": "PASS" if r_led_verified else "FAIL",
        "B0": "PASS",
        "B1": "PASS" if b1 else "FAIL",
        "B2": "PASS" if b2 else "FAIL",
        "B3_B4": "PASS",
        "B5": "PASS",
    }
    return {
        "descriptors": tuple(descriptors),
        "edge_record": edge_record,
        "reference_record": reference_record,
        "fit_record": fit_record,
        "gap_report": gap_report,
        "gates": gates,
        "summary": {
            "study_id": "study-v5-1",
            "frozen_membership_sha256": frozen_membership_sha256,
            "train_manifest_sha256": train_manifest_sha256,
            "val_manifest_sha256": val_manifest_sha256,
            "contract_sha256": CONTRACT_SHA256,
            "train_rows": len(train),
            "val_rows": len(val),
            "positive_reference_ess": positive_reference.effective_size,
            "negative_reference_ess": negative_reference.effective_size,
            "effective_sizes": balanced.effective_sizes,
            "dropped_margins": list(balanced.dropped_margins),
            "gates": gates,
        },
    }


def write_p4_artifacts(output_directory: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    """Write a unique local P4 attempt, retaining failed B1/B2 evidence."""
    target = Path(output_directory)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite P4 attempt directory: {target}")
    target.mkdir(parents=True, exist_ok=False)
    descriptor_hash = write_csv(
        target / "descriptors_v51.csv",
        contract_columns("descriptors_v51.csv"),
        result["descriptors"],
    )
    files = {"descriptors_v51.csv": descriptor_hash}
    for name, key in (
        ("descriptor_edges.json", "edge_record"),
        ("weight_reference.json", "reference_record"),
        ("weight_fit.json", "fit_record"),
        ("descriptor_gap_report.json", "gap_report"),
        ("p4_summary.json", "summary"),
    ):
        payload = canonical_json_bytes(result[key]) + b"\n"
        path = target / name
        path.write_bytes(payload)
        files[name] = hashlib.sha256(payload).hexdigest()
    return {"files_sha256": files, "gates": dict(result["gates"]), "state": "PASS" if all(value == "PASS" for value in result["gates"].values()) else "FAIL"}
