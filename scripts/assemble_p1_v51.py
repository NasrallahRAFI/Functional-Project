"""Reconcile partial P1 inventories into the complete attempt-local input set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT = WORKSPACE.parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from dronedet.cnn_v51.artifacts import contract_columns, write_csv  # noqa: E402
from dronedet.cnn_v51.gates import GateState, append_gate  # noqa: E402
from dronedet.cnn_v51.identity import canonical_relative_path, source_file_id  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    attempt = PROJECT / "local/study-v5-1/attempts" / args.attempt_id
    partial = attempt / "partial-p1-ae-dd"
    new = attempt / "p1-itu-musan"
    out = attempt / "p1-complete"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty P1 output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    paths = json.loads((PROJECT / "local/study-v5-1/paths.json").read_text(encoding="utf-8"))
    legacy_inventory = pd.read_csv(partial / "inventory_v51.partial.csv", dtype=str, keep_default_na=False)
    legacy_labels = pd.read_csv(partial / "label_register.partial.csv", dtype=str, keep_default_na=False)
    new_inventory = pd.read_csv(new / "inventory_itu_musan.csv", dtype=str, keep_default_na=False)
    new_labels = pd.read_csv(new / "label_register_itu_musan.csv", dtype=str, keep_default_na=False)
    source_rows = pd.concat([legacy_inventory, new_inventory], ignore_index=True).to_dict(orient="records")
    source_by_id = {row["source_file_id"]: row for row in source_rows}
    label_rows = pd.concat([legacy_labels, new_labels], ignore_index=True).to_dict(orient="records")
    for row in source_rows:
        for field in ("bytes", "native_sr", "channels", "native_frames", "label"):
            row[field] = int(row[field])
        row["label_weak"] = row["label_weak"].lower() == "true"
        for field in ("negative_stratum", "source_metadata_sha256"):
            row[field] = row[field] or None
    for row in label_rows:
        row["label"] = int(row["label"])
        row["label_weak"] = row["label_weak"].lower() == "true"
        row["source_metadata_sha256"] = row["source_metadata_sha256"] or None

    # Preserve every V4 AE/DD row and identity field; P2 will rederive samples
    # from the immutable original source files and compare this exact map.
    manifest = PROJECT / "storage/corpora/corpus-002/manifest_v4_full.csv"
    use = ["clip_id", "source", "source_file", "source_group", "group", "label", "label_name",
           "sub_class", "style", "drone_type", "recording", "start_sample", "n_samples", "kept",
           "exclusion_reason", "duplicate_representative", "pinned"]
    data = pd.read_csv(manifest, usecols=use, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    candidate_specs = json.loads((new / "candidate_specs_itu_musan.json").read_text(encoding="utf-8"))
    legacy_reference = []
    registry = {}
    for code, sid, key in (("ae", "al-emadi-r001", "ae_root"), ("dd", "thesis-r001", "dd_root")):
        rec = json.loads((PROJECT / f"repository/registry/sources/{sid}.json").read_text(encoding="utf-8"))
        root = (PROJECT / rec["original"]).resolve()
        configured = Path(paths[key])
        configured = (PROJECT / configured).resolve() if not configured.is_absolute() else configured.resolve()
        if configured != root:
            raise ValueError(f"{code} configured path differs from registry")
        registry[code] = (sid, root)
    for row in data.to_dict(orient="records"):
        code = row["source"]
        if code not in registry:
            continue
        sid, root = registry[code]
        audio = Path(row["source_file"]).resolve()
        relative = canonical_relative_path(audio.relative_to(root).as_posix())
        file_id = source_file_id(sid, relative)
        inv = source_by_id.get(file_id)
        if inv is None:
            raise ValueError(f"legacy V4 source file absent from P1 inventory: {relative}")
        # Candidate metadata is copied only from the approved V4 mapping.
        label = int(row["label"])
        if code == "ae":
            label_basis = "folder"
            negative = (("N2" if row["style"].casefold() == "esc" else "N1") if label == 0 else None)
            substratum = row["drone_type"] if label else row["style"]
        else:
            label_basis = "filename"
            negative = {"BACKGROUND": "N1", "HELICOPTER": "N5"}.get(row["sub_class"])
            substratum = row["sub_class"]
        candidate = {
            "clip_id": row["clip_id"], "legacy_clip_id": row["clip_id"], "source_file_id": file_id,
            "source": code, "source_revision": sid, "file_sha256": inv["file_sha256"],
            "recording_id": row["recording"] or Path(relative).stem,
            "group_key": row["group"], "substratum": substratum, "negative_stratum": negative,
            "label": label, "label_basis": label_basis, "label_weak": False,
            "start_sample_16k": int(row["start_sample"]), "n_samples": int(row["n_samples"]),
            "candidate_role": "base", "kept": row["kept"].lower() == "true",
            "exclusion_reason": row["exclusion_reason"] or None,
            "duplicate_representative": row["duplicate_representative"] or None,
            "pinned": row["pinned"].lower() == "true", "audio_path": str(audio), "relative_path": relative,
        }
        candidate_specs.append(candidate)
        legacy_reference.append({k: candidate[k] for k in ("clip_id", "source", "group_key", "duplicate_representative", "kept", "exclusion_reason", "pinned")})

    if len([r for r in source_rows if r["source"] == "ae"]) != 11704:
        raise ValueError("AE legacy file inventory differs from V4 reference")
    if len([r for r in source_rows if r["source"] == "dd"]) != 90:
        raise ValueError("DD legacy physical-file inventory differs from V4 reference")
    if len([r for r in candidate_specs if r["source"] == "ae"]) != 11704:
        raise ValueError("AE candidate count differs from V4 reference")
    if len([r for r in candidate_specs if r["source"] == "dd"]) != 900:
        raise ValueError("DD candidate count differs from V4 reference")

    inv_hash = write_csv(out / "inventory_v51.csv", contract_columns("inventory_v51.csv"), source_rows, sort_by=None)
    label_hash = write_csv(out / "label_register.csv", contract_columns("label_register.csv"), label_rows, sort_by=None)
    (out / "candidate_specs.json").write_text(json.dumps(candidate_specs, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    (out / "legacy_reference.json").write_text(json.dumps(legacy_reference, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    quarantine = json.loads((new / "quarantine_itu_musan.json").read_text(encoding="utf-8"))
    (out / "quarantine_itu_musan.json").write_text(json.dumps(quarantine, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    gates = attempt / "phase_gates.jsonl"
    evidence = {"source_inventory_rows": len(source_rows), "label_register_rows": len(label_rows),
                "candidate_specs": len(candidate_specs), "legacy_reference_rows": len(legacy_reference),
                "inventory_sha256": inv_hash, "label_register_sha256": label_hash,
                "legacy_kept": {s: sum(r["source"] == s and r["kept"] for r in candidate_specs) for s in ("ae", "dd")},
                "scope": "complete metadata/candidate inventory assembled; waveform decoding is P2"}
    append_gate(gates, attempt_id=args.attempt_id, phase="P1", gate="P1_COMPLETE_SOURCE_SET",
                state=GateState.PASS, evidence=evidence)
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
