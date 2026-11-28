"""Build attempt-local ITU/MUSAN metadata inventories and candidate windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import sys

import soundfile as sf

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT = WORKSPACE.parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from dronedet.cnn_v51.artifacts import contract_columns, write_csv  # noqa: E402
from dronedet.cnn_v51.gates import GateState, append_gate  # noqa: E402
from dronedet.cnn_v51.identity import canonical_relative_path, clip_id, source_file_id  # noqa: E402

X1 = {"drone", "uav", "quadcopter", "quadrotor", "multirotor"}
X2 = {"helicopter", "propeller", "rotor", "aircraft", "airplane", "jet", "engine", "motor"}
TOKEN = re.compile(r"[^a-z0-9]+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokens(value: str) -> set[str]:
    return {item for item in TOKEN.split(value.lower()) if item}


def read_checksum_inventory(path: Path) -> dict[str, tuple[int, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames != ["path", "bytes", "sha256"]:
            raise ValueError(f"unexpected checksum schema: {path}")
        output = {}
        for row in rows:
            rel = canonical_relative_path(row["path"])
            if rel in output:
                raise ValueError(f"duplicate inventory path: {rel}")
            output[rel] = int(row["bytes"]), row["sha256"]
        return output


def merged_itu_groups(groups: dict) -> dict[int, str]:
    intervals = sorted(groups["intervals"], key=lambda row: (float(row["start_s"]), int(row["group_id"])))
    merged: list[dict] = []
    mapping: dict[int, str] = {}
    for row in intervals:
        current = {"start": float(row["start_s"]), "end": float(row["end_s"]), "ids": [int(row["group_id"])]}
        if merged and current["start"] <= merged[-1]["end"] + 1.0:
            merged[-1]["end"] = max(merged[-1]["end"], current["end"])
            merged[-1]["ids"].append(current["ids"][0])
        else:
            merged.append(current)
    for group in merged:
        canonical = min(group["ids"])
        name = f"drone_interval_{canonical:02d}"
        for original in group["ids"]:
            mapping[original] = name
    return mapping


def annotation_index(root: Path) -> tuple[dict[str, list[tuple[str, str]]], list[dict[str, str]]]:
    by_stem: dict[str, list[tuple[str, str]]] = {}
    docs: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.upper() not in {"ANNOTATIONS", "LICENSE"}:
            continue
        rel = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        docs.append({"path": rel, "kind": path.name.upper(), "sha256": digest})
        if path.name.upper() != "ANNOTATIONS":
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            fields = line.strip().split()
            if not fields:
                continue
            stem = Path(fields[0]).stem
            by_stem.setdefault(stem, []).append((rel, " ".join(fields[1:])))
    return by_stem, docs


def p2_spec(source: str, revision: str, file_id: str, file_hash: str, relative: str,
            group: str, substratum: str, negative: str | None, label: int, basis: str,
            weak: bool, recording: str, start: int, role: str) -> dict:
    return {
        "clip_id": clip_id(source, file_id, start, 16000), "legacy_clip_id": None,
        "source_file_id": file_id, "source": source, "source_revision": revision,
        "file_sha256": file_hash, "recording_id": recording, "group_key": group,
        "substratum": substratum, "negative_stratum": negative, "label": label,
        "label_basis": basis, "label_weak": weak, "start_sample_16k": start,
        "n_samples": 16000, "candidate_role": role, "kept": True,
        "exclusion_reason": None, "duplicate_representative": None, "pinned": False,
        "audio_path": "", "relative_path": relative,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--reuse-source-hash-attempt")
    args = parser.parse_args()
    attempt = PROJECT / "local/study-v5-1/attempts" / args.attempt_id
    if not attempt.is_dir():
        raise SystemExit("attempt does not exist")
    gate_path = attempt / "phase_gates.jsonl"
    out = attempt / "p1-itu-musan"
    out.mkdir(parents=True, exist_ok=False)
    paths = json.loads((PROJECT / "local/study-v5-1/paths.json").read_text(encoding="utf-8"))
    sources: dict[str, dict] = {}
    for code, sid, key in (("itu", "itu-aris-r001", "itu_root"), ("musan", "musan-r001", "musan_root")):
        reg = json.loads((PROJECT / f"repository/registry/sources/{sid}.json").read_text(encoding="utf-8"))
        inventory = PROJECT / reg["inventory"]
        if sha256_file(inventory) != reg["inventory_sha256"]:
            raise ValueError(f"{code} checksum inventory hash differs from registry")
        root = (PROJECT / reg["original"]).resolve()
        configured = Path(paths[key])
        configured = (PROJECT / configured).resolve() if not configured.is_absolute() else configured.resolve()
        if configured != root:
            raise ValueError(f"{code} configured path differs from registered root")
        expected = read_checksum_inventory(inventory)
        sources[code] = {"revision": sid, "root": root, "expected": expected, "inventory_hash": reg["inventory_sha256"]}

    prior_gate = None
    if args.reuse_source_hash_attempt:
        prior_path = PROJECT / "local/study-v5-1/attempts" / args.reuse_source_hash_attempt / "phase_gates.jsonl"
        for row in (json.loads(line) for line in prior_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()):
            if row.get("gate") == "P1_REGISTERED_NEW_SOURCE_BYTES" and row.get("state") == "PASS":
                prior_gate = row
        if not prior_gate:
            raise ValueError("prior attempt has no passing new-source byte-integrity gate")
    byte_evidence = {}
    for code, item in sources.items():
        if prior_gate:
            prior = prior_gate["evidence"].get(code, {})
            if prior.get("inventory_sha256") != item["inventory_hash"]:
                raise ValueError(f"{code} inventory changed since byte verification")
            byte_evidence[code] = {**prior, "reused_verified_attempt": args.reuse_source_hash_attempt}
        else:
            actual = {p.relative_to(item["root"]).as_posix() for p in item["root"].rglob("*") if p.is_file()}
            expected = set(item["expected"])
            if actual != expected:
                raise ValueError(f"{code} tree paths differ from checksum inventory")
            for rel, (size, digest) in item["expected"].items():
                path = item["root"].joinpath(*rel.split("/"))
                if path.stat().st_size != size or sha256_file(path) != digest:
                    raise ValueError(f"{code} file checksum mismatch: {rel}")
            byte_evidence[code] = {"files": len(expected), "bytes": sum(x[0] for x in item["expected"].values()),
                                   "inventory_sha256": item["inventory_hash"]}

    inventory_rows: list[dict] = []
    label_rows: list[dict] = []
    specs: list[dict] = []
    excluded: list[dict] = []
    itu_root = sources["itu"]["root"]
    itu_groups_path = itu_root / "groups.json"
    groups = json.loads(itu_groups_path.read_text(encoding="utf-8"))
    interval_mapping = merged_itu_groups(groups)
    metadata_hash = sha256_file(itu_groups_path)
    segment_by_path = {}
    for seg in groups["segments"]:
        rel = canonical_relative_path(seg["file"])
        if rel in segment_by_path:
            raise ValueError(f"ITU segment repeated in groups.json: {rel}")
        segment_by_path[rel] = seg
    for rel in sorted(segment_by_path):
        if rel not in sources["itu"]["expected"]:
            raise ValueError(f"ITU metadata names a file absent from registered inventory: {rel}")
        seg = segment_by_path[rel]
        label = int(seg["label"] == "drone")
        original = str(seg["group"])
        if label:
            match = re.fullmatch(r"drone_interval_(\d+)", original)
            if not match or int(match.group(1)) not in interval_mapping:
                raise ValueError(f"ITU drone file has unresolved interval group: {rel}: {original}")
            group = interval_mapping[int(match.group(1))]
            basis, weak, negative, substratum = "interval_annotation", False, None, "drone"
        else:
            if not re.fullmatch(r"background_block_\d+", original):
                raise ValueError(f"ITU background file has unresolved block: {rel}: {original}")
            group = original
            basis, weak, negative, substratum = "unannotated_stretch", True, "N1", "background"
        size, file_hash = sources["itu"]["expected"][rel]
        path = itu_root.joinpath(*rel.split("/"))
        info = sf.info(str(path))
        if info.samplerate != 22050 or info.channels != 1 or info.frames != 22050:
            raise ValueError(f"ITU file violates the registered one-second mono 22.05-kHz format: {rel}")
        fid = source_file_id("itu-aris-r001", rel)
        row = {"source_file_id": fid, "source": "itu", "source_revision": "itu-aris-r001", "relative_path": rel,
               "file_sha256": file_hash, "bytes": size, "native_sr": info.samplerate, "channels": info.channels,
               "encoding": f"{info.format}/{info.subtype}", "native_frames": info.frames,
               "recording_id": "itu-aris-lab-session", "original_group_key": original, "group_key": f"itu:{group}",
               "source_class": seg["label"], "label": label, "label_basis": basis, "label_weak": weak,
               "substratum": substratum, "negative_stratum": negative, "source_metadata_sha256": metadata_hash,
               "prior_exposure": "unestablished"}
        inventory_rows.append(row)
        label_rows.append({k: row[k] for k in ("source_file_id", "source", "source_class", "label", "label_basis", "label_weak", "source_metadata_sha256")})
        spec = p2_spec("itu", "itu-aris-r001", fid, file_hash, rel, row["group_key"], substratum, negative,
                       label, basis, weak, row["recording_id"], 0, "base")
        spec["audio_path"] = str(path)
        specs.append(spec)

    musan_root = sources["musan"]["root"]
    annotations, documentation = annotation_index(musan_root)
    doc_digest = hashlib.sha256(json.dumps(documentation, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    musan_audio = sorted(path for path in musan_root.rglob("*.wav") if path.is_file())
    stem_to_paths: dict[str, list[Path]] = {}
    annotation_audit: list[dict[str, object]] = []
    missing_annotations: list[str] = []
    x2_files = 0
    for path in musan_audio:
        stem_to_paths.setdefault(path.stem, []).append(path)
    for path in musan_audio:
        rel = canonical_relative_path(path.relative_to(musan_root).as_posix())
        if rel not in sources["musan"]["expected"]:
            raise ValueError(f"MUSAN audio absent from registered checksum inventory: {rel}")
        parts = Path(rel).parts
        category = parts[0]
        if category not in {"noise", "music", "speech"}:
            raise ValueError(f"unexpected MUSAN audio class directory: {rel}")
        info = sf.info(str(path))
        if info.samplerate != 16000 or info.channels != 1:
            raise ValueError(f"MUSAN file violates the registered mono 16-kHz format: {rel}")
        associated = annotations.get(path.stem, [])
        if len(stem_to_paths[path.stem]) > 1:
            raise ValueError(f"ambiguous MUSAN annotation basename: {path.stem}")
        all_tokens = tokens(path.name + " " + rel)
        for _, fields in associated:
            all_tokens |= tokens(fields)
        size, file_hash = sources["musan"]["expected"][rel]
        fid = source_file_id("musan-r001", rel)
        if not associated:
            missing_annotations.append(rel)
        matched_x2 = sorted(all_tokens & X2)
        if matched_x2:
            x2_files += 1
        annotation_audit.append({"source_file_id": fid, "relative_path": rel,
                                 "annotation_files": sorted({doc for doc, _ in associated}),
                                 "annotation_rows": len(associated), "annotation_missing": not bool(associated),
                                 "x1_tokens": sorted(all_tokens & X1), "x2_tokens": matched_x2})
        canonical_group = f"musan:sha256:{file_hash}"
        if all_tokens & X1:
            excluded.append({"source_file_id": fid, "source": "musan", "relative_path": rel,
                             "file_sha256": file_hash, "reason": "K1_X1_drone_token",
                             "matched_tokens": sorted(all_tokens & X1), "annotation_rows": len(associated)})
            continue
        negative = "N5" if all_tokens & X2 else {"noise": "N1", "music": "N4", "speech": "N3"}[category]
        substratum = "X2_rotor_vehicle" if negative == "N5" else category
        row = {"source_file_id": fid, "source": "musan", "source_revision": "musan-r001", "relative_path": rel,
               "file_sha256": file_hash, "bytes": size, "native_sr": info.samplerate, "channels": info.channels,
               "encoding": f"{info.format}/{info.subtype}", "native_frames": info.frames,
               "recording_id": fid, "original_group_key": rel, "group_key": canonical_group,
               "source_class": category, "label": 0, "label_basis": "dataset_definition", "label_weak": False,
               "substratum": substratum, "negative_stratum": negative, "source_metadata_sha256": doc_digest,
               "prior_exposure": "prior_diagnostic" if category == "noise" else "unestablished"}
        inventory_rows.append(row)
        label_rows.append({k: row[k] for k in ("source_file_id", "source", "source_class", "label", "label_basis", "label_weak", "source_metadata_sha256")})
        n = info.frames
        max_m = 12 if category == "noise" else 4
        k_count = n // 16000
        m = min(max_m, k_count)
        if m == 0:
            continue
        if m == 1:
            base_starts = [0]
        elif k_count < max_m:
            base_starts = [16000 * k for k in range(m)]
        else:
            base_starts = [int(k * (n - 16000) // (m - 1)) for k in range(m)]
        stride = 16000 if category == "noise" else 160000
        reservoir_starts = list(range(0, n - 16000 + 1, stride))
        start_roles = {start: ("base_and_reservoir" if start in set(reservoir_starts) else "base") for start in base_starts}
        for start in reservoir_starts:
            start_roles.setdefault(start, "reservoir_candidate")
        for start, role in sorted(start_roles.items()):
            spec = p2_spec("musan", "musan-r001", fid, file_hash, rel, canonical_group, substratum, negative,
                           0, "dataset_definition", False, fid, start, role)
            spec["audio_path"] = str(path)
            specs.append(spec)

    # The MUSAN registry includes all files; require every WAV and every declared
    # ITU segment to have been represented, while preserving excluded files.
    if len(segment_by_path) != 5491:
        raise ValueError(f"unexpected ITU segment count: {len(segment_by_path)}")
    if len(musan_audio) != 2016:
        raise ValueError(f"unexpected MUSAN audio count: {len(musan_audio)}")
    inventory_rows.sort(key=lambda r: (r["source"], r["source_file_id"]))
    label_rows.sort(key=lambda r: (r["source"], r["source_file_id"]))
    inv_hash = write_csv(out / "inventory_itu_musan.csv", contract_columns("inventory_v51.csv"), inventory_rows, sort_by=None)
    labels_hash = write_csv(out / "label_register_itu_musan.csv", contract_columns("label_register.csv"), label_rows, sort_by=None)
    (out / "candidate_specs_itu_musan.json").write_text(json.dumps(specs, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    (out / "quarantine_itu_musan.json").write_text(json.dumps(excluded, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (out / "source_documentation.json").write_text(json.dumps(documentation, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (out / "musan_annotation_audit.json").write_text(json.dumps(annotation_audit, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    evidence = {"byte_integrity": byte_evidence, "inventory_rows": len(inventory_rows), "label_rows": len(label_rows),
                "inventory_sha256": inv_hash, "label_register_sha256": labels_hash, "candidate_specs": len(specs),
                "quarantine_x1_files": len(excluded), "musan_documentation_files": len(documentation),
                "musan_documentation_index_sha256": doc_digest,
                "musan_annotation_missing_files": len(missing_annotations),
                "musan_annotation_missing_paths": missing_annotations,
                "musan_x2_files_remapped_to_n5": x2_files,
                "itu_drone_segments": sum(r["source"] == "itu" and r["label"] == 1 for r in inventory_rows),
                "itu_background_segments": sum(r["source"] == "itu" and r["label"] == 0 for r in inventory_rows),
                "scope": "ITU/MUSAN metadata and candidates only; no waveform decode or canonical membership"}
    append_gate(gate_path, attempt_id=args.attempt_id, phase="P1", gate="P1_REGISTERED_NEW_SOURCE_BYTES",
                state=GateState.PASS, evidence=byte_evidence)
    append_gate(gate_path, attempt_id=args.attempt_id, phase="P1", gate="P1_ITU_MUSAN_METADATA_CANDIDATES",
                state=GateState.PASS, evidence=evidence)
    append_gate(gate_path, attempt_id=args.attempt_id, phase="P1", gate="P1_COMPLETE_SOURCE_SET",
                state=GateState.NOT_RUN, evidence="AE/DD and ITU/MUSAN partial inventories exist; they must be reconciled with full candidate and K1/quarantine contract before P1 closure.")
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
