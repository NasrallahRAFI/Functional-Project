"""Create a noncanonical, metadata-only partial P1 inventory for AE/DD."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT = WORKSPACE.parents[1]
sys_path = str(WORKSPACE / "src")
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from dronedet.cnn_v51.artifacts import contract_columns, write_csv  # noqa: E402
from dronedet.cnn_v51.gates import GateState, append_gate  # noqa: E402
from dronedet.cnn_v51.identity import canonical_relative_path, source_file_id  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checksums(root: Path, inventory: Path) -> dict[str, tuple[int, str]]:
    with inventory.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames != ["path", "bytes", "sha256"]:
            raise ValueError(f"unexpected registered checksum schema: {inventory.name}")
        expected: dict[str, tuple[int, str]] = {}
        for row in rows:
            rel = canonical_relative_path(row["path"])
            if rel in expected:
                raise ValueError(f"duplicate source inventory path: {rel}")
            expected[rel] = (int(row["bytes"]), row["sha256"])
    return expected


def verify_tree(root: Path, expected: dict[str, tuple[int, str]]) -> dict[str, int]:
    paths = sorted(root.joinpath(*rel.split("/")).resolve() for rel in expected)
    actual_rel = {path.relative_to(root.resolve()).as_posix() for path in paths}
    if actual_rel != set(expected):
        raise ValueError("registered source checksum inventory does not match source tree paths")

    def verify(path: Path) -> tuple[str, int, str]:
        rel = path.relative_to(root.resolve()).as_posix()
        return rel, path.stat().st_size, sha256_file(path)

    with ThreadPoolExecutor(max_workers=6) as pool:
        for rel, size, digest in pool.map(verify, paths):
            expected_size, expected_digest = expected[rel]
            if (size, digest) != (expected_size, expected_digest):
                raise ValueError(f"registered source byte identity mismatch: {rel}")
    return {"files": len(expected), "bytes": sum(size for size, _ in expected.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--reuse-source-hash-attempt")
    args = parser.parse_args()
    attempt_dir = PROJECT / "local/study-v5-1/attempts" / args.attempt_id
    if not attempt_dir.is_dir():
        raise SystemExit("attempt does not exist; initialize a new V5.1 attempt first")

    paths_file = PROJECT / "local/study-v5-1/paths.json"
    local_paths = json.loads(paths_file.read_text(encoding="utf-8"))
    v4_manifest = PROJECT / "storage/corpora/corpus-002/manifest_v4_full.csv"
    source_meta = {}
    for code, registry_id, path_key in (
        ("ae", "al-emadi-r001", "ae_root"),
        ("dd", "thesis-r001", "dd_root"),
    ):
        record = json.loads((PROJECT / f"repository/registry/sources/{registry_id}.json").read_text(encoding="utf-8"))
        registered_root = (PROJECT / record["original"]).resolve()
        configured = Path(local_paths[path_key])
        configured_root = (PROJECT / configured).resolve() if not configured.is_absolute() else configured.resolve()
        if configured_root != registered_root:
            raise ValueError(f"{code} local path does not resolve to its registered source root")
        inventory_path = PROJECT / record["inventory"]
        if sha256_file(inventory_path) != record["inventory_sha256"]:
            raise ValueError(f"{code} registered checksum inventory identity changed")
        expected = load_checksums(registered_root, inventory_path)
        source_meta[code] = {
            "revision": registry_id,
            "root": registered_root,
            "inventory": inventory_path,
            "expected": expected,
        }

    byte_evidence = {}
    if args.reuse_source_hash_attempt:
        prior_gates = PROJECT / "local/study-v5-1/attempts" / args.reuse_source_hash_attempt / "phase_gates.jsonl"
        prior_rows = [json.loads(line) for line in prior_gates.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        prior = next((r for r in reversed(prior_rows) if r["gate"] == "P1_AVAILABLE_SOURCE_BYTES"), None)
        if not prior or prior["state"] != "PASS" or not isinstance(prior.get("evidence"), dict):
            raise ValueError("reused attempt has no passing byte-integrity record")
        prior_evidence = prior["evidence"]
        for code, item in source_meta.items():
            prior_source = prior_evidence.get(code, {})
            registered_hash = sha256_file(item["inventory"])
            if prior_source.get("inventory_sha256") != registered_hash:
                raise ValueError(f"{code} checksum-inventory identity changed since the verified attempt")
            byte_evidence[code] = {
                "files": len(item["expected"]),
                "bytes": sum(size for size, _ in item["expected"].values()),
                "inventory_sha256": registered_hash,
                "reused_verified_attempt": args.reuse_source_hash_attempt,
            }
    else:
        for code, item in source_meta.items():
            verified = verify_tree(item["root"], item["expected"])
            verified["inventory_sha256"] = sha256_file(item["inventory"])
            byte_evidence[code] = verified

    # Select the approved text-metadata fields at read time. The V4 manifest
    # also contains descriptors and split columns; none are retained here.
    allowed_manifest_fields = [
        "source", "source_file", "source_group", "group", "label", "label_name",
        "sub_class", "style", "drone_type", "recording",
    ]
    header = pd.read_csv(v4_manifest, nrows=0, encoding="utf-8-sig").columns.tolist()
    if not set(allowed_manifest_fields).issubset(header):
        raise ValueError("V4 metadata reference lacks fields required for the declared AE/DD mapping")
    selected_manifest = pd.read_csv(
        v4_manifest,
        usecols=allowed_manifest_fields,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    manifest_rows: dict[str, list[dict[str, str]]] = {"ae": [], "dd": []}
    for row in selected_manifest.to_dict(orient="records"):
        code = row["source"]
        if code not in source_meta:
            continue
        audio_path = Path(row["source_file"]).resolve()
        relative = canonical_relative_path(audio_path.relative_to(source_meta[code]["root"]).as_posix())
        selected_row = {key: row[key] for key in allowed_manifest_fields}
        selected_row["relative_path"] = relative
        manifest_rows[code].append(selected_row)

    source_rows: list[dict[str, object]] = []
    label_rows: dict[str, dict[str, object]] = {}
    for code, item in source_meta.items():
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in manifest_rows[code]:
            grouped.setdefault(row["relative_path"], []).append(row)
        if len(grouped) != len(set(grouped)):
            raise ValueError(f"{code} contains duplicate source-file paths in V4 metadata")

        # The selected AE/DD audio membership is bound to the already registered
        # V4 map. Headers are read without decoding; bytes were verified above.
        for relative in sorted(grouped):
            rows = grouped[relative]
            identities = {
                (r["label"], r["label_name"], r["sub_class"], r["group"], r["source_group"], r["style"], r["drone_type"])
                for r in rows
            }
            if len(identities) != 1:
                raise ValueError(f"conflicting V4 text metadata for {code}:{relative}")
            label_text, label_name, source_class, group_key, original_group, style, drone_type = next(iter(identities))
            label = int(label_text)
            if code == "ae":
                parts = relative.split("/")
                folder_class = parts[1] if len(parts) > 2 and parts[0] == "Binary_Drone_Audio" else ""
                expected_v4_name = "drone" if folder_class == "yes_drone" else folder_class
                if folder_class not in {"unknown", "yes_drone"} or label_name != expected_v4_name:
                    raise ValueError(f"AE folder class and registered metadata disagree at {relative}")
                if label != int(folder_class == "yes_drone"):
                    raise ValueError(f"AE folder class and registered binary label disagree at {relative}")
                basis = "folder"
                negative_stratum = ("N2" if style.casefold() == "esc" else "N1") if label == 0 else None
                substratum = drone_type if label == 1 else style
                source_class = folder_class
            else:
                stem_class = Path(relative).stem.split("_", 1)[0].upper()
                if stem_class != source_class or source_class not in {"DRONE", "BACKGROUND", "HELICOPTER"}:
                    raise ValueError(f"DD filename and registered class metadata disagree at {relative}")
                if label != int(source_class == "DRONE"):
                    raise ValueError(f"DD class and registered binary label disagree at {relative}")
                basis = "filename"
                negative_stratum = {"BACKGROUND": "N1", "HELICOPTER": "N5"}.get(source_class)
                substratum = source_class

            size, registered_hash = item["expected"][relative]
            audio_path = item["root"].joinpath(*relative.split("/"))
            header = sf.info(str(audio_path))
            file_id = source_file_id(item["revision"], relative)
            sample_row = rows[0]
            recording_id = sample_row["recording"] or Path(relative).stem
            inventory_row = {
                "source_file_id": file_id,
                "source": code,
                "source_revision": item["revision"],
                "relative_path": relative,
                "file_sha256": registered_hash,
                "bytes": size,
                "native_sr": int(header.samplerate),
                "channels": int(header.channels),
                "encoding": f"{header.format}/{header.subtype}",
                "native_frames": int(header.frames),
                "recording_id": recording_id,
                "original_group_key": original_group or group_key,
                "group_key": group_key,
                "source_class": source_class,
                "label": label,
                "label_basis": basis,
                "label_weak": False,
                "substratum": substratum,
                "negative_stratum": negative_stratum,
                "source_metadata_sha256": None,
                "prior_exposure": "prior_training_or_test",
            }
            source_rows.append(inventory_row)
            label_rows[file_id] = {
                "source_file_id": file_id,
                "source": code,
                "source_class": source_class,
                "label": label,
                "label_basis": basis,
                "label_weak": False,
                "source_metadata_sha256": None,
            }

    source_rows.sort(key=lambda row: (str(row["source"]), str(row["source_file_id"])))
    label_materialized = [label_rows[key] for key in sorted(label_rows)]
    partial_dir = attempt_dir / "partial-p1-ae-dd"
    partial_dir.mkdir(parents=True, exist_ok=False)
    inventory_hash = write_csv(partial_dir / "inventory_v51.partial.csv", contract_columns("inventory_v51.csv"), source_rows, sort_by=None)
    label_hash = write_csv(partial_dir / "label_register.partial.csv", contract_columns("label_register.csv"), label_materialized, sort_by=None)

    states = {
        "P1_AVAILABLE_SOURCE_BYTES": (GateState.PASS, byte_evidence),
        "P1_AE_DD_METADATA_HEADERS": (GateState.PASS, {
            "files": len(source_rows), "inventory_sha256": inventory_hash,
            "label_register_sha256": label_hash,
            "scope": "AE/DD partial only; contract schemas and headers, no waveform decode",
        }),
        "P1_COMPLETE_SOURCE_SET": (GateState.NOT_RUN,
            "ITU and MUSAN snapshots are absent; MIC/Wang remain absent from default OOD inventory"),
    }
    gate_path = attempt_dir / "phase_gates.jsonl"
    for gate, (state, evidence) in states.items():
        append_gate(gate_path, attempt_id=args.attempt_id, phase="P1", gate=gate, state=state, evidence=evidence)

    print(json.dumps({
        "attempt_id": args.attempt_id,
        "source_hash_verification": byte_evidence,
        "partial_audio_inventory_rows": len(source_rows),
        "inventory_sha256": inventory_hash,
        "label_register_sha256": label_hash,
        "canonical_final_inventory_written": False,
        "p1_complete_source_set": "NOT_RUN",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
