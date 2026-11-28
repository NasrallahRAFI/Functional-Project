"""Bind P3 freeze inputs to registered inventories and the approved contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT = WORKSPACE.parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from dronedet.cnn_v51.artifacts import contract_columns, write_csv  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--artifact-attempt-id")
    args = parser.parse_args()
    attempt = PROJECT / "local/study-v5-1/attempts" / args.attempt_id
    artifact_attempt = PROJECT / "local/study-v5-1/attempts" / (args.artifact_attempt_id or args.attempt_id)
    p1 = artifact_attempt / "p1-complete"
    p2 = artifact_attempt / "p2/mechanical_ledger.csv"
    if not p2.is_file():
        raise FileNotFoundError(f"P2 ledger missing: {p2}")
    out = attempt / "p3-inputs"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite P3 input bindings: {out}")
    out.mkdir(parents=True)
    quarantine_rows = json.loads((p1 / "quarantine_itu_musan.json").read_text(encoding="utf-8"))
    quarantine = []
    for row in quarantine_rows:
        quarantine.append({"source_file_id": row["source_file_id"], "source": row["source"],
                           "relative_path": row["relative_path"], "file_sha256": row["file_sha256"],
                           "bytes": int(row.get("bytes", 0)), "reason": row["reason"]})
    # X1 exclusions are metadata-only; their size is already source-registered.
    if quarantine:
        inv_rows = []
        import csv
        with (p1 / "inventory_v51.csv").open("r", encoding="utf-8", newline="") as stream:
            inv_rows = list(csv.DictReader(stream))
        sizes = {row["source_file_id"]: int(row["bytes"]) for row in inv_rows}
        for row in quarantine:
            row["bytes"] = sizes.get(row["source_file_id"], row["bytes"])
    write_csv(out / "quarantine_manifest.csv", contract_columns("quarantine_manifest.csv"), quarantine,
              sort_by="source_file_id")

    source_hashes = {}
    source_codes = []
    for code, sid in (("ae", "al-emadi-r001"), ("dd", "thesis-r001"),
                      ("itu", "itu-aris-r001"), ("musan", "musan-r001")):
        rec = json.loads((PROJECT / f"repository/registry/sources/{sid}.json").read_text(encoding="utf-8"))
        inv_path = PROJECT / rec["inventory"]
        digest = sha256(inv_path)
        if digest != rec["inventory_sha256"]:
            raise ValueError(f"registered source inventory changed for {sid}")
        source_hashes[sid] = digest
        source_codes.append(code)
    dependencies = {}
    for line in (WORKSPACE / "requirements-v51.txt").read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            name, version = item.split("==", 1)
            dependencies[name] = version
    config = __import__("yaml").safe_load((WORKSPACE / "config/v51.yaml").read_text(encoding="utf-8"))
    bindings = {
        "study_id": "study-v5-1", "source_inventory_sha256": source_hashes,
        "plan_sha256": config["approval"]["plan_sha256"],
        "contract_sha256": config["approval"]["contract_sha256"],
        "dependencies": dependencies, "seed": 51,
        "group_id_grid_rules": {
            "grouping": "source-qualified physical recording/group; exact-byte MUSAN aliases share canonical group",
            "pooled_split": "two-stage SGKF; seed=51; n_test=7; n_val=6; input sorted by clip_id",
            "musan_split": {"noise": [0.6, 0.2, 0.2], "music_speech": [0.8, 0.1, 0.1],
                            "rank_key": ["source_revision", "canonical_relative_path"]},
            "new_clip_grid_samples": 16000, "sample_rate": 16000,
        },
        "source_list": sorted(source_codes), "L": 0.5,
        "transform_hash": "bb7a21a1e5fb317c43e1e214bc04af6e46e04d75053383d1c62ef6849c9466c7",
        "b2_reference_floor_policy": config["sampling"]["effective_size"],
    }
    (out / "bindings.json").write_text(json.dumps(bindings, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"quarantine_rows": len(quarantine), "source_inventory_sha256": source_hashes,
                      "source_list": bindings["source_list"], "bindings_sha256": sha256(out / "bindings.json")},
                     sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
