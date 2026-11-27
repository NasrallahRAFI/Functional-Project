"""Run the authorized V5.1 P5 phase from a Google Colab/Drive layout.

This is a path adapter for Colab. P5 calculations and gate rules are delegated
to the same implementation used by the local runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_source_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source roots must use SOURCE=PATH")
    source, raw_path = value.split("=", 1)
    if not source or not raw_path:
        raise argparse.ArgumentTypeError("source roots must use SOURCE=PATH")
    return source, Path(raw_path).expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True,
                        help="Drive copy of the study-v5-1 code workspace")
    parser.add_argument("--frozen-attempt", type=Path, required=True,
                        help="P3 directory containing FROZEN.json and split manifests")
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--p4-attempt", type=Path, required=True,
                        help="P4 directory containing p4_summary.json and weight_fit.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="New, empty P5 output directory on Drive")
    parser.add_argument("--source-root", type=_parse_source_root, action="append", default=[],
                        metavar="SOURCE=PATH", help="One registered source root per train/val source")
    args = parser.parse_args()

    study_root = args.study_root.expanduser().resolve()
    sys.path.insert(0, str(study_root / "src"))
    from dronedet.cnn_v51.artifacts import read_csv
    from dronedet.cnn_v51.config import load_v51_config
    from dronedet.cnn_v51.phase_p5 import build_p5_cache, b6_level_gate, write_p5_artifacts

    frozen_attempt = args.frozen_attempt.expanduser().resolve()
    p4_attempt = args.p4_attempt.expanduser().resolve()
    inventory_path = args.source_inventory.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing P5 output: {output}")
    if not frozen_attempt.is_dir() or not p4_attempt.is_dir():
        raise SystemExit("frozen P3 and P4 input directories must exist")

    frozen = json.loads((frozen_attempt / "FROZEN.json").read_text(encoding="utf-8"))
    contract_path = study_root / "docs" / "v5.1" / "DATA_PREPARATION_CNN_V5_1_CONTRACT.json"
    if not contract_path.is_file():
        raise SystemExit(f"missing approved V5.1 contract: {contract_path}")
    contract_sha256 = _sha256(contract_path)
    expected_contract_sha256 = frozen.get("contract_sha256")
    if contract_sha256 != expected_contract_sha256:
        raise SystemExit(
            "approved contract hash mismatch: "
            f"frozen={expected_contract_sha256!r}, local={contract_sha256}"
        )
    for filename in ("train.csv", "val.csv", "test.csv", "ood.csv"):
        if _sha256(frozen_attempt / filename) != frozen["artifact_sha256"][filename]:
            raise SystemExit(f"frozen membership artifact hash mismatch: {filename}")

    p4_summary = json.loads((p4_attempt / "p4_summary.json").read_text(encoding="utf-8"))
    p4_weights = json.loads((p4_attempt / "weight_fit.json").read_text(encoding="utf-8"))
    if p4_weights.get("frozen_membership_sha256") != frozen["membership_sha256"]:
        raise SystemExit("P4 artifact is bound to a different frozen membership")
    if p4_weights.get("train_manifest_sha256") != frozen["artifact_sha256"]["train.csv"]:
        raise SystemExit("P4 artifact is bound to a different train manifest")
    required_p4 = {"R-LED", "B0", "B1", "B2", "B3_B4"}
    gates = p4_summary.get("gates", {})
    if required_p4 - set(gates) or any(gates[gate] != "PASS" for gate in required_p4):
        raise SystemExit("P5 requires passing P4 identity, reference and weight gates")

    train = read_csv(frozen_attempt / "train.csv", "train.csv")
    val = read_csv(frozen_attempt / "val.csv", "val.csv")
    test = read_csv(frozen_attempt / "test.csv", "test.csv")
    ood = read_csv(frozen_attempt / "ood.csv", "ood.csv")
    inventory = read_csv(inventory_path, "inventory_v51.csv")
    config = load_v51_config(study_root / "config" / "v51.yaml")

    roots = dict(args.source_root)
    required_sources = {str(row["source"]) for row in train + val}
    if required_sources - roots.keys():
        raise SystemExit(f"missing source roots: {sorted(required_sources - roots.keys())}")
    for source in required_sources:
        if not roots[source].is_dir():
            raise SystemExit(f"source root for {source} is not a directory: {roots[source]}")

    def resolve_source(source_row: dict) -> Path:
        source = str(source_row["source"])
        root = roots[source]
        rel = str(source_row["relative_path"])
        if rel.startswith(("/", "\\")) or "\\" in rel:
            raise SystemExit(f"non-canonical source relative path: {rel!r}")
        path = root.joinpath(*rel.split("/")).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SystemExit("inventory path escapes its configured source root") from exc
        return path

    print(f"P5 source rows: train={len(train)}, val={len(val)}; sources={sorted(required_sources)}")
    print(f"Frozen membership: {frozen['membership_sha256']}")
    print("P5 A-chk ranks every eligible train crop pair exactly; this is CPU-bound.")
    result = build_p5_cache(
        train, val, inventory,
        sealed_source_file_ids={row["source_file_id"] for row in test + ood},
        train_manifest_sha256=frozen["artifact_sha256"]["train.csv"],
        val_manifest_sha256=frozen["artifact_sha256"]["val.csv"],
        frozen_membership_sha256=frozen["membership_sha256"],
        config=config.frontend, path_resolver=resolve_source,
    )
    weights = {clip_id: float(weight) for clip_id, weight in
               zip(p4_weights["clip_ids"], p4_weights["w_train"])}
    result["b6"] = b6_level_gate(train, result["band_lookup"], weights)
    written = write_p5_artifacts(output, result, config=config.frontend)
    print(json.dumps({"state": written["state"], "summary": written["summary"],
                      "artifact_sha256": written["artifact_sha256"],
                      "output": str(output)}, sort_keys=True, separators=(",", ":")))
    return 0 if written["state"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
