"""Build V5.1 train/validation caches, statistics and P5 gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

STUDY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STUDY_ROOT.parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from dronedet.cnn_v51.artifacts import read_csv  # noqa: E402
from dronedet.cnn_v51.config import load_v51_config  # noqa: E402
from dronedet.cnn_v51.phase_p5 import (  # noqa: E402
    b6_level_gate, build_p5_cache, validate_p5_authorization, write_p5_artifacts,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-attempt", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--p4-attempt", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=STUDY_ROOT / "config" / "v51.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    attempts_root = PROJECT_ROOT / "local" / "study-v5-1" / "attempts"
    frozen_attempt, p4_attempt, output = (path.resolve() for path in (args.frozen_attempt, args.p4_attempt, args.output))
    for label, path in (("frozen attempt", frozen_attempt), ("P4 attempt", p4_attempt), ("P5 output", output)):
        try:
            path.relative_to(attempts_root.resolve())
        except ValueError as exc:
            raise SystemExit(f"{label} must stay under ignored attempt directory {attempts_root}") from exc
    if output.exists():
        raise SystemExit(f"refusing to overwrite P5 attempt: {output}")

    frozen = json.loads((frozen_attempt / "FROZEN.json").read_text(encoding="utf-8"))
    validate_p5_authorization(frozen)
    for filename in ("train.csv", "val.csv", "test.csv", "ood.csv"):
        if _sha256(frozen_attempt / filename) != frozen["artifact_sha256"][filename]:
            raise SystemExit(f"frozen membership artifact hash mismatch: {filename}")
    p4_summary = json.loads((p4_attempt / "p4_summary.json").read_text(encoding="utf-8"))
    p4_weights = json.loads((p4_attempt / "weight_fit.json").read_text(encoding="utf-8"))
    # The P4 summary is a compact gate index; its detailed identity bindings
    # live in weight_fit.json. Validate those bindings before consuming weights.
    if p4_weights.get("frozen_membership_sha256") != frozen["membership_sha256"]:
        raise SystemExit("P4 artifact is bound to a different frozen membership")
    if p4_weights.get("train_manifest_sha256") != frozen["artifact_sha256"]["train.csv"]:
        raise SystemExit("P4 artifact is bound to a different train manifest")
    required_p4 = {"R-LED", "B0", "B1", "B2", "B3_B4"}
    if required_p4 - set(p4_summary["gates"]) or any(p4_summary["gates"][gate] != "PASS" for gate in required_p4):
        raise SystemExit("P5 requires passing P4 identity, reference and weight gates")

    train = read_csv(frozen_attempt / "train.csv", "train.csv")
    val = read_csv(frozen_attempt / "val.csv", "val.csv")
    test = read_csv(frozen_attempt / "test.csv", "test.csv")
    ood = read_csv(frozen_attempt / "ood.csv", "ood.csv")
    inventory = read_csv(args.source_inventory, "inventory_v51.csv")
    config = load_v51_config(args.config)
    paths_file = PROJECT_ROOT / "local" / "study-v5-1" / "paths.json"
    local_paths = json.loads(paths_file.read_text(encoding="utf-8"))
    roots: dict[str, Path] = {}
    for source in {row["source"] for row in train + val}:
        source_input = config.inputs.get(source)
        if not source_input or not source_input.get("path_key") or source_input["path_key"] not in local_paths:
            raise SystemExit(f"missing registered local root binding for source {source}")
        root = Path(local_paths[source_input["path_key"]])
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        root = root.resolve()
        registry_path = PROJECT_ROOT / "repository" / "registry" / "sources" / f"{source_input['source_revision']}.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        registered = Path(registry["original"])
        if not registered.is_absolute():
            registered = PROJECT_ROOT / registered
        if root != registered.resolve():
            raise SystemExit(f"configured {source} root differs from registered source root")
        roots[source] = root

    def resolve_source(source_row: dict) -> Path:
        root = roots[str(source_row["source"])]
        path = root.joinpath(*str(source_row["relative_path"]).split("/")).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SystemExit("inventory path escapes its registered source root") from exc
        return path

    result = build_p5_cache(
        train, val, inventory,
        sealed_source_file_ids={row["source_file_id"] for row in test + ood},
        train_manifest_sha256=frozen["artifact_sha256"]["train.csv"],
        val_manifest_sha256=frozen["artifact_sha256"]["val.csv"],
        frozen_membership_sha256=frozen["membership_sha256"],
        config=config.frontend, path_resolver=resolve_source,
    )
    weights = {clip_id: float(weight) for clip_id, weight in zip(p4_weights["clip_ids"], p4_weights["w_train"])}
    result["b6"] = b6_level_gate(train, result["band_lookup"], weights)
    written = write_p5_artifacts(output, result, config=config.frontend)
    print(json.dumps({"state": written["state"], "summary": written["summary"],
                      "artifact_sha256": written["artifact_sha256"]}, sort_keys=True, separators=(",", ":")))
    return 0 if written["state"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
