"""Re-derive permitted train/val waveforms and build the V5.1 P4 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

STUDY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STUDY_ROOT.parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from dronedet.cnn_v51.artifacts import CONTRACT_PATH, read_csv  # noqa: E402
from dronedet.cnn_v51.config import load_v51_config  # noqa: E402
from dronedet.cnn_v51.phase_p4 import fit_p4, rederive_train_val_descriptors, write_p4_artifacts  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-attempt", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=STUDY_ROOT / "config" / "v51.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--descriptor-cache", type=Path)
    parser.add_argument("--reuse-descriptors", type=Path)
    args = parser.parse_args()

    attempts_root = PROJECT_ROOT / "local" / "study-v5-1" / "attempts"
    frozen_attempt = args.frozen_attempt.resolve()
    output = args.output.resolve()
    for name, path in (("frozen attempt", frozen_attempt), ("P4 output", output)):
        try:
            path.relative_to(attempts_root.resolve())
        except ValueError as exc:
            raise SystemExit(f"{name} must stay under the ignored attempt directory {attempts_root}") from exc
    if output.exists():
        raise SystemExit(f"refusing to overwrite P4 attempt: {output}")

    frozen = json.loads((frozen_attempt / "FROZEN.json").read_text(encoding="utf-8"))
    artifact_hashes = frozen["artifact_sha256"]
    for filename in ("train.csv", "val.csv", "test.csv", "ood.csv"):
        path = frozen_attempt / filename
        if _sha256(path) != artifact_hashes[filename]:
            raise SystemExit(f"frozen membership artifact hash mismatch: {filename}")

    train = read_csv(frozen_attempt / "train.csv", "train.csv")
    val = read_csv(frozen_attempt / "val.csv", "val.csv")
    # Metadata only: these IDs seal physical files against post-S9 waveform access.
    test = read_csv(frozen_attempt / "test.csv", "test.csv")
    ood = read_csv(frozen_attempt / "ood.csv", "ood.csv")
    inventory = read_csv(args.source_inventory, "inventory_v51.csv")
    config = load_v51_config(args.config)
    paths_file = PROJECT_ROOT / "local" / "study-v5-1" / "paths.json"
    local_paths = json.loads(paths_file.read_text(encoding="utf-8"))

    inventory_sha256 = _sha256(args.source_inventory)
    descriptor_cache = args.descriptor_cache or (output.parent / f"{output.name}_descriptors.json")
    try:
        descriptor_cache.resolve().relative_to(attempts_root.resolve())
    except ValueError as exc:
        raise SystemExit(f"descriptor cache must stay under {attempts_root}") from exc
    if args.reuse_descriptors:
        cached = json.loads(args.reuse_descriptors.read_text(encoding="utf-8"))
        if cached["train_manifest_sha256"] != artifact_hashes["train.csv"] or cached["val_manifest_sha256"] != artifact_hashes["val.csv"]:
            raise SystemExit("reused P4 descriptors are bound to different frozen manifests")
        if cached["source_inventory_sha256"] != inventory_sha256:
            raise SystemExit("reused P4 descriptors are bound to a different source inventory")
        descriptors = cached["rows"]
        if any(row.get("split") not in {"train", "val"} for row in descriptors):
            raise SystemExit("descriptor cache contains a forbidden split")
    else:
        roots: dict[str, Path] = {}
        for source in {row["source"] for row in train + val}:
            source_input = config.inputs.get(source)
            if not source_input:
                raise SystemExit(f"missing local path binding for train/val source {source}")
            path_key = source_input.get("path_key")
            if not path_key or path_key not in local_paths:
                raise SystemExit(f"missing path key for train/val source {source}")
            root = Path(local_paths[path_key])
            if not root.is_absolute():
                root = PROJECT_ROOT / root
            root = root.resolve()
            registry_path = PROJECT_ROOT / "repository" / "registry" / "sources" / f"{source_input['source_revision']}.json"
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registered_root = Path(registry["original"])
            if not registered_root.is_absolute():
                registered_root = PROJECT_ROOT / registered_root
            if root != registered_root.resolve():
                raise SystemExit(f"configured {source} root differs from its registered root")
            roots[source] = root

        def resolve_source(source_row: dict) -> Path:
            source = str(source_row["source"])
            root = roots[source]
            target = root.joinpath(*str(source_row["relative_path"]).split("/")).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise SystemExit(f"source-relative path escapes its registered root: {source_row['source_file_id']}") from exc
            return target

        descriptors = rederive_train_val_descriptors(
            train,
            val,
            inventory,
            sealed_source_file_ids={row["source_file_id"] for row in test + ood},
            train_manifest_sha256=artifact_hashes["train.csv"],
            val_manifest_sha256=artifact_hashes["val.csv"],
            config=config.frontend,
            path_resolver=resolve_source,
        )
        if descriptor_cache.exists():
            raise SystemExit(f"refusing to overwrite descriptor checkpoint: {descriptor_cache}")
        descriptor_cache.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"train_manifest_sha256": artifact_hashes["train.csv"],
                      "val_manifest_sha256": artifact_hashes["val.csv"],
                      "source_inventory_sha256": inventory_sha256, "rows": list(descriptors)}
        descriptor_cache.write_text(json.dumps(checkpoint, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    result = fit_p4(
        train_manifest_rows=train,
        val_manifest_rows=val,
        train_descriptor_rows=[row for row in descriptors if row["split"] == "train"],
        val_descriptor_rows=[row for row in descriptors if row["split"] == "val"],
        train_manifest_sha256=artifact_hashes["train.csv"],
        val_manifest_sha256=artifact_hashes["val.csv"],
        frozen_membership_sha256=frozen["membership_sha256"],
        admitted_sources=frozen["source_list"],
        r_led_verified=True,
        contract=contract,
    )
    summary = write_p4_artifacts(output, result)
    summary["descriptor_checkpoint_sha256"] = _sha256(descriptor_cache)
    (output / "p4_summary.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if summary["state"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
