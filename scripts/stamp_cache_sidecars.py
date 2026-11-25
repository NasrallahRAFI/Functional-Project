"""Verify existing cache provenance; metadata is written only by canonical builders.

This former adoption script no longer invents test dimensions or stamps old
artifacts as newly built. Test sidecars are checked as historical metadata only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dronedet.cache_metadata import file_sha256, sidecar_path, verify_cache_sidecar
from dronedet.config import legacy_params_hash, load_project_config
from dronedet.data_preparation import locate_dataset_root
from dronedet.features import file_to_power, load_cache_index, load_power_cache, load_tabular, tabular


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sample_count < 1:
        parser.error("sample count must be positive")
    if args.output and args.output.exists():
        raise FileExistsError("refusing to overwrite evidence")
    project = load_project_config(ROOT / "config.yaml")
    paths, config = project["P"], project["fe"]
    raw = locate_dataset_root(paths["raw_root"]) / "Binary_Drone_Audio"
    rng = np.random.default_rng(args.seed)
    results = {}
    for split in ("train", "val"):
        index = load_cache_index(paths, split)
        power = load_power_cache(paths, split)
        features = load_tabular(paths, split)["X"]
        sampled = sorted(rng.choice(len(index), size=min(args.sample_count, len(index)), replace=False))
        numbers = []
        for i in sampled:
            row = index.iloc[int(i)]
            fresh, _ = file_to_power(raw / str(row.clip_id), config)
            start, count = int(row.start_frame), int(row.n_frames)
            stored = power[:, start:start + count]
            if stored.shape != fresh.shape:
                raise ValueError(f"{split}: recomputed shape mismatch for {row.clip_id}")
            numbers.append({"clip_id": str(row.clip_id), "position": int(i),
                            "power_max_abs": float(np.max(abs(fresh - stored))),
                            "tabular_max_abs": float(np.max(abs(tabular(fresh, config) - features[int(i)])))})
        results[split] = {"status": "PASS" if all(x["power_max_abs"] == 0 and x["tabular_max_abs"] == 0 for x in numbers) else "FAIL",
                          "sampled_clips": len(numbers), "recomputations": numbers}
        power._mmap.close()
    test_metadata = {}
    train_digest = file_sha256(paths["splits"] / "train.csv")
    test_digest = file_sha256(paths["splits"] / "test.csv")
    for path in (paths["spec"] / "mel_power_test.npy", paths["tab"] / "test.npz"):
        metadata = verify_cache_sidecar(path, frontend_params_hash=legacy_params_hash(config),
                                       train_manifest_sha256=train_digest)
        if metadata.get("split_manifest_sha256") != test_digest:
            raise ValueError("test sidecar split-manifest identity differs")
        test_metadata[path.name] = {"status": "HISTORICAL_METADATA_ONLY",
                                    "sidecar_sha256": file_sha256(sidecar_path(path)),
                                    "recorded_shape": metadata.get("array_shape"),
                                    "recorded_dtype": metadata.get("array_dtype"),
                                    "recorded_provenance": metadata.get("provenance"),
                                    "array_payload_read": False, "metadata_rewritten": False}
    result = {"generated_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
              "status": "PASS" if all(v["status"] == "PASS" for v in results.values()) else "FAIL",
              "train_validation": results, "test_sidecars": test_metadata,
              "sidecars_written": 0, "test_array_payloads_opened": False}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    summary = {"status": result["status"], "sidecars_written": 0,
               "test_array_payloads_opened": False, "record": str(args.output) if args.output else None,
               "recomputations": {split: {"sampled_clips": record["sampled_clips"],
                   "max_power_abs": max(row["power_max_abs"] for row in record["recomputations"]),
                   "max_tabular_abs": max(row["tabular_max_abs"] for row in record["recomputations"])}
                   for split, record in results.items()}}
    print(json.dumps(summary if args.output else result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
