"""Verify frozen V1 identities and record their V2 domain-separated successors."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dronedet.config import (cache_content_hash, config_hash, decision_hash,
                             legacy_cache_content_hash,
                             legacy_config_hash, load_config, normalization_lineage_hash,
                             tabular_content_hash, transform_hash)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LEGACY_PARAMS = "97544883461de11a1ee51cc17d7028431d6d0ea8f175effc8e63bc18595f28fc"
EXPECTED_LEGACY_LINEAGE = "327b5e0c54632f3c7e40533d2d0b521faa876cbaf58b8dd8c5bfa57fe9f6b8f0"
EXPECTED_LEGACY_POWER = "6d435113ecb663cc08145c56b34da6def8fda2283acbd2855da02dacfe7360d0"
EXPECTED_STATS_FILE = "545aa9cfdded3ea9a8aa4fd2053e1b6d55c80329a4cdada1f246571cf1a2e5df"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_record(output: Path) -> dict:
    config = load_config(ROOT / "config.yaml")
    stats_path = ROOT / "data/processed/spectrograms/norm_stats_global.npz"
    stats_file_hash = sha256(stats_path)
    import numpy as np
    with np.load(stats_path, allow_pickle=False) as archive:
        stats = {"mean": archive["mean"], "std": archive["std"]}
    old_params, old_lineage = legacy_config_hash(config), legacy_config_hash(config, stats)
    if old_params != EXPECTED_LEGACY_PARAMS:
        raise RuntimeError(f"V1 config identity differs from the frozen value: {old_params}")
    if old_lineage != EXPECTED_LEGACY_LINEAGE:
        raise RuntimeError(f"V1 statistics identity differs from the frozen value: {old_lineage}")
    old_power = legacy_cache_content_hash(config)
    if old_power != EXPECTED_LEGACY_POWER:
        raise RuntimeError(f"V1 power-cache identity differs from the frozen value: {old_power}")
    if stats_file_hash != EXPECTED_STATS_FILE:
        raise RuntimeError(f"frozen statistics file digest differs: {stats_file_hash}")
    record = {
        "migration": "frontend_identity_v1_to_v2",
        "verified_before_derivation": True,
        "identity_schema": 2,
        "legacy": {"params_hash": old_params, "power_content_hash": old_power,
                   "statistics_lineage_hash": old_lineage,
                   "statistics_file_sha256": stats_file_hash},
        "new": {"power_content_hash": cache_content_hash(config),
                "tabular_content_hash": tabular_content_hash(config),
                "transform_hash": transform_hash(config),
                "normalization_lineage_hash": normalization_lineage_hash(config, stats),
                "decision_hash": decision_hash(
                    config, policy_id="threshold-single-v1", smoothing_windows=1
                ),
                "config_hash_alias": config_hash(config, stats)},
        "numeric_inputs_unchanged": True,
        "weights_or_manifests_changed": False,
        "decision_settings": {"policy_id": "threshold-single-v1",
                               "interval_s": config.decision_interval_s,
                               "smoothing_windows": 1},
        "statistics_fit_policy": "train_stats_eligible_per_band_floor_mask_v1",
        "note": "V2 identities describe domains; they do not restamp frozen artifacts.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to replace migration record: {output}")
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/interim/frontend_identity_migration_v2.json")
    args = parser.parse_args()
    record = create_record(args.output)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
