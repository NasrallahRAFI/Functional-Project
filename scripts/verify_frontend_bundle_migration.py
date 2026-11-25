"""Verify the immutable V1 bundle, then build and challenge a temporary V2 bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dronedet.bundle import BundleVerificationError, verify_bundle, write_bundle
from dronedet.cache_metadata import file_sha256
from dronedet.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-output", type=Path,
                        default=ROOT / "data/interim/frontend_bundle_v2_migration")
    parser.add_argument("--record-output", type=Path,
                        default=ROOT / "data/interim/frontend_bundle_v2_migration.json")
    args = parser.parse_args()
    if args.bundle_output.exists() or args.record_output.exists():
        raise FileExistsError("refusing to replace a previous bundle migration artifact")

    config = load_config(ROOT / "config.yaml")
    installed = ROOT / "data/processed/golden/frontend_bundle"
    legacy_result = verify_bundle(installed, config)
    fixture_paths = {"golden16": ROOT / "data/processed/golden/golden_16000.wav",
                     "golden48": ROOT / "data/processed/golden/golden_48000.wav"}
    audio = {}
    fixture_metadata = {}
    for name, path in fixture_paths.items():
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        expected_rate = 16000 if name == "golden16" else 48000
        if rate != expected_rate:
            raise RuntimeError(f"{path.name}: expected native {expected_rate} Hz, got {rate}")
        audio[name] = samples[:, 0] if config.channel == 0 else samples
        fixture_metadata[name] = {"source_sha256": file_sha256(path), "source_rate_hz": rate,
                                  "frames": int(samples.shape[0]), "channels": int(samples.shape[1])}
    with np.load(ROOT / "data/processed/spectrograms/norm_stats_global.npz",
                 allow_pickle=False) as archive:
        stats = {"mean": archive["mean"].copy(), "std": archive["std"].copy()}
    write_bundle(args.bundle_output, config, stats, audio)
    migrated_result = verify_bundle(args.bundle_output, config)

    from dronedet import features
    original_resampler = features.resample

    def altered_resampler(values, source_rate, target_rate, *, quality="HQ"):
        result = original_resampler(values, source_rate, target_rate, quality=quality)
        return result * np.float32(0.5) if source_rate == 48000 else result

    try:
        with patch("dronedet.features.resample", side_effect=altered_resampler):
            verify_bundle(args.bundle_output, config)
    except BundleVerificationError as exc:
        altered_result = {"rejected": True, "reason_code": exc.reason_code,
                          "message": str(exc), "pass": exc.reason_code == "GOLDEN_MISMATCH"
                          and "golden48" in str(exc)}
    else:
        altered_result = {"rejected": False, "pass": False}

    record = {"status": "PASS" if altered_result["pass"] else "FAIL",
              "test_audio_or_array_payloads_opened": False,
              "installed_v1_bundle": {"path": str(installed.relative_to(ROOT)),
                                      "result": legacy_result},
              "v2_bundle": {"path": str(args.bundle_output.relative_to(ROOT)),
                            "result": migrated_result,
                            "fixtures": fixture_metadata,
                            "golden_db_tolerance": 1e-3,
                            "normalized_tensor_tolerance": 1e-6,
                            "altered_48khz_resampler": altered_result},
              "statistics_sha256": file_sha256(ROOT / "data/processed/spectrograms/norm_stats_global.npz"),
              "bundle_was_written_outside_processed_data": args.bundle_output.resolve().is_relative_to(
                  (ROOT / "data/interim").resolve())}
    args.record_output.parent.mkdir(parents=True, exist_ok=True)
    args.record_output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    if record["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
