import unittest
import json
import tempfile
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from dronedet.bundle import (BundleVerificationError, run_bundle_rejection_checks,
                             verify_bundle, write_bundle)
from dronedet.config import load_config


ROOT = Path(__file__).resolve().parents[2]


class BundleTamperContractTests(unittest.TestCase):
    def test_re_stamped_bundle_rejects_tampering(self):
        bundle = ROOT / "data/processed/golden/frontend_bundle"
        result = run_bundle_rejection_checks(bundle, load_config(ROOT / "config.yaml"))
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(value.get("pass", True) for value in result.values()
                            if isinstance(value, dict)))

    @staticmethod
    def _write_native_bundle(root: Path, *, stereo: bool = False):
        config = load_config(ROOT / "config.yaml")
        if stereo:
            config = replace(config, channel=1)
        with np.load(ROOT / "data/processed/spectrograms/norm_stats_global.npz",
                     allow_pickle=False) as archive:
            stats = {"mean": archive["mean"].copy(), "std": archive["std"].copy()}
        t16 = np.arange(8000, dtype=np.float32) / 16000
        t48 = np.arange(24000, dtype=np.float32) / 48000
        audio = {"golden16": (0.2 * np.sin(2 * np.pi * 440 * t16)).astype(np.float32),
                 "golden48": (0.2 * np.sin(2 * np.pi * 440 * t48)).astype(np.float32)}
        if stereo:
            audio = {name: np.column_stack((value, np.roll(value, 7))).astype(np.float32)
                     for name, value in audio.items()}
        bundle = root / "bundle"
        write_bundle(bundle, config, stats, audio)
        return bundle, config

    def test_native_rate_wavs_verify_resampling_and_normalized_goldens(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle, config = self._write_native_bundle(Path(folder))
            result = verify_bundle(bundle, config)
            self.assertEqual(result["golden_source_rates"]["golden16"]["source_rate_hz"], 16000)
            self.assertFalse(result["golden_source_rates"]["golden16"]["resampled"])
            self.assertEqual(result["golden_source_rates"]["golden48"]["source_rate_hz"], 48000)
            self.assertTrue(result["golden_source_rates"]["golden48"]["resampled"])
            for value in result["normalized_outputs"].values():
                self.assertTrue(value["verified"])
                self.assertEqual(value["shape"], [1, 64, 28])

    def test_native_stereo_fixture_uses_the_configured_channel(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle, config = self._write_native_bundle(Path(folder), stereo=True)
            result = verify_bundle(bundle, config)
            self.assertTrue(result["golden_source_rates"]["golden48"]["resampled"])
            self.assertTrue(all(value["verified"] for value in result["normalized_outputs"].values()))

    def test_nan_golden_and_invalid_statistics_are_rejected_specifically(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle, config = self._write_native_bundle(Path(folder))
            golden = bundle / "golden/golden16_db.npy"
            values = np.load(golden, allow_pickle=False)
            values[0, 0] = np.nan
            np.save(golden, values, allow_pickle=False)
            with self.assertRaisesRegex(BundleVerificationError, "NaN or infinity") as error:
                verify_bundle(bundle, config)
            self.assertEqual(error.exception.reason_code, "GOLDEN_NONFINITE")
        with tempfile.TemporaryDirectory() as folder:
            bundle, config = self._write_native_bundle(Path(folder))
            with np.load(bundle / "stats.npz", allow_pickle=False) as archive:
                mean, std = archive["mean"], archive["std"]
            std[0] = 0
            np.savez(bundle / "stats.npz", mean=mean, std=std)
            metadata = json.loads((bundle / "stats.json").read_text(encoding="utf-8"))
            metadata["std"] = std.tolist()
            (bundle / "stats.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(BundleVerificationError, "strictly positive") as error:
                verify_bundle(bundle, config)
            self.assertEqual(error.exception.reason_code, "STATS_STD_INVALID")

    def test_48khz_goldens_fail_when_the_loader_resampler_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle, config = self._write_native_bundle(Path(folder))
            from dronedet import features
            original = features.resample

            def changed(audio, source_rate, target_rate, *, quality="HQ"):
                result = original(audio, source_rate, target_rate, quality=quality)
                return result * np.float32(0.5) if source_rate == 48000 else result

            with patch("dronedet.features.resample", side_effect=changed):
                with self.assertRaisesRegex(BundleVerificationError, "golden48") as error:
                    verify_bundle(bundle, config)
            self.assertEqual(error.exception.reason_code, "GOLDEN_MISMATCH")

    def test_native_fixture_rate_metadata_must_match_filename_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle, config = self._write_native_bundle(Path(folder))
            audio, _ = sf.read(bundle / "golden/golden48_audio.wav", dtype="float32")
            sf.write(bundle / "golden/golden48_audio.wav", audio, 44100, subtype="FLOAT")
            with self.assertRaisesRegex(BundleVerificationError, "expected 48000") as error:
                verify_bundle(bundle, config)
            self.assertEqual(error.exception.reason_code, "GOLDEN_RATE_INVALID")


if __name__ == "__main__":
    unittest.main()
