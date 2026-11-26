import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import soundfile as sf

from dronedet import features
from dronedet.config import load_config
from dronedet.features import transform
from dronedet.features import fit_cached_normalizer, global_stats
from dronedet.normalization import (GLOBAL_STATS_POLICY, Normalizer,
                                    mel_norm_equivalence_check)
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


class SilenceNormalizationTests(unittest.TestCase):
    def test_equivalence_diagnostic_converts_each_clip_arm_to_db_once(self):
        config = load_config(ROOT / "config.yaml")
        rng = np.random.default_rng(71)
        area = rng.uniform(0.01, 2.0, size=(config.n_mels, config.n_win + 3)).astype(np.float32)
        peak = area * np.linspace(0.7, 1.4, config.n_mels, dtype=np.float32)[:, None]
        with tempfile.TemporaryDirectory() as folder:
            raw_root = Path(folder)
            clip = "synthetic/clip.wav"
            path = raw_root / clip
            path.parent.mkdir(parents=True)
            sf.write(path, np.zeros(12000, dtype=np.float32), config.sr)

            def fake_file_to_power(_path, cfg):
                return (area if cfg.mel_norm == "area" else peak), config.sr

            with patch("dronedet.features.file_to_power", side_effect=fake_file_to_power), \
                 patch("dronedet.features.to_db", wraps=features.to_db) as db_conversion:
                result = mel_norm_equivalence_check(
                    [{"clip_id": clip, "split": "train", "stats_eligible": True,
                      "label_name": "synthetic"}], raw_root, config,
                    min_nonfloor_cells_per_band=1)
            self.assertIn(result["status"], {"pass", "stop_no_floor_residual_exceeds_threshold"})
            self.assertEqual(db_conversion.call_count, 2)

    def test_masked_fit_uses_stored_training_eligibility_and_per_band_cells(self):
        config = load_config(ROOT / "config.yaml")
        power = np.ones((config.n_mels, config.n_win + 1), dtype=np.float32)
        power[:, 0] = 0
        power[:, 1] = 0.01
        row = {"clip_id": "eligible", "split": "train", "stats_eligible": True,
               "silent": False, "near_silent": False, "crop_all_zero": False}
        excluded = {**row, "clip_id": "excluded", "stats_eligible": False,
                    "crop_all_zero": True}
        fitted = Normalizer.fit([row, excluded], config, {"eligible": power})
        # A whole floor frame is omitted. Each remaining cell contributes:
        # one at -20 dB and n_win-1 at 0 dB in each band.
        expected = np.concatenate((np.asarray([-20.0]), np.zeros(config.n_win - 1)))
        np.testing.assert_allclose(fitted.mean, expected.mean(), atol=1e-6)
        np.testing.assert_allclose(fitted.std, expected.std(), atol=1e-6)
        self.assertEqual(fitted.excluded_count, 1)
        self.assertEqual(fitted.fit_record["policy"], GLOBAL_STATS_POLICY)
        self.assertEqual(fitted.fit_record["retained_cells_per_band"], [config.n_win] * config.n_mels)

        partial = power.copy()
        partial[0, 2] = 0
        fitted_partial = Normalizer.fit([row], config, {"eligible": partial})
        self.assertEqual(fitted_partial.fit_record["masked_cells_per_band"][0], 2)
        self.assertEqual(fitted_partial.fit_record["masked_cells_per_band"][1], 1)

    def test_fit_rejects_missing_or_conflicting_eligibility_and_nontrain_rows(self):
        config = load_config(ROOT / "config.yaml")
        row = {"clip_id": "clip", "split": "train", "stats_eligible": True,
               "silent": False, "near_silent": False, "crop_all_zero": False}
        power = {"clip": np.ones((config.n_mels, config.n_win), np.float32)}
        for bad in ({**row, "split": "val"}, {**row, "split": "test"},
                    {**row, "stats_eligible": "False"}, {**row, "silent": True}):
            with self.subTest(row=bad), self.assertRaises(ValueError):
                Normalizer.fit([bad], config, power)
        missing = {key: value for key, value in row.items() if key != "stats_eligible"}
        with self.assertRaisesRegex(ValueError, "stored boolean"):
            Normalizer.fit([missing], config, power)

    def test_cache_adapter_requires_explicit_training_manifest_and_valid_spans(self):
        config = load_config(ROOT / "config.yaml")
        cache = np.ones((config.n_mels, config.n_win), np.float32)
        index = pd.DataFrame([{"clip_id": "clip", "start_frame": 0,
                               "n_frames": config.n_win, "stats_eligible": True}])
        with self.assertRaisesRegex(ValueError, "explicit training"):
            global_stats(cache, index, config)
        with self.assertRaisesRegex(ValueError, "train rows only"):
            fit_cached_normalizer(cache, index, [{"clip_id": "clip", "split": "val"}], config)
        wrong = index.copy()
        wrong.loc[0, "start_frame"] = -1
        with self.assertRaisesRegex(ValueError, "span"):
            fit_cached_normalizer(cache, wrong, [{"clip_id": "clip", "split": "train"}], config)

    def test_floor_silence_is_finite_and_strongly_negative(self):
        config = load_config(ROOT / "config.yaml")
        with np.load(ROOT / "data/processed/spectrograms/norm_stats_global.npz",
                     allow_pickle=False) as archive:
            stats = {"mean": archive["mean"], "std": archive["std"]}
        silence = np.zeros((config.n_mels, config.n_win), dtype=np.float32)
        normalized = transform(silence, config, stats)
        self.assertTrue(np.isfinite(normalized).all())
        self.assertLess(float(normalized.max()), -4.0)
        self.assertGreater(float(normalized.min()), -5.3)


class V4PreparationTests(unittest.TestCase):
    """Synthetic checks for V4 descriptors, shared flags, manifests and STOP behavior."""

    def test_periodogram_band_level_preserves_calibrated_sine_power(self):
        from dronedet.data_preparation import v4_descriptor
        t = np.arange(16000) / 16000
        x = (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        result = v4_descriptor(x)
        self.assertAlmostEqual(result["band_100_4000_dbfs"], 20 * np.log10(.1 / np.sqrt(2)), places=5)
        self.assertEqual(result["rolloff99_hz"], 1000.)
        self.assertAlmostEqual(result["rms_dbfs"], result["band_100_4000_dbfs"], places=5)

    def test_periodogram_does_not_remove_dc_and_zero_has_recorded_sentinel(self):
        from dronedet.data_preparation import v4_descriptor
        dc = v4_descriptor(np.full(16000, .25, dtype=np.float32))
        self.assertEqual(dc["rolloff99_hz"], 0.)
        self.assertAlmostEqual(dc["rms_dbfs"], 20 * np.log10(.25))
        zero = v4_descriptor(np.zeros(16000, np.float32))
        self.assertEqual((zero["rms_dbfs"], zero["band_100_4000_dbfs"], zero["rolloff99_hz"]), (-240., -240., 0.))

    def test_shared_crop_flag_uses_central_frames_and_exclusion_union(self):
        config = load_config(ROOT / "config.yaml")
        power = np.zeros((64, config.n_win + 4), np.float32)
        power[:, 0] = 1
        flags = features.power_quality_flags(power, config, silent=False, near_silent=False)
        self.assertEqual(flags["win_start_frame"], 2)
        self.assertTrue(flags["crop_all_zero"])
        self.assertFalse(flags["stats_eligible"])
        power[:, 2] = 1
        self.assertTrue(features.power_quality_flags(power, config, silent=False, near_silent=False)["stats_eligible"])
        self.assertFalse(features.power_quality_flags(power, config, silent=False, near_silent=True)["stats_eligible"])

    def test_float32_duplicate_hash_distinguishes_zero_lengths(self):
        from dronedet.data_preparation import v4_descriptor
        a = v4_descriptor(np.zeros(16000, np.float32))["pcm_sha256"]
        b = v4_descriptor(np.zeros(16384, np.float32))["pcm_sha256"]
        self.assertNotEqual(a, b)

    def test_source_only_audit_uses_pooled_source_majority(self):
        from dronedet.data_preparation import v4_source_only_audit
        frame = pd.DataFrame({"source": ["ae"] * 5 + ["dd"] * 5,
                              "label": [0, 0, 0, 1, 1, 0, 0, 1, 1, 1]})
        _, result = v4_source_only_audit(frame)
        self.assertEqual(result["majority"], {"ae": 0, "dd": 1})
        self.assertAlmostEqual(result["balanced_accuracy"], .6)
        self.assertFalse(result["is_model_evaluation"])

    def test_stratum_split_retains_whole_groups_and_is_deterministic(self):
        from dronedet.splits import two_stage_split
        records = []
        for source in ("ae", "dd"):
            for label in (0, 1):
                for group in range(30):
                    for segment in range(3):
                        records.append({"clip_id": f"{source}-{label}-{group:02}-{segment}",
                                        "group": f"{source}-{label}-{group:02}",
                                        "label": label, "stratum": f"{label}|{source}"})
        frame = pd.DataFrame(records)
        first = two_stage_split(frame, 42, label_column="stratum")
        pd.testing.assert_series_equal(first, two_stage_split(frame, 42, label_column="stratum"))
        frame["split"] = first
        self.assertTrue(frame.groupby("group").split.nunique().eq(1).all())
        self.assertEqual(set(frame.split), {"train", "val", "test"})

    def test_manifest_appends_v4_fields_without_dropping_legacy_schema(self):
        from dronedet.splits import MANIFEST_COLUMNS, write_manifest
        row = {name: 0 for name in MANIFEST_COLUMNS}
        row.update(clip_id="dd__example__0000000", source="dd", file="example.wav", pcm_sha256="a" * 64)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.csv"
            write_manifest(pd.DataFrame([row]), path, extra_columns=["source", "file", "pcm_sha256"])
            frame = pd.read_csv(path)
            self.assertEqual(list(frame.columns[:len(MANIFEST_COLUMNS)]), MANIFEST_COLUMNS)
            self.assertEqual(list(frame.columns[-3:]), ["source", "file", "pcm_sha256"])

    def test_failed_inventory_gate_records_stop_and_does_not_publish_splits(self):
        import hashlib
        import json
        import os
        import yaml
        from dronedet.data_preparation import run_v4_preparation, V4PreparationStop
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            checkout = base / "workspaces/study-v4"
            checkout.mkdir(parents=True)
            config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
            (checkout / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
            plan = checkout / "docs/v4/DATA_PREPARATION_CNN_V4_0.md"
            plan.parent.mkdir(parents=True)
            plan.write_text("synthetic plan", encoding="utf-8")
            for relative in ("src/dronedet/data_preparation.py", "src/dronedet/features.py", "src/dronedet/splits.py", "src/dronedet/resample.py"):
                p = checkout / relative
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("# synthetic source", encoding="utf-8")
            original = base / "storage/sources/al-emadi/al-emadi-r001/original/Binary_Drone_Audio/unknown/silence001.wav"
            original.parent.mkdir(parents=True)
            sf.write(original, np.zeros(16000, np.float32), 16000, subtype="PCM_16")
            before = hashlib.sha256(original.read_bytes()).hexdigest()
            with patch("subprocess.check_output", return_value=b"synthetic-commit\n"), \
                    patch("dronedet.data_preparation.__file__", str(checkout / "src/dronedet/data_preparation.py")), \
                    patch.dict(os.environ, {"USERPROFILE": folder}):
                with self.assertRaisesRegex(V4PreparationStop, "A2_COUNTS"):
                    run_v4_preparation(checkout / "config.yaml")
            status = json.loads((base / "storage/corpora/corpus-002/audits/status.json").read_text())
            self.assertEqual(status["status"], "STOP")
            self.assertFalse((base / "storage/splits/split-002/train.csv").exists())
            self.assertEqual(before, hashlib.sha256(original.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
