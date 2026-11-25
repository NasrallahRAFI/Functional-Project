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


if __name__ == "__main__":
    unittest.main()
