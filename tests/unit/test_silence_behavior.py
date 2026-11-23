import unittest
from pathlib import Path

import numpy as np

from dronedet.config import load_config
from dronedet.features import transform


ROOT = Path(__file__).resolve().parents[2]


class SilenceNormalizationTests(unittest.TestCase):
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
