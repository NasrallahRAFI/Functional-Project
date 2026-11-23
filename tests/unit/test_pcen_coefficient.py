import unittest
from pathlib import Path

import librosa
import numpy as np

from dronedet.config import load_config
from dronedet.normalization import pcen_transform


ROOT = Path(__file__).resolve().parents[2]


class PCENCoefficientTests(unittest.TestCase):
    def test_window_mean_zi_uses_the_librosa_time_constant_coefficient(self):
        config = load_config(ROOT / "config.yaml")
        power = np.random.default_rng(314159).random((config.n_mels, 59), dtype=np.float32)
        magnitude = np.sqrt(power) * np.float32(config.pcen_input_scale)
        T = config.pcen_time_constant * config.sr / config.hop
        b = (np.sqrt(1.0 + 4.0 * T**2) - 1.0) / (2.0 * T**2)
        zi = ((1.0 - b) * magnitude.mean(axis=-1, keepdims=True)).astype(np.float32)
        expected = librosa.pcen(magnitude, sr=config.sr, hop_length=config.hop,
                                time_constant=config.pcen_time_constant, b=b,
                                gain=config.pcen_gain, bias=config.pcen_bias,
                                power=config.pcen_power, eps=config.pcen_eps,
                                zi=zi, axis=-1)
        actual = pcen_transform(power, config)
        self.assertTrue(np.allclose(actual, expected, rtol=0.0, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
