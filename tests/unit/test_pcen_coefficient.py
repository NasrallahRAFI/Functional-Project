import unittest
import json
import tempfile
from unittest.mock import patch
from pathlib import Path

import librosa
import numpy as np

from dronedet.config import load_config
from dronedet.normalization import pcen_transform
from dronedet.data_preparation import measure_resampler_edge, record_resampler_measurement


ROOT = Path(__file__).resolve().parents[2]


class PCENCoefficientTests(unittest.TestCase):
    def test_passband_measurement_uses_both_rates_and_the_lower_swept_edge(self):
        def measured(source_rate, target_rate, **kwargs):
            edge = 7540.0 if source_rate == 48000 else 7530.0
            return edge, np.asarray([1000, 7530]), np.asarray([0.0, -1.0])

        with patch("dronedet.resample.measure_swept_passband", side_effect=measured) as call:
            result = measure_resampler_edge()
        self.assertEqual({args.args[0] for args in call.call_args_list}, {44100, 48000})
        self.assertEqual(result["f1db"], 7530.0)
        self.assertEqual(result["fmax"], 7500.0)
        self.assertEqual(result["method"], "log_swept_chirp_instantaneous_frequency_v1")

    def test_frozen_passband_mismatch_is_reported_without_rewriting_config(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            config_path = folder / "config.yaml"
            config_path.write_bytes((ROOT / "config.yaml").read_bytes())
            before = config_path.read_bytes()
            marker = folder / "data/processed/splits/FROZEN.txt"
            marker.parent.mkdir(parents=True)
            marker.write_text("frozen")
            record = folder / "measurement.json"
            result = record_resampler_measurement(config_path, record,
                        measurement={"f1db": 7550.0, "fmax": 7500.0})
            self.assertEqual(result["status"], "MISMATCH_FROZEN_CONFIG")
            self.assertFalse(result["configuration_updated"])
            self.assertEqual(config_path.read_bytes(), before)
            self.assertEqual(json.loads(record.read_text())["f1db"], 7550.0)

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
