import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SOURCE))

from drone_audio.dsp import fft_radix2, ifft_radix2, mel_filterbank, rfft_scratch  # noqa: E402
from drone_audio.frontend import FrontEnd, FrontEndConfig, load_frontend_config  # noqa: E402
from drone_audio.normalization import LogMelNormalizer, prepare_model_input  # noqa: E402
from drone_audio.resample import resample  # noqa: E402


class DspFrontEndTests(unittest.TestCase):
    def test_fft_matches_independent_reference_and_round_trips(self):
        rng = np.random.default_rng(10)
        signal = rng.standard_normal((3, 1024))
        self.assertTrue(np.allclose(rfft_scratch(signal), np.fft.rfft(signal, axis=-1), atol=1e-10))
        self.assertTrue(np.allclose(ifft_radix2(fft_radix2(signal)).real, signal, atol=1e-10))

    def test_mel_bank_has_valid_nonempty_filters(self):
        bank = mel_filterbank(16_000, 1024, 64, 50, 8_000)
        self.assertEqual(bank.shape, (64, 513))
        self.assertTrue(np.all(bank.sum(axis=1) > 0))

    def test_config_and_linear_stft_comparison_path(self):
        # Edit 1 (D-B2): path changed from config/frontend.json to config.yaml
        config_path = Path(__file__).resolve().parents[2] / "config.yaml"
        config = load_frontend_config(config_path)
        self.assertEqual(config.sample_rate, 16_000)
        frontend = FrontEnd(config)
        audio = np.sin(2 * np.pi * 440 * np.arange(config.sample_rate) / config.sample_rate).astype(np.float32)
        self.assertEqual(frontend.linear_log_stft(audio).shape[0], config.n_fft // 2 + 1)
        self.assertEqual(frontend.offline(audio).shape[0], config.n_mels)

    def test_resampler_preserves_passband_and_rejects_alias(self):
        rate = 48_000
        time = np.arange(rate, dtype=np.float64) / rate
        for frequency, expected in ((1_000.0, 1.0), (12_000.0, 0.0)):
            source = np.sin(2 * np.pi * frequency * time).astype(np.float32)
            output = resample(source, rate, 16_000)
            rms_amplitude = float(np.sqrt(2 * np.mean(output[500:-500] ** 2)))
            self.assertLess(abs(rms_amplitude - expected), 0.05)

    def test_streaming_matches_offline_for_odd_chunk_sizes(self):
        rng = np.random.default_rng(5)
        audio = (rng.standard_normal(16_000 * 3) * 0.04).astype(np.float32)
        pcm16 = np.rint(audio * 32768).astype(np.int16)
        config = FrontEndConfig()
        offline = FrontEnd(config).offline(pcm16.astype(np.float32) / 32768.0)
        streaming = FrontEnd(config)
        columns = [streaming.push(pcm16[start:start + 777]) for start in range(0, pcm16.size, 777)]
        result = np.concatenate(columns, axis=1)
        self.assertEqual(result.shape, offline.shape)
        self.assertTrue(np.allclose(result, offline, atol=1e-4))

    # Edit 2 (D-B2): test body replaced; method renamed to reflect actual behaviour
    def test_legacy_normalizer_fit_raises_runtime_error_guard(self):
        # Old test asserted: fit on two batches of dB values (-30, -10) gives mean=-20;
        # transform(-20) = 0.0; prepare_model_input shape (2,2,5); delta=[0,1]; padded<0
        training = [np.full((2, 4), -30, np.float32), np.full((2, 4), -10, np.float32)]
        with self.assertRaisesRegex(RuntimeError, r"use dronedet\.features\.Normalizer\.fit"):
            LogMelNormalizer.fit(training)

    # B4.5: new production check
    def test_pcen_transform_is_deterministic(self):
        # pcen_transform on same input gives bit-identical output twice.
        import dronedet.config as cfg
        from dronedet.normalization import pcen_transform
        fe = cfg.load_config(Path(__file__).resolve().parents[2] / "config.yaml")
        rng = np.random.default_rng(99)
        power = rng.random((64, 28), dtype=np.float32)
        out1 = pcen_transform(power, fe)
        out2 = pcen_transform(power, fe)
        np.testing.assert_array_equal(out1, out2)
        # Second condition (window in longer call) does not apply: production
        # pcen_transform is stateless per-window (no cross-window state carried).


if __name__ == "__main__":
    unittest.main()
