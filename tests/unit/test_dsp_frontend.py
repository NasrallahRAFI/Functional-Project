import queue
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[2] / "src"
PROJECT = SOURCE.parent
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(PROJECT))

from demo.run_demo import (
    CaptureCallback,
    CaptureEvent,
    LiveCaptureConsumer,
    LiveWindowBuilder,
    Ring,
    SessionSequenceGuard,
    VoteSmoother,
    build_parser,
    check_stream_parity,
    live_mode,
    validate_args,
)
from drone_audio.dsp import (
    fft_radix2,
    ifft_radix2,
    mel_filterbank,
    rfft_scratch,
)
from drone_audio.frontend import (
    FrontEnd,
    FrontEndConfig,
    load_frontend_config,
)
from drone_audio.normalization import (
    LogMelNormalizer,
)
from drone_audio.resample import resample
from dronedet.config import load_config
from dronedet.features import centre_window, random_window, standardize
from dronedet.legacy_features import cache_split_power as legacy_cache_split_power
from dronedet.legacy_features import centred_window as legacy_centred_window
from dronedet.legacy_features import prepare_waveform_window, random_training_window
from dronedet.legacy_features import (
    standardize_audio as legacy_standardize_audio,
)
from scripts.live_detector import LiveState


class DspFrontEndTests(unittest.TestCase):
    def test_legacy_window_helpers_share_the_canonical_selection_logic(self):
        canonical = load_config(PROJECT / "config.yaml")
        legacy = load_frontend_config(PROJECT / "config.yaml")
        power = np.arange(canonical.n_mels * (canonical.n_win + 9), dtype=np.float32)
        power = power.reshape(canonical.n_mels, canonical.n_win + 9)
        self.assertTrue(
            np.array_equal(
                legacy_centred_window(power, legacy.n_win),
                centre_window(power, canonical),
            )
        )
        self.assertTrue(
            np.array_equal(
                random_training_window(power, legacy.n_win, np.random.default_rng(123)),
                random_window(power, canonical, np.random.default_rng(123)),
            )
        )

    def test_legacy_standardization_adapts_to_canonical_channel_and_resampler(self):
        canonical = load_config(PROJECT / "config.yaml")
        legacy = load_frontend_config(PROJECT / "config.yaml")
        source = np.zeros((24_000, 2), dtype=np.float32)
        source[:, 0] = np.sin(np.arange(source.shape[0], dtype=np.float32) * 0.01)
        source[:1200, 0] = 1.0
        source[:, 1] = 0.25
        canonical_audio = standardize(source, 48_000, canonical)
        expected, _ = prepare_waveform_window(canonical_audio, legacy)
        actual, flags = legacy_standardize_audio(source, 48_000, legacy)
        self.assertTrue(np.array_equal(actual, expected))
        self.assertEqual(flags, FrontEnd.quality_flags(source[:, 0]))

    def test_short_legacy_windows_keep_their_historical_half_window_cutoff(self):
        config = replace(
            load_frontend_config(PROJECT / "config.yaml"), clip_length_s=1024 / 16_000
        )
        audio = np.arange(600, dtype=np.float32)
        result, silent = prepare_waveform_window(audio, config)
        self.assertFalse(silent)
        self.assertEqual(config.win_samples, 1024)
        self.assertTrue(np.array_equal(result[:600], audio))
        self.assertEqual(result.size, config.win_samples)

    def test_legacy_cache_writer_fails_closed_with_canonical_replacement(self):
        frontend = FrontEnd(load_frontend_config(PROJECT / "config.yaml"))
        with (
            pytest.warns(
                DeprecationWarning, match="dronedet.features.cache_split_power"
            ),
            self.assertRaisesRegex(RuntimeError, "legacy cache format is retired"),
        ):
            legacy_cache_split_power(
                [], split="train", output_dir=PROJECT / "tmp", frontend=frontend
            )

    def test_fixed_ring_preserves_chronology_through_wraparound_and_reset(self):
        ring = Ring(5)
        self.assertFalse(ring.push(np.array([0, 1, 2], dtype=np.float32)))
        self.assertTrue(np.array_equal(ring.data, [0, 1, 2]))
        self.assertTrue(ring.push(np.array([3, 4, 5], dtype=np.float32)))
        self.assertTrue(np.array_equal(ring.data, [1, 2, 3, 4, 5]))
        ring.push(np.array([6, 7], dtype=np.float32))
        self.assertTrue(np.array_equal(ring.data, [3, 4, 5, 6, 7]))
        ring.reset()
        self.assertEqual(ring.data.size, 0)
        self.assertFalse(ring.push(np.array([8, 9], dtype=np.float32)))
        self.assertTrue(np.array_equal(ring.data, [8, 9]))

    def test_vote_smoother_thresholds_raw_scores_and_requires_a_full_majority(self):
        smoother = VoteSmoother(3)
        self.assertIsNone(smoother.push(0.51, 0.5))
        self.assertIsNone(smoother.push(0.51, 0.5))
        self.assertTrue(smoother.push(0.0, 0.5))
        smoother.reset()
        for score in (0.51, 0.51):
            self.assertIsNone(smoother.push(score, 0.6))
        self.assertFalse(smoother.push(0.0, 0.6))
        smoother.reset()
        for score in (2.75, 2.8):
            self.assertIsNone(smoother.push(score, 2.75))
        self.assertTrue(smoother.push(2.4, 2.75))

    def test_live_decision_cadence_is_quantized_to_the_nearest_hop(self):
        config = load_config(Path(__file__).resolve().parents[2] / "config.yaml")
        builder = LiveWindowBuilder(config, config.sr)
        audio = np.zeros(config.sr * 4, dtype=np.float32)
        for start in range(0, len(audio), 1000):
            builder.push(audio[start : start + 1000])
        builder.finish()
        cadence_hops = round(config.decision_interval_s * config.sr / config.hop)
        cadence_samples = cadence_hops * config.hop
        self.assertGreaterEqual(len(builder.emitted_starts), 3)
        self.assertEqual(
            np.diff(builder.emitted_starts).tolist(),
            [cadence_samples] * (len(builder.emitted_starts) - 1),
        )
        cadence_error_s = cadence_samples / config.sr - config.decision_interval_s
        self.assertAlmostEqual(cadence_error_s, -0.004, places=6)

    def test_actual_callback_queue_drops_start_a_fresh_contiguous_generation(self):
        config = load_config(Path(__file__).resolve().parents[2] / "config.yaml")
        audio_queue = queue.Queue(maxsize=2)
        callback = CaptureCallback(audio_queue)
        consumer = LiveCaptureConsumer(config, config.sr)
        windows = []

        def ramp(block_number):
            start = block_number * 240
            return (np.arange(240, dtype=np.float32) + start)[:, None]

        def deliver(block_number):
            callback(ramp(block_number), 240, None, None)
            windows.extend(consumer.process(audio_queue.get_nowait()))

        for number in range(40):
            deliver(number)
        callback(ramp(40), 240, None, None)
        callback(ramp(41), 240, None, None)
        callback(ramp(42), 240, None, None)
        callback(ramp(43), 240, None, None)
        self.assertEqual(callback.counters.software_drops, 2)
        self.assertEqual(callback.generation, 2)
        windows.extend(consumer.process(audio_queue.get_nowait()))
        windows.extend(consumer.process(audio_queue.get_nowait()))
        for number in range(44, 80):
            deliver(number)

        self.assertGreaterEqual(len(windows), 2)
        self.assertEqual(consumer.recoveries, 1)
        for window in windows:
            np.testing.assert_array_equal(
                np.diff(window), np.ones(len(window) - 1, np.float32)
            )
        stale = CaptureEvent(0, np.zeros(240, dtype=np.float32))
        self.assertEqual(consumer.process(stale), [])
        self.assertEqual(consumer.discarded_stale_events, 1)

    def test_callback_hardware_overflow_and_invalid_samples_emit_reset_markers(self):
        class Status:
            input_overflow = True

            def __str__(self):
                return "input overflow"

        config = load_config(Path(__file__).resolve().parents[2] / "config.yaml")
        audio_queue = queue.Queue(maxsize=4)
        callback = CaptureCallback(audio_queue)
        consumer = LiveCaptureConsumer(config, config.sr)
        callback(np.ones((240, 1), np.float32), 240, None, None)
        consumer.process(audio_queue.get_nowait())
        callback(np.ones((240, 1), np.float32), 240, None, Status())
        event = audio_queue.get_nowait()
        self.assertEqual(event.reason, "input_overflow")
        self.assertIsNone(event.samples)
        self.assertEqual(consumer.process(event), [])
        self.assertEqual(callback.counters.hardware_overflows, 1)
        invalid = np.ones((240, 1), np.float32)
        invalid[12, 0] = np.nan
        callback(invalid, 240, None, None)
        event = audio_queue.get_nowait()
        self.assertEqual(event.reason, "invalid_audio")
        self.assertEqual(consumer.process(event), [])
        self.assertEqual(callback.counters.invalid_blocks, 1)

    def test_one_hundred_seeded_queue_overflow_bursts_recover_contiguously(self):
        config = load_config(Path(__file__).resolve().parents[2] / "config.yaml")
        rng = np.random.default_rng(881)
        for _ in range(100):
            drop_count = int(rng.integers(1, 5))
            audio_queue = queue.Queue(maxsize=1)
            callback = CaptureCallback(audio_queue)
            consumer = LiveCaptureConsumer(config, config.sr)

            def block(number):
                start = number * 240
                return (np.arange(240, dtype=np.float32) + start)[:, None]

            callback(block(0), 240, None, None)
            for number in range(1, drop_count + 1):
                callback(block(number), 240, None, None)
            consumer.process(audio_queue.get_nowait())
            post = drop_count + 1
            callback(block(post), 240, None, None)
            windows = consumer.process(audio_queue.get_nowait())
            for number in range(post + 1, post + 37):
                callback(block(number), 240, None, None)
                windows.extend(consumer.process(audio_queue.get_nowait()))
            self.assertEqual(callback.counters.software_drops, drop_count)
            self.assertEqual(consumer.recoveries, 1)
            self.assertTrue(windows)
            for window in windows:
                np.testing.assert_array_equal(
                    np.diff(window), np.ones(len(window) - 1, np.float32)
                )

    def test_streaming_parity_contract_still_passes_at_both_rates(self):
        config = load_config(Path(__file__).resolve().parents[2] / "config.yaml")
        result = check_stream_parity(config, seed=73, duration_s=1.5)
        self.assertEqual(result["status"], "PASS", result)

    def test_browser_sequence_guard_resets_on_gaps_sessions_and_rate_changes(self):
        guard = SessionSequenceGuard()
        changed, reasons = guard.observe("session-a", 48000, 0)
        self.assertTrue(changed)
        self.assertEqual(
            set(reasons), {"session_changed", "sample_rate_changed", "sequence_gap"}
        )
        changed, reasons = guard.observe("session-a", 48000, 1)
        self.assertFalse(changed)
        self.assertEqual(reasons, [])
        changed, reasons = guard.observe("session-a", 48000, 3)
        self.assertTrue(changed)
        self.assertEqual(reasons, ["sequence_gap"])
        changed, reasons = guard.observe("session-b", 44100, 0)
        self.assertTrue(changed)
        self.assertEqual(
            set(reasons), {"session_changed", "sample_rate_changed", "sequence_gap"}
        )

    def test_browser_live_state_buffers_three_votes_and_resets_on_sequence_gap(self):
        config = load_config(PROJECT / "config.yaml")

        class FixedModel:
            def __call__(self, tensor):
                import torch

                return torch.tensor([[2.9]], dtype=torch.float32)

        state = LiveState(
            config,
            {"mean": np.zeros(64, np.float32), "std": np.ones(64, np.float32)},
            FixedModel(),
            threshold=2.75,
        )
        rng = np.random.default_rng(390)
        chunks = [rng.normal(0, 0.1, 8000).astype(np.float32) for _ in range(3)]
        outputs = [
            state.process(chunk, 16000, "session-one", sequence)
            for sequence, chunk in enumerate(chunks)
        ]
        self.assertEqual(
            [item["status"] for item in outputs], ["warming_up", "warming_up", "scored"]
        )
        self.assertTrue(outputs[-1]["drone"])
        self.assertAlmostEqual(outputs[-1]["score"], 2.9, places=6)
        restarted = state.process(chunks[0], 16000, "session-two", 0)
        self.assertEqual(restarted["status"], "warming_up")
        self.assertTrue(restarted["recovered"])
        after_gap = state.process(chunks[1], 16000, "session-two", 2)
        self.assertEqual(after_gap["status"], "warming_up")
        self.assertEqual(after_gap["recovery_reasons"], ["sequence_gap"])

    def test_live_hardware_protocol_command_parses_without_opening_audio(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--live",
                "--scorer",
                "stub",
                "--minutes",
                "5",
                "--debug-stall-at",
                "60,180",
                "--debug-stall-seconds",
                "2",
                "--log-json",
                "data/interim/session.json",
            ]
        )
        self.assertEqual(validate_args(args, parser), (60.0, 180.0))
        bad = parser.parse_args(
            ["--live", "--minutes", "5", "--debug-stall-at", "180,60"]
        )
        with self.assertRaises(SystemExit):
            validate_args(bad, parser)

    def test_debug_stall_runs_consumer_pause_against_the_real_capture_queue(self):
        config = load_config(PROJECT / "config.yaml")

        class FakeStream:
            def __init__(self, **kwargs):
                self.callback = kwargs["callback"]
                self.blocksize = 240
                self.stop_event = threading.Event()
                self.index = 0

            def __enter__(self):
                def produce():
                    while not self.stop_event.is_set():
                        start = self.index * 240
                        block = (
                            np.arange(start, start + 240, dtype=np.float32) / 100000
                        )[:, None]
                        self.callback(block, 240, None, None)
                        self.index += 1
                        time.sleep(0.015)

                self.thread = threading.Thread(target=produce, daemon=True)
                self.thread.start()
                return self

            def __exit__(self, *args):
                self.stop_event.set()
                self.thread.join(timeout=1)

        fake_audio = type(
            "FakeAudio",
            (),
            {
                "query_devices": staticmethod(
                    lambda **kwargs: {
                        "default_samplerate": 16000,
                        "hostapi": 0,
                        "name": "synthetic",
                    }
                ),
                "query_hostapis": staticmethod(
                    lambda index: {"name": "synthetic-host"}
                ),
                "InputStream": FakeStream,
            },
        )
        with tempfile.TemporaryDirectory() as folder:
            log_path = Path(folder) / "session.json"
            with patch.dict(sys.modules, {"sounddevice": fake_audio}):
                record = live_mode(
                    config,
                    minutes=0.03,
                    debug_stall_at=(0,),
                    debug_stall_seconds=0.6,
                    log_json=log_path,
                )
            self.assertEqual(record["run_status"], "completed_by_duration")
            self.assertIn(0, record["debug_stalls_triggered_seconds"])
            self.assertGreater(record["software_drop_blocks"], 0)
            self.assertGreater(record["recoveries"], 0)
            self.assertTrue(log_path.is_file())

    def test_fft_matches_independent_reference_and_round_trips(self):
        rng = np.random.default_rng(10)
        signal = rng.standard_normal((3, 1024))
        self.assertTrue(
            np.allclose(rfft_scratch(signal), np.fft.rfft(signal, axis=-1), atol=1e-10)
        )
        self.assertTrue(
            np.allclose(ifft_radix2(fft_radix2(signal)).real, signal, atol=1e-10)
        )

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
        audio = np.sin(
            2 * np.pi * 440 * np.arange(config.sample_rate) / config.sample_rate
        ).astype(np.float32)
        self.assertEqual(
            frontend.linear_log_stft(audio).shape[0], config.n_fft // 2 + 1
        )
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
        columns = [
            streaming.push(pcm16[start : start + 777])
            for start in range(0, pcm16.size, 777)
        ]
        result = np.concatenate(columns, axis=1)
        self.assertEqual(result.shape, offline.shape)
        self.assertTrue(np.allclose(result, offline, atol=1e-4))

    # Edit 2 (D-B2): test body replaced; method renamed to reflect actual behaviour
    def test_legacy_normalizer_fit_raises_runtime_error_guard(self):
        # Old test asserted: fit on two batches of dB values (-30, -10) gives mean=-20;
        # transform(-20) = 0.0; prepare_model_input shape (2,2,5); delta=[0,1]; padded<0
        training = [np.full((2, 4), -30, np.float32), np.full((2, 4), -10, np.float32)]
        with self.assertRaisesRegex(
            RuntimeError, r"use dronedet\.features\.Normalizer\.fit"
        ):
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
