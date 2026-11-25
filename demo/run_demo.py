"""Front-end streaming demo with a deliberately stubbed detector."""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dronedet.config import FrontEndConfig, load_config
from dronedet.features import clip_flags, load_audio, mel_power, to_db, transform
from dronedet.resample import resample, resample_stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("dronedet.demo")


class StubScorer:
    def score(self, feature: np.ndarray) -> float:
        return float(np.clip(np.mean(np.abs(feature)), 0, 1))


def _stats_for_stub(
    config: FrontEndConfig, stats: dict[str, np.ndarray] | None
) -> dict[str, np.ndarray]:
    # A neutral transform keeps the non-model demo executable before train stats exist.
    return (
        stats
        if stats is not None
        else {
            "mean": np.zeros(config.n_mels, np.float32),
            "std": np.ones(config.n_mels, np.float32),
        }
    )


class VoteSmoother:
    """Threshold each raw score, then require a strict majority over a full window."""

    def __init__(self, count: int):
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("smoothing window count must be a positive integer")
        self.count = count
        self.values: list[bool] = []

    def reset(self) -> None:
        self.values = []

    def push(self, score: float, threshold: float) -> bool | None:
        if not np.isfinite(score) or not np.isfinite(threshold):
            raise ValueError("score and decision threshold must be finite")
        self.values = (self.values + [bool(score >= threshold)])[-self.count :]
        if len(self.values) < self.count:
            return None
        return sum(self.values) >= self.count // 2 + 1


class Ring:
    """Fixed-capacity float32 ring; returned data is chronological and owned."""

    def __init__(self, size: int):
        if not isinstance(size, int) or size <= 0:
            raise ValueError("ring size must be a positive integer")
        self.size = size
        self._buffer = np.empty(size, dtype=np.float32)
        self._count = 0
        self._write = 0

    @property
    def data(self) -> np.ndarray:
        """Return current samples oldest-first as a detached array."""
        if self._count == 0:
            return np.empty(0, dtype=np.float32)
        start = (self._write - self._count) % self.size
        stop = start + self._count
        if stop <= self.size:
            return self._buffer[start:stop].copy()
        return np.concatenate((self._buffer[start:], self._buffer[: stop - self.size]))

    def push(self, x: np.ndarray) -> bool:
        values = np.asarray(x, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError("ring input must be one-dimensional")
        if values.size >= self.size:
            self._buffer[:] = values[-self.size :]
            self._count = self.size
            self._write = 0
            return True
        first = min(values.size, self.size - self._write)
        self._buffer[self._write : self._write + first] = values[:first]
        remaining = values.size - first
        if remaining:
            self._buffer[:remaining] = values[first:]
        self._write = (self._write + values.size) % self.size
        self._count = min(self.size, self._count + values.size)
        return self._count == self.size

    def reset(self) -> None:
        self._count = 0
        self._write = 0


class LiveWindowBuilder:
    """Streaming resampler/ring that suppresses output after a discontinuity."""

    def __init__(self, config: FrontEndConfig, source_rate: int) -> None:
        self.config = config
        self.source_rate = source_rate
        self.reset()

    def reset(self) -> None:
        """Start a fresh stream generation and discard buffered audio."""
        self.converter = (
            None
            if self.source_rate == self.config.sr
            else resample_stream(
                self.source_rate, self.config.sr, quality=self.config.resample_quality
            )
        )
        self.ring = Ring(self.config.win_samples)
        self.pending = np.empty(0, dtype=np.float32)
        self.samples_since_reset = 0
        self.next_window_end = self.config.win_samples
        self.emitted_starts: list[int] = []

    def push(self, block: np.ndarray, *, overflow: bool = False) -> list[np.ndarray]:
        if overflow:
            self.reset()
            return []
        x = np.asarray(block, dtype=np.float32)
        if x.ndim != 1 or not np.all(np.isfinite(x)):
            self.reset()
            return []
        converted = (
            x
            if self.converter is None
            else self.converter.resample_chunk(x, last=False)
        )
        return self._consume(converted)

    def finish(self) -> list[np.ndarray]:
        """Flush the streaming resampler and emit any final complete windows."""
        if self.converter is None:
            return self._consume(np.empty(0, dtype=np.float32))
        tail = self.converter.resample_chunk(np.empty(0, dtype=np.float32), last=True)
        self.converter = None
        return self._consume(tail)

    def _consume(self, converted: np.ndarray) -> list[np.ndarray]:
        self.pending = np.concatenate(
            (self.pending, np.asarray(converted, dtype=np.float32))
        )
        windows: list[np.ndarray] = []
        hop = self.config.hop
        cadence_hops = max(
            1, round(self.config.decision_interval_s * self.config.sr / hop)
        )
        cadence = cadence_hops * hop
        while len(self.pending) >= hop:
            piece = self.pending[:hop]
            self.pending = self.pending[hop:]
            self.samples_since_reset += len(piece)
            ready = self.ring.push(piece)
            if ready and self.samples_since_reset >= self.next_window_end:
                windows.append(self.ring.data)
                self.emitted_starts.append(
                    self.samples_since_reset - self.config.win_samples
                )
                self.next_window_end += cadence
        return windows


@dataclass(frozen=True)
class CaptureEvent:
    generation: int
    samples: np.ndarray | None
    reason: str | None = None
    status: str | None = None


class CaptureCounters:
    """Callback-owned counters; the callback performs no logging or DSP."""

    def __init__(self) -> None:
        self.software_drops = 0
        self.hardware_overflows = 0
        self.invalid_blocks = 0
        self.marker_enqueue_failures = 0
        self.blocks_received = 0
        self.block_size_counts: dict[str, int] = {}
        self.status_flags: dict[str, int] = {}


class CaptureCallback:
    """Copy bounded blocks into a FIFO and label every discontinuity generation."""

    def __init__(self, audio_queue: queue.Queue, *, channel: int = 0) -> None:
        self.audio_queue = audio_queue
        self.channel = channel
        self.generation = 0
        self.counters = CaptureCounters()

    def _mark_discontinuity(self, reason: str, status: str | None = None) -> None:
        self.generation += 1
        marker = CaptureEvent(self.generation, None, reason, status)
        try:
            self.audio_queue.put_nowait(marker)
        except queue.Full:
            self.counters.marker_enqueue_failures += 1

    def __call__(
        self, indata: np.ndarray, frames: int, time_info: Any, status: Any
    ) -> None:
        status_text = str(status) if status else None
        self.counters.blocks_received += 1
        size_key = str(int(frames))
        self.counters.block_size_counts[size_key] = (
            self.counters.block_size_counts.get(size_key, 0) + 1
        )
        if status_text:
            self.counters.status_flags[status_text] = (
                self.counters.status_flags.get(status_text, 0) + 1
            )
        if status and getattr(status, "input_overflow", False):
            self.counters.hardware_overflows += 1
            self._mark_discontinuity("input_overflow", status_text)
            return
        block = np.asarray(indata)
        if (
            block.ndim != 2
            or block.shape[0] != frames
            or block.shape[1] <= self.channel
            or frames <= 0
            or not np.isfinite(block[:, self.channel]).all()
        ):
            self.counters.invalid_blocks += 1
            self._mark_discontinuity("invalid_audio", status_text)
            return
        event = CaptureEvent(
            self.generation,
            block[:, self.channel].astype(np.float32, copy=True),
            None,
            status_text,
        )
        try:
            self.audio_queue.put_nowait(event)
        except queue.Full:
            self.counters.software_drops += 1
            self.generation += 1


class LiveCaptureConsumer:
    """Apply generation boundaries before audio can enter the streaming builder."""

    def __init__(self, config: FrontEndConfig, source_rate: int) -> None:
        self.config = config
        self.source_rate = source_rate
        self.builder = LiveWindowBuilder(config, source_rate)
        self.smoother = VoteSmoother(config.smoothing_windows)
        self.generation = 0
        self.discarded_stale_events = 0
        self.recoveries = 0

    def process(self, event: CaptureEvent) -> list[np.ndarray]:
        if event.generation < self.generation:
            self.discarded_stale_events += 1
            return []
        if event.generation > self.generation:
            self.builder.reset()
            self.smoother.reset()
            self.generation = event.generation
            self.recoveries += 1
        if event.reason is not None or event.samples is None:
            return []
        return self.builder.push(event.samples)


class SessionSequenceGuard:
    """Detect missing/reordered browser chunks, session restarts and rate changes."""

    def __init__(self) -> None:
        self.session_id = None
        self.source_rate = None
        self.expected_sequence = None
        self.generation = 0

    def observe(
        self, session_id: str, source_rate: int, sequence: int
    ) -> tuple[bool, list[str]]:
        reasons = []
        if self.session_id != session_id:
            reasons.append("session_changed")
        if self.source_rate != source_rate:
            reasons.append("sample_rate_changed")
        if self.expected_sequence != sequence:
            reasons.append("sequence_gap")
        self.session_id = session_id
        self.source_rate = source_rate
        self.expected_sequence = sequence + 1
        if reasons:
            self.generation += 1
        return bool(reasons), reasons


def check_stream_parity(
    config: FrontEndConfig, *, seed: int = 20261123, duration_s: float = 6.0
) -> dict:
    """Deterministic G14 T1-T3 streaming parity and discontinuity contract."""
    rng = np.random.default_rng(seed)
    n_source = round(duration_s * config.sr)
    clean = rng.normal(0.0, 0.08, n_source).astype(np.float32)
    builder = LiveWindowBuilder(config, config.sr)
    t1 = []
    for start in range(0, len(clean), 1000):
        t1.extend(builder.push(clean[start : start + 1000]))
    t1.extend(builder.finish())
    expected16 = [
        clean[start : start + config.win_samples] for start in builder.emitted_starts
    ]
    t1_ok = len(t1) == len(expected16) and all(
        np.array_equal(a, b) for a, b in zip(t1, expected16)
    )

    source48 = rng.normal(0.0, 0.08, round(duration_s * 48_000)).astype(np.float32)
    offline = resample(source48, 48_000, config.sr, quality=config.resample_quality)
    sizes = (160, 1000, 4096, 12000)
    t2 = {}
    for chunk_size in sizes:
        stream = resample_stream(48_000, config.sr, quality=config.resample_quality)
        parts = [
            stream.resample_chunk(source48[start : start + chunk_size], last=False)
            for start in range(0, len(source48), chunk_size)
        ]
        parts.append(stream.resample_chunk(np.empty(0, np.float32), last=True))
        streamed = np.concatenate(parts).astype(np.float32)
        n = min(len(offline), len(streamed))
        max_resample_error = float(
            np.max(np.abs(offline[:n] - streamed[:n]), initial=0.0)
        )
        length_match = len(offline) == len(streamed)
        # Measure any startup offset over a 4096-sample prefix, if exact alignment fails.
        candidates = []
        for lag in range(-32, 33):
            a0 = max(0, lag)
            b0 = max(0, -lag)
            width = min(4096, len(offline) - a0, len(streamed) - b0)
            candidates.append(
                (
                    float(
                        np.max(
                            np.abs(
                                offline[a0 : a0 + width] - streamed[b0 : b0 + width]
                            ),
                            initial=0.0,
                        )
                    ),
                    lag,
                )
            )
        prefix_error, offset = min(candidates)
        live = LiveWindowBuilder(config, 48_000)
        windows = []
        for start in range(0, len(source48), chunk_size):
            windows.extend(live.push(source48[start : start + chunk_size]))
        windows.extend(live.finish())
        expected = [
            offline[start : start + config.win_samples] for start in live.emitted_starts
        ]
        db_errors = []
        if len(windows) == len(expected):
            for actual_window, expected_window in zip(windows, expected):
                actual_db = to_db(mel_power(actual_window, config), config)
                expected_db = to_db(mel_power(expected_window, config), config)
                db_errors.append(
                    float(np.max(np.abs(actual_db - expected_db), initial=0.0))
                )
        max_db_error = max(db_errors, default=float("inf"))
        passed = (
            length_match
            and max_resample_error <= 1e-6
            and offset == 0
            and len(windows) == len(expected)
            and max_db_error <= 1e-3
        )
        t2[str(chunk_size)] = {
            "stream_samples": len(streamed),
            "offline_samples": len(offline),
            "max_resampled_signal_abs_difference": max_resample_error,
            "startup_offset_samples": offset,
            "prefix_max_abs_difference": prefix_error,
            "windows_streamed": len(windows),
            "windows_offline": len(expected),
            "max_logmel_abs_difference_db": max_db_error,
            "pass": passed,
        }

    recovery = LiveWindowBuilder(config, 48_000)
    pre = rng.normal(0.0, 0.08, 48_000).astype(np.float32)
    for start in range(0, len(pre), 4096):
        recovery.push(pre[start : start + 4096])
    old_converter = recovery.converter
    cleared = recovery.push(np.zeros(4096, np.float32), overflow=True)
    ring_cleared = not recovery.ring.data.size
    reset_ok = (
        not cleared
        and ring_cleared
        and recovery.samples_since_reset == 0
        and recovery.converter is not None
        and recovery.converter is not old_converter
    )
    post = rng.normal(0.0, 0.08, 48_000).astype(np.float32)
    post_windows = []
    for start in range(0, len(post), 4096):
        post_windows.extend(recovery.push(post[start : start + 4096]))
    post_windows.extend(recovery.finish())
    post_offline = resample(post, 48_000, config.sr, quality=config.resample_quality)
    first_match = bool(
        post_windows
        and recovery.emitted_starts
        and recovery.emitted_starts[0] == 0
        and np.array_equal(post_windows[0], post_offline[: config.win_samples])
    )
    t3_ok = reset_ok and first_match
    return {
        "status": "PASS"
        if t1_ok and all(x["pass"] for x in t2.values()) and t3_ok
        else "STOP",
        "seed": seed,
        "t1_16khz": {
            "windows": len(t1),
            "expected_windows": len(expected16),
            "exact": t1_ok,
        },
        "t2_48khz": t2,
        "t3_overflow_recovery": {
            "ring_cleared": ring_cleared,
            "resampler_replaced": reset_ok,
            "first_post_gap_window_matches_offline": first_match,
            "fresh_samples_before_first_window": (
                recovery.emitted_starts[0] + config.win_samples
                if recovery.emitted_starts
                else None
            ),
            "pass": t3_ok,
        },
    }


def file_mode(
    path: str,
    config: FrontEndConfig,
    stats: dict[str, np.ndarray] | None = None,
    chunk_size: int = 2048,
) -> list[tuple[float, float, bool | None, dict[str, bool]]]:
    audio, source_rate = load_audio(path, config)
    if not np.all(np.isfinite(audio)):
        LOG.warning("file contains non-finite audio; skipping it in demo mode")
        return []
    stats = _stats_for_stub(config, stats)
    builder = LiveWindowBuilder(config, source_rate)
    scorer = StubScorer()
    smoother = VoteSmoother(config.smoothing_windows)
    outputs = []

    def score_windows(windows: list[np.ndarray]) -> None:
        for window in windows:
            flags = clip_flags(window, config)
            feat = transform(mel_power(window, config), config, stats)
            end_sample = builder.emitted_starts.pop(0) + config.win_samples
            raw_score = scorer.score(feat)
            decision = smoother.push(raw_score, 0.5)
            outputs.append((end_sample / config.sr, raw_score, decision, flags))
            LOG.info(
                "stub raw_score=%.4f majority=%s flags=%s", raw_score, decision, flags
            )

    for start in range(0, len(audio), chunk_size):
        score_windows(builder.push(audio[start : start + chunk_size]))
    score_windows(builder.finish())
    return outputs


def live_mode(
    config: FrontEndConfig,
    stats: dict[str, np.ndarray] | None = None,
    *,
    minutes: float | None = None,
    debug_stall_at: tuple[float, ...] = (),
    debug_stall_seconds: float = 2.0,
    log_json: str | Path | None = None,
    scorer_name: str = "stub",
) -> dict[str, object]:
    import sounddevice as sd

    if scorer_name != "stub":
        raise ValueError("the live demo currently supports only --scorer stub")
    audio_queue: queue.Queue = queue.Queue(maxsize=32)
    capture = CaptureCallback(audio_queue, channel=config.channel)
    device_info = {}
    host_api_info = {}
    device_rate = None
    consumer = None
    actual_blocksize = None
    stats = _stats_for_stub(config, stats)
    scorer = StubScorer()
    started = time.monotonic()
    score_times = []
    scored_windows = 0
    skipped_windows = 0
    quiet_windows = 0
    warmup_periods = []
    active_warmup = {
        "generation": 0,
        "started_elapsed_s": 0.0,
        "fresh_samples_required": config.win_samples,
    }
    triggered_stalls = set()
    reported_software_drops = 0
    run_status = "running"
    try:
        device_info = sd.query_devices(kind="input")
        device_rate = int(device_info["default_samplerate"])
        host_api_info = sd.query_hostapis(device_info["hostapi"])
        consumer = LiveCaptureConsumer(config, device_rate)
        with sd.InputStream(
            samplerate=device_rate,
            channels=max(1, config.channel + 1),
            dtype="float32",
            callback=capture,
        ) as stream:
            LOG.info("live capture at %d Hz; Ctrl+C to stop", device_rate)
            actual_blocksize = getattr(stream, "blocksize", None)
            while True:
                elapsed = time.monotonic() - started
                if minutes is not None and elapsed >= minutes * 60:
                    run_status = "completed_by_duration"
                    break
                for stall_second in debug_stall_at:
                    if stall_second not in triggered_stalls and elapsed >= stall_second:
                        triggered_stalls.add(stall_second)
                        LOG.warning(
                            "debug stall begins at %.3f seconds for %.3f seconds",
                            stall_second,
                            debug_stall_seconds,
                        )
                        time.sleep(debug_stall_seconds)
                        LOG.warning(
                            "debug stall ended; capture generations may have advanced"
                        )
                timeout = (
                    0.25
                    if minutes is None
                    else max(0.001, min(0.25, minutes * 60 - elapsed))
                )
                try:
                    event = audio_queue.get(timeout=timeout)
                except queue.Empty:
                    continue
                prior_generation = consumer.generation
                windows = consumer.process(event)
                if consumer.generation != prior_generation:
                    if active_warmup is not None:
                        active_warmup["ended_elapsed_s"] = time.monotonic() - started
                        active_warmup["end_reason"] = (
                            event.reason or "generation_change"
                        )
                        warmup_periods.append(active_warmup)
                    active_warmup = {
                        "generation": consumer.generation,
                        "started_elapsed_s": time.monotonic() - started,
                        "fresh_samples_required": config.win_samples,
                    }
                if event.reason:
                    LOG.warning(
                        "capture discontinuity generation=%d reason=%s status=%s",
                        event.generation,
                        event.reason,
                        event.status,
                    )
                if capture.counters.software_drops > reported_software_drops:
                    reported_software_drops = capture.counters.software_drops
                    LOG.warning(
                        "capture queue drops=%d generation=%d",
                        reported_software_drops,
                        capture.generation,
                    )
                for window in windows:
                    front_started = time.perf_counter()
                    flags = clip_flags(window, config)
                    if flags["silent"] or flags["near_silent"]:
                        consumer.smoother.reset()
                        skipped_windows += 1
                        quiet_windows += 1
                        if active_warmup is None:
                            active_warmup = {
                                "generation": consumer.generation,
                                "started_elapsed_s": time.monotonic() - started,
                                "fresh_samples_required": config.win_samples,
                                "restart_reason": "quiet_window",
                            }
                        continue
                    feat = transform(mel_power(window, config), config, stats)
                    raw_score = scorer.score(feat)
                    decision = consumer.smoother.push(raw_score, 0.5)
                    score_times.append((time.perf_counter() - front_started) * 1000)
                    if decision is None:
                        skipped_windows += 1
                    else:
                        scored_windows += 1
                        if active_warmup is not None:
                            active_warmup["ended_elapsed_s"] = (
                                time.monotonic() - started
                            )
                            active_warmup["end_reason"] = "majority_window_ready"
                            warmup_periods.append(active_warmup)
                            active_warmup = None
                    LOG.info(
                        "stub raw_score=%.4f majority=%s flags=%s front-end %.2f ms generation=%d",
                        raw_score,
                        decision,
                        flags,
                        score_times[-1],
                        consumer.generation,
                    )
    except KeyboardInterrupt:
        run_status = "interrupted"
        raise
    except Exception:
        run_status = "failed"
        raise
    finally:
        ended = time.monotonic()
        duration = ended - started
        if active_warmup is not None:
            active_warmup["ended_elapsed_s"] = duration
            active_warmup["end_reason"] = "session_ended_before_majority"
            warmup_periods.append(active_warmup)
        timing = {
            "count": len(score_times),
            "mean_ms": float(np.mean(score_times)) if score_times else None,
            "p95_ms": float(np.percentile(score_times, 95)) if score_times else None,
            "max_ms": float(np.max(score_times)) if score_times else None,
        }
        session_record = {
            "run_status": run_status,
            "scorer": scorer_name,
            "host_api": host_api_info.get("name"),
            "device_name": device_info.get("name"),
            "native_sample_rate_hz": device_rate,
            "configured_stream_blocksize": actual_blocksize,
            "callback_block_size_counts": capture.counters.block_size_counts,
            "actual_duration_s": duration,
            "requested_minutes": minutes,
            "software_drop_blocks": capture.counters.software_drops,
            "hardware_overflow_blocks": capture.counters.hardware_overflows,
            "invalid_blocks": capture.counters.invalid_blocks,
            "marker_enqueue_failures": capture.counters.marker_enqueue_failures,
            "generations": capture.generation + 1,
            "recoveries": consumer.recoveries if consumer else 0,
            "warmup_periods": warmup_periods,
            "scored_windows": scored_windows,
            "skipped_windows": skipped_windows,
            "quiet_windows": quiet_windows,
            "callback_blocks_received": capture.counters.blocks_received,
            "status_flags": capture.counters.status_flags,
            "debug_stalls_requested_seconds": list(debug_stall_at),
            "debug_stalls_triggered_seconds": sorted(triggered_stalls),
            "debug_stall_seconds": debug_stall_seconds,
            "frontend_timing": timing,
            "decision_interval_s": config.decision_interval_s,
            "decision_interval_hop_quantized_s": round(
                config.decision_interval_s * config.sr / config.hop
            )
            * config.hop
            / config.sr,
            "decision_policy": "threshold-majority-v1",
            "smoothing_windows": config.smoothing_windows,
            "test_audio_or_array_payloads_opened": False,
        }
        if log_json is not None:
            output = Path(log_json)
            if output.exists():
                raise FileExistsError(f"refusing to replace live-session log: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(session_record, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        LOG.info("live session summary %s", json.dumps(session_record, allow_nan=False))
    return session_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the shared frontend using a stub detector."
    )
    parser.add_argument(
        "--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml")
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--file", help="WAV input for offline frontend processing")
    mode.add_argument("--live", action="store_true", help="read the default microphone")
    parser.add_argument("--scorer", choices=("stub",), default="stub")
    parser.add_argument(
        "--minutes", type=float, help="stop live capture after this many minutes"
    )
    parser.add_argument(
        "--debug-stall-at",
        default="",
        help="comma-separated elapsed-second offsets for induced consumer stalls",
    )
    parser.add_argument("--debug-stall-seconds", type=float, default=2.0)
    parser.add_argument(
        "--log-json",
        type=Path,
        help="write a non-overwriting structured live-session log",
    )
    return parser


def parse_stall_times(value: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    try:
        times = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "stall offsets must be comma-separated seconds"
        ) from exc
    if (
        any(not np.isfinite(item) or item < 0 for item in times)
        or tuple(sorted(set(times))) != times
    ):
        raise argparse.ArgumentTypeError(
            "stall offsets must be unique, finite and increasing"
        )
    return times


def validate_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[float, ...]:
    if not np.isfinite(args.debug_stall_seconds) or args.debug_stall_seconds <= 0:
        parser.error("--debug-stall-seconds must be finite and positive")
    if args.live:
        if args.minutes is not None and (
            not np.isfinite(args.minutes) or args.minutes <= 0
        ):
            parser.error("--minutes must be finite and positive")
        try:
            stalls = parse_stall_times(args.debug_stall_at)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
        if args.minutes is not None and any(
            value >= args.minutes * 60 for value in stalls
        ):
            parser.error("each debug stall must start before --minutes elapses")
        if stalls and args.minutes is None:
            parser.error("debug stalls require a bounded --minutes live session")
        return stalls
    if args.minutes is not None or args.debug_stall_at:
        parser.error("--minutes and debug stalls require --live")
    return ()


def main(argv: list[str] | None = None) -> dict[str, object] | None:
    parser = build_parser()
    args = parser.parse_args(argv)
    stalls = validate_args(args, parser)
    cfg = load_config(args.config)
    if args.live:
        return live_mode(
            cfg,
            minutes=args.minutes,
            debug_stall_at=stalls,
            debug_stall_seconds=args.debug_stall_seconds,
            log_json=args.log_json,
            scorer_name=args.scorer,
        )
    outputs = file_mode(args.file, cfg)
    print(f"Processed {len(outputs)} windows with stub detector")


if __name__ == "__main__":
    main()
