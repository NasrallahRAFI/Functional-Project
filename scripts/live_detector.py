"""Local browser service for continuous inference with the explicitly selected CNN package."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timezone

import numpy as np
import torch
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from demo.run_demo import (
    LiveWindowBuilder,
    SessionSequenceGuard,
    VoteSmoother,
)
from dronedet.config import decision_hash
from dronedet.features import clip_flags, mel_power, transform
from dronedet.inference import load_detector as load_verified_detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("dronedet.live")
OWNER_LEASE_SECONDS = 15


def load_detector():
    return load_verified_detector(ROOT)


class LiveState:
    def __init__(self, config, stats, model, threshold):
        self.config, self.stats, self.model, self.threshold = (
            config,
            stats,
            model,
            threshold,
        )
        self.decision_identity = (
            decision_hash(config, threshold=threshold) if threshold is not None
            else "raw-score-only-v1"
        )
        self.lock = threading.Lock()
        self.source_rate = None
        self.builder = None
        self.smoother = VoteSmoother(config.smoothing_windows)
        self.session_id = None
        self.expected_sequence = None
        self.generation = 0
        self.continuity = SessionSequenceGuard()
        self.owner_token = None
        self.owner_session = None
        self.owner_touched = 0.0

    def claim(self, session_id: str) -> str:
        with self.lock:
            if self.owner_token and time.monotonic() - self.owner_touched >= OWNER_LEASE_SECONDS:
                try:
                    stop_score_log(self.owner_session, self.owner_token, "owner_lease_expired")
                except OSError:
                    LOG.exception("Could not finalize the expired ambient score log")
                self.owner_token = self.owner_session = None
                self.owner_touched = 0.0
                self.reset()
            if self.owner_token:
                raise RuntimeError("Another monitor tab already owns the microphone session.")
            self.owner_token = uuid.uuid4().hex
            self.owner_session = session_id
            self.owner_touched = time.monotonic()
            self.continuity = SessionSequenceGuard()
            self.reset(session_id=session_id)
            return self.owner_token

    def release(self, token: str) -> None:
        with self.lock:
            if not self.owner_token or token != self.owner_token:
                raise ValueError("Session owner token is invalid or expired.")
            try:
                stop_score_log(self.owner_session, token, "monitor_stopped")
            except OSError:
                LOG.exception("Could not finalize the ambient score log during monitor stop")
            self.owner_token = self.owner_session = None
            self.owner_touched = 0.0
            self.reset()

    def reset(self, *, session_id=None, source_rate=None, next_sequence=None):
        self.generation += 1
        self.session_id = session_id
        self.source_rate = source_rate
        self.expected_sequence = next_sequence
        self.builder = (
            LiveWindowBuilder(self.config, source_rate)
            if source_rate is not None
            else None
        )
        self.smoother.reset()

    def process(
        self, samples: np.ndarray, source_rate: int, session_id: str, sequence: int,
        owner_token: str | None = None, *, require_owner: bool = False,
    ):
        if source_rate < 8000 or source_rate > 192000:
            raise ValueError("Unsupported microphone sample rate.")
        if samples.size == 0 or not np.isfinite(samples).all():
            with self.lock:
                self.continuity.observe(session_id, source_rate, sequence)
                self.reset(
                    session_id=session_id,
                    source_rate=source_rate,
                    next_sequence=sequence + 1,
                )
            raise ValueError("Audio chunk is empty or contains invalid samples.")
        with self.lock:
            if self.owner_token and time.monotonic() - self.owner_touched >= OWNER_LEASE_SECONDS:
                try:
                    stop_score_log(self.owner_session, self.owner_token, "owner_lease_expired")
                except OSError:
                    LOG.exception("Could not finalize the expired ambient score log")
                self.owner_token = self.owner_session = None
                self.owner_touched = 0.0
                self.reset()
            if require_owner and not self.owner_token:
                raise ValueError("Microphone session ownership is missing or expired.")
            if self.owner_token and (owner_token != self.owner_token or session_id != self.owner_session):
                raise ValueError("Microphone session ownership is invalid or expired.")
            if self.owner_token:
                self.owner_touched = time.monotonic()
            if (self.continuity.session_id == session_id
                    and self.continuity.source_rate == source_rate
                    and self.continuity.expected_sequence is not None
                    and sequence < self.continuity.expected_sequence):
                raise ValueError("Audio chunk is stale or duplicated.")
            discontinuity, reasons = self.continuity.observe(
                session_id, source_rate, sequence
            )
            if discontinuity:
                self.reset(
                    session_id=session_id,
                    source_rate=source_rate,
                    next_sequence=sequence + 1,
                )
            else:
                self.expected_sequence = sequence + 1
            windows = self.builder.push(samples)
            if not windows:
                return {
                    "status": "buffering",
                    "generation": self.generation,
                    "recovered": discontinuity,
                    "recovery_reasons": reasons,
                    "decision_identity": self.decision_identity,
                }
            result = {"status": "standby"}
            window_trace = []
            for window in windows:
                flags = clip_flags(window, self.config)
                window_rms_dbfs = round(
                    float(20 * np.log10(max(float(np.sqrt(np.mean(window.astype(np.float64) ** 2))), 1e-15))),
                    3,
                )
                if flags["silent"] or flags["near_silent"]:
                    self.smoother.values.clear()
                    result = {
                        "status": "standby",
                        "reason": "quiet",
                        "window_rms_dbfs": window_rms_dbfs,
                        "votes": [],
                    }
                    window_trace.append({**result, "window_rms_dbfs": window_rms_dbfs})
                    continue
                with torch.inference_mode():
                    features = transform(
                        mel_power(window, self.config), self.config, self.stats
                    )
                    score = float(
                        self.model(torch.from_numpy(features).unsqueeze(0)).item()
                    )
                if not np.isfinite(score):
                    raise ValueError("CNN output must be finite")
                if self.threshold is None:
                    result = {
                        "status": "score_only", "score": score, "raw_score": score,
                        "threshold": None, "drone": None, "votes": [], "votes_required": 0,
                        "window_rms_dbfs": window_rms_dbfs, "clipped": flags["clipped"],
                        "generation": self.generation, "recovered": discontinuity,
                        "recovery_reasons": reasons, "decision_identity": self.decision_identity,
                    }
                    window_trace.append(result.copy())
                    continue
                decision = self.smoother.push(score, self.threshold)
                if decision is None:
                    result = {
                        "status": "warming_up",
                        "raw_score": score,
                        "threshold": self.threshold,
                        "votes_received": len(self.smoother.values),
                        "votes_required": self.smoother.count,
                        "votes": self.smoother.values.copy(),
                        "raw_vote": bool(score >= self.threshold),
                        "window_rms_dbfs": window_rms_dbfs,
                        "generation": self.generation,
                        "recovered": discontinuity,
                        "recovery_reasons": reasons,
                        "decision_identity": self.decision_identity,
                    }
                    window_trace.append(result.copy())
                    continue
                result = {
                    "status": "scored",
                    "score": score,
                    "raw_score": score,
                    "threshold": self.threshold,
                    "drone": decision,
                    "votes": self.smoother.values.copy(),
                    "raw_vote": bool(score >= self.threshold),
                    "votes_required": self.smoother.count,
                    "window_rms_dbfs": window_rms_dbfs,
                    "clipped": flags["clipped"],
                    "generation": self.generation,
                    "recovered": discontinuity,
                    "recovery_reasons": reasons,
                    "decision_identity": self.decision_identity,
                }
                window_trace.append(result.copy())
            result["window_trace"] = window_trace
            return result


CONFIG = STATS = MODEL = THRESHOLD = SEED = ARTIFACT = STATE = MODEL_INFO = None
INSTANCE_ID = None
RECORDINGS_DIR = ROOT / "runtime" / "recordings"
SCORE_LOG_DIR = PROJECT / "local" / "reports" / "study-v4" / "ambient-score-logs"
SCORE_LOGS: dict[str, dict] = {}
SCORE_LOG_LOCK = threading.Lock()


def _score_log_owned(session_id: str, owner_token: str) -> dict | None:
    with SCORE_LOG_LOCK:
        record = SCORE_LOGS.get(session_id)
        if not record or record["owner_token"] != owner_token:
            return None
        return record.copy()


def start_score_log(session_id: str, owner_token: str, environment_label: str) -> dict:
    with STATE.lock:
        if (not STATE.owner_token or owner_token != STATE.owner_token
                or session_id != STATE.owner_session
                or time.monotonic() - STATE.owner_touched >= OWNER_LEASE_SECONDS):
            raise ValueError("An active monitor session is required to start an ambient log.")
    label = " ".join(environment_label.split())[:120] or "unspecified"
    SCORE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    filename = f"ambient-v4-{started.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}.jsonl"
    destination = SCORE_LOG_DIR / filename
    record = {"path": destination, "owner_token": owner_token,
              "started_monotonic": time.monotonic(), "windows": 0}
    header = {"record_type": "session_start", "schema_version": 1,
              "session_id": session_id, "started_at_utc": started.isoformat(),
              "environment_label": label, "purpose": "ambient_only_no_drone",
              "model_id": MODEL_INFO.get("model_id"),
              "model_version": MODEL_INFO.get("model_version"),
              "run_id": MODEL_INFO.get("run_id"),
              "checkpoint_sha256": MODEL_INFO.get("checkpoint_sha256"),
              "statistics_sha256": MODEL_INFO.get("statistics_sha256"),
              "frontend_hash": MODEL_INFO.get("frontend_hash"),
              "threshold_selected": False, "threshold": None,
              "smoothing_windows_for_candidate_replay": CONFIG.smoothing_windows,
              "cadence_seconds": MODEL_INFO.get("cadence_seconds"),
              "audio_saved": False, "test_access": False}
    with SCORE_LOG_LOCK:
        if session_id in SCORE_LOGS:
            raise ValueError("An ambient score log is already active for this session.")
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(header, allow_nan=False) + "\n")
        SCORE_LOGS[session_id] = record
    return {"status": "logging", "filename": filename,
            "path": str(destination), "started_at_utc": started.isoformat()}


def append_score_log(session_id: str, owner_token: str, sequence: int,
                     source_rate: int, capture_drops: int, source_frame: int | None,
                     result: dict) -> None:
    record = _score_log_owned(session_id, owner_token)
    if record is None:
        return
    windows = result.get("window_trace", [])
    now = datetime.now(timezone.utc).isoformat()
    entries = []
    for index, window in enumerate(windows):
        score = window.get("raw_score")
        rms = window.get("window_rms_dbfs")
        if score is not None and not math.isfinite(float(score)):
            raise ValueError("Cannot log a non-finite CNN score.")
        if rms is not None and not math.isfinite(float(rms)):
            raise ValueError("Cannot log a non-finite RMS value.")
        entries.append({"record_type": "window", "timestamp_utc": now,
            "elapsed_seconds": round(time.monotonic() - record["started_monotonic"], 3),
            "session_id": session_id, "chunk_sequence": sequence,
            "window_in_chunk": index, "source_sample_rate_hz": source_rate,
            "source_frame": source_frame, "capture_dropped_frames_total": capture_drops,
            "status": window.get("status"), "reason": window.get("reason"),
            "raw_logit": score, "rms_dbfs": rms,
            "generation": window.get("generation", result.get("generation")),
            "recovered": bool(window.get("recovered", result.get("recovered", False))),
            "recovery_reasons": window.get("recovery_reasons", result.get("recovery_reasons", []))})
    if not entries:
        return
    with SCORE_LOG_LOCK:
        active = SCORE_LOGS.get(session_id)
        if not active or active["owner_token"] != owner_token:
            return
        try:
            with active["path"].open("a", encoding="utf-8", newline="\n") as stream:
                for entry in entries:
                    stream.write(json.dumps(entry, allow_nan=False) + "\n")
        except OSError:
            SCORE_LOGS.pop(session_id, None)
            raise
        active["windows"] += len(entries)


def stop_score_log(session_id: str, owner_token: str, reason: str = "user_stopped") -> dict:
    with SCORE_LOG_LOCK:
        record = SCORE_LOGS.get(session_id)
        if not record:
            return {"status": "inactive"}
        if record["owner_token"] != owner_token:
            raise ValueError("Ambient score log owner is invalid.")
        footer = {"record_type": "session_end", "session_id": session_id,
                  "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                  "elapsed_seconds": round(time.monotonic() - record["started_monotonic"], 3),
                  "window_records": record["windows"], "end_reason": reason,
                  "audio_saved": False}
        try:
            with record["path"].open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(footer, allow_nan=False) + "\n")
        finally:
            del SCORE_LOGS[session_id]
        return {"status": "saved", "filename": record["path"].name,
                "path": str(record["path"]), "window_records": record["windows"]}


def save_diagnostic_recording(samples, source_rate, session_id, owner_token):
    if not 8000 <= source_rate <= 192000:
        raise ValueError("Unsupported recording sample rate.")
    if samples.ndim != 1 or not 1 <= samples.size <= source_rate * 10 or not np.isfinite(samples).all():
        raise ValueError("Recording must contain finite mono audio of at most 10 seconds.")
    with STATE.lock:
        if (not STATE.owner_token or owner_token != STATE.owner_token
                or session_id != STATE.owner_session
                or time.monotonic() - STATE.owner_touched >= OWNER_LEASE_SECONDS):
            raise ValueError("An active monitor session is required to save a recording.")
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"microphone_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}.wav"
        destination = RECORDINGS_DIR / filename
        temporary = destination.with_suffix(".pending")
        try:
            sf.write(temporary, samples, source_rate, format="WAV", subtype="FLOAT")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {"status": "saved", "filename": filename, "sample_rate": source_rate,
            "duration_seconds": samples.size / source_rate}


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript"}
    request_timeout_seconds = 5

    def setup(self):
        super().setup()
        self.connection.settimeout(self.request_timeout_seconds)

    def end_headers(self):
        # Reopening the monitor must not reuse a cached V1 page or script.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "live"), **kwargs)

    def log_message(self, fmt, *args):
        LOG.info("%s - %s", self.address_string(), fmt % args)

    def _json(self, status: int, payload: dict):
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _local_host_ok(self) -> bool:
        host = self.headers.get("Host", "")
        return host in {f"127.0.0.1:{self.server.server_port}",
                        f"localhost:{self.server.server_port}"}

    def _same_origin_ok(self) -> bool:
        return self._local_host_ok() and self.headers.get("Origin", "") == f"http://{self.headers.get('Host', '')}"

    def do_GET(self):
        if not self._local_host_ok():
            return self._json(403, {"error": "Host is not allowed."})
        path = urlparse(self.path).path
        if path == "/":
            self.send_response(302)
            self.send_header("Location", "/live/index.html")
            self.end_headers()
            return
        if path == "/api/health":
            return self._json(
                200,
                {
                    "status": "ready",
                    "application_id": "local-drone-detector",
                    "protocol_version": 2,
                    "instance_id": INSTANCE_ID,
                    **MODEL_INFO,
                },
            )
        if path == "/api/score-log/status":
            session_id = self.headers.get("X-Session-ID", "")
            owner_token = self.headers.get("X-Session-Token", "")
            record = _score_log_owned(session_id, owner_token)
            return self._json(200, {"active": record is not None,
                "window_records": record["windows"] if record else 0,
                "filename": record["path"].name if record else None})
        if path in {"/live/index.html", "/live/app.js", "/live/styles.css", "/live/audio-capture-worklet.js"}:
            original = self.path
            self.path = "/" + path.rsplit("/", 1)[1]
            try:
                return super().do_GET()
            finally:
                self.path = original
        return self._json(404, {"error": "Resource not found."})

    def do_POST(self):
        if not self._local_host_ok() or not self._same_origin_ok():
            return self._json(403, {"error": "Request must come from the local monitor origin."})
        path = urlparse(self.path).path
        if path == "/api/diagnostic-recording":
            try:
                if self.headers.get_content_type() != "application/octet-stream":
                    raise ValueError("Recording must be float32 audio.")
                source_rate = int(self.headers.get("X-Sample-Rate", "0"))
                length = int(self.headers.get("Content-Length", "0"))
                if not 8000 <= source_rate <= 192000 or not 0 < length <= source_rate * 10 * 4 or length % 4:
                    raise ValueError("Recording size or sample rate is invalid; maximum 10 seconds.")
                self.connection.settimeout(15)
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("Recording upload was truncated.")
                result = save_diagnostic_recording(np.frombuffer(body, dtype="<f4"), source_rate,
                    self.headers.get("X-Session-ID", ""), self.headers.get("X-Session-Token", ""))
                return self._json(200, result)
            except (ValueError, TimeoutError) as exc:
                return self._json(400, {"error": str(exc)})
            except OSError:
                LOG.exception("Could not save diagnostic recording")
                return self._json(500, {"error": "Could not save the recording locally. Check available disk space."})
        if path in {"/api/score-log/start", "/api/score-log/stop"}:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 4096:
                    raise ValueError("Score log request size is invalid.")
                payload = json.loads(self.rfile.read(length))
                session_id = str(payload.get("session_id", ""))
                owner_token = str(payload.get("owner_token", ""))
                if path.endswith("/start"):
                    if payload.get("confirmed_ambient_only") is not True:
                        raise ValueError("Confirm that this session has no drone present.")
                    if payload.get("threshold_selected") is not False:
                        raise ValueError("Ambient collection only supports the pending-threshold V4 score monitor.")
                    label = str(payload.get("environment_label", ""))
                    if len(label) > 120:
                        raise ValueError("Environment label must be 120 characters or fewer.")
                    return self._json(200, start_score_log(session_id, owner_token, label))
                return self._json(200, stop_score_log(session_id, owner_token))
            except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
                return self._json(409, {"error": str(exc)})
            except OSError:
                LOG.exception("Could not write the ambient-only score log")
                return self._json(507, {"error": "Could not write the local score log. Check available disk space and folder permissions."})
        if path in {"/api/session/start", "/api/session/stop"}:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 4096:
                    raise ValueError("Session request size is invalid.")
                payload = json.loads(self.rfile.read(length))
                if path.endswith("/start"):
                    session_id = str(payload.get("session_id", ""))
                    if not session_id or len(session_id) > 80:
                        raise ValueError("Session ID is invalid.")
                    return self._json(200, {"owner_token": STATE.claim(session_id)})
                STATE.release(str(payload.get("owner_token", "")))
                return self._json(200, {"status": "stopped"})
            except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
                return self._json(409, {"error": str(exc)})
        if path != "/api/audio":
            return self._json(404, {"error": "Endpoint not found."})
        try:
            if self.headers.get_content_type() != "application/octet-stream":
                raise ValueError("Audio content type must be application/octet-stream.")
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2_000_000 or length % 4:
                raise ValueError("Audio chunk size is invalid.")
            source_rate = int(self.headers.get("X-Sample-Rate", "0"))
            session_id = self.headers.get("X-Session-ID", "")
            owner_token = self.headers.get("X-Session-Token", "")
            sequence = int(self.headers.get("X-Chunk-Sequence", "-1"))
            capture_drops = int(self.headers.get("X-Capture-Drops", "0"))
            source_frame_header = self.headers.get("X-Source-Frame")
            source_frame = int(source_frame_header) if source_frame_header is not None else None
            if not session_id or len(session_id) > 80 or sequence < 0:
                raise ValueError("Audio session or sequence metadata is invalid.")
            if capture_drops < 0 or (source_frame is not None and source_frame < 0):
                raise ValueError("Audio continuity metadata is invalid.")
            if not 8000 <= source_rate <= 192000:
                raise ValueError("Unsupported microphone sample rate.")
            if length > source_rate * 2 * 4:
                raise ValueError("Audio chunk is longer than two seconds.")
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Audio request was truncated.")
            samples = np.frombuffer(body, dtype="<f4").copy()
            if samples.size > source_rate * 2:
                raise ValueError("Audio chunk is longer than two seconds.")
            result = STATE.process(samples, source_rate, session_id, sequence, owner_token,
                                   require_owner=True)
            try:
                append_score_log(session_id, owner_token, sequence, source_rate,
                                 capture_drops, source_frame, result)
            except OSError:
                LOG.exception("Ambient score log write failed; inference remains active")
                result["score_log_error"] = "Local score logging stopped after a write error. This session log may be incomplete."
            return self._json(200, result)
        except (ValueError, RuntimeError, TimeoutError) as exc:
            return self._json(400, {"error": str(exc)})
        except Exception:
            LOG.exception("Live inference failed")
            return self._json(
                500, {"error": "Audio inference failed. Restart the local detector."}
            )


def main(argv=None):
    global CONFIG, STATS, MODEL, THRESHOLD, SEED, ARTIFACT, STATE, MODEL_INFO, INSTANCE_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify and warm the selected CNN, then exit")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--instance-id", default=uuid.uuid4().hex)
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("the local detector can only bind to loopback")
    if not args.instance_id or len(args.instance_id) > 80:
        parser.error("instance ID must contain 1–80 characters")
    detector = load_verified_detector(ROOT)
    CONFIG, STATS, MODEL, THRESHOLD, SEED, ARTIFACT = (
        detector.config, detector.stats, detector.model, detector.threshold,
        detector.seed, detector.artifact,
    )
    MODEL_INFO = detector.health_metadata()
    if args.check:
        print(json.dumps({"status": "ready", **detector.health_metadata()}))
        return 0
    INSTANCE_ID = args.instance_id
    STATE = LiveState(CONFIG, STATS, MODEL, THRESHOLD)
    host, port = args.host, args.port
    server = ThreadingHTTPServer((host, port), Handler)
    LOG.info(
        "Loaded verified %s seed %s from %s; threshold %s; validation_eligible=%s experimental=%s",
        detector.model_id,
        SEED,
        ARTIFACT,
        THRESHOLD,
        detector.validation_eligible,
        detector.experimental_activation,
    )
    LOG.info("Listening at http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("Stopping local detector")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    main()
