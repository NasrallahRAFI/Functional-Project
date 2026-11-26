"""Local release checks for the selected CNN and its HTTP/session boundary."""
from __future__ import annotations

import json
import sys
import threading
import time
import tempfile
from unittest.mock import patch
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.live_detector as live
from dronedet.inference import load_detector
from dronedet.models.cnn import parameter_count
from dronedet.resample import resample


class LiveDetectorReleaseTests(unittest.TestCase):
    def test_recording_is_explicit_authenticated_and_preserves_float_samples(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(live, 'RECORDINGS_DIR', Path(directory)):
            audio = np.linspace(-0.2, 0.2, 1600, dtype='<f4')
            headers = {'Content-Type': 'application/octet-stream', 'X-Sample-Rate': '16000',
                       'X-Session-ID': 'recording-test', 'X-Session-Token': 'invalid'}
            status, _ = self.request('/api/diagnostic-recording', method='POST', body=audio.tobytes(), headers=headers)
            self.assertEqual(status, 400)
            self.assertEqual(list(Path(directory).iterdir()), [])
            headers['X-Session-Token'] = live.STATE.claim('recording-test')
            status, body = self.request('/api/diagnostic-recording', method='POST', body=audio.tobytes(), headers=headers)
            result = json.loads(body)
            self.assertEqual(status, 200)
            decoded, rate = sf.read(Path(directory) / result['filename'], dtype='float32')
            self.assertEqual(rate, 16000)
            np.testing.assert_array_equal(decoded, audio)
            self.assertEqual(self.request('/runtime/recordings/' + result['filename'])[0], 404)
            bad_audio = np.array([float('nan')], dtype='<f4')
            self.assertEqual(self.request('/api/diagnostic-recording', method='POST', body=bad_audio.tobytes(), headers=headers)[0], 400)
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    @classmethod
    def setUpClass(cls):
        cls.detector = load_detector(ROOT)
        live.CONFIG = cls.detector.config
        live.STATS = cls.detector.stats
        live.MODEL = cls.detector.model
        live.THRESHOLD = cls.detector.threshold
        live.SEED = cls.detector.seed
        live.ARTIFACT = cls.detector.artifact
        live.MODEL_INFO = cls.detector.health_metadata()
        live.INSTANCE_ID = "release-test-instance"
        live.STATE = live.LiveState(live.CONFIG, live.STATS, live.MODEL, live.THRESHOLD)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), live.Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)

    def setUp(self):
        live.STATE = live.LiveState(live.CONFIG, live.STATS, live.MODEL, live.THRESHOLD)

    def request(self, path, *, method="GET", body=None, headers=None):
        header_values = dict(headers or {})
        if method == "POST":
            header_values.setdefault("Origin", self.base)
        request = urllib.request.Request(self.base + path, data=body, headers=header_values, method=method)
        try:
            response = urllib.request.urlopen(request, timeout=8)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, response.read()

    def test_registered_best_checkpoint_is_loaded_and_scores_finitely(self):
        meta = self.detector.health_metadata()
        self.assertEqual((meta["model_id"], meta["stage"], meta["variant"], meta["seed"]),
                         ("CNN_V4.0", "A0", "depth_3", 42))
        self.assertEqual(meta["checkpoint_sha256"],
                         "f4372da59644395c174514944aa6387901ed483f64c137fd1ae4e31c83092049")
        self.assertIsNone(meta["threshold"])
        self.assertFalse(self.detector.model.training)
        self.assertEqual(parameter_count(self.detector.model), 23585)
        score = self.detector.score_window(np.zeros(self.detector.config.win_samples, np.float32))
        self.assertTrue(np.isfinite(score))

    def test_http_scores_match_direct_v2_inference_at_browser_sample_rates(self):
        # Independent offline resampling and explicit slices catch serving-path
        # preprocessing drift, wrong weights/statistics, and threshold changes.
        for rate in (16000, 44100, 48000):
            with self.subTest(rate=rate):
                live.STATE = live.LiveState(live.CONFIG, live.STATS, live.MODEL, live.THRESHOLD)
                session = f"parity-{rate}"
                token = live.STATE.claim(session)
                source = np.random.default_rng(42).normal(0, 0.03, rate * 4).astype('<f4')
                offline = resample(source, rate, 16000, quality=live.CONFIG.resample_quality)
                votes = []
                scored = 0
                for seq, offset in enumerate(range(0, len(source), rate // 2)):
                    status, body = self.request('/api/audio', method='POST',
                        body=source[offset:offset + rate // 2].tobytes(), headers={
                            'Content-Type': 'application/octet-stream', 'X-Sample-Rate': str(rate),
                            'X-Session-ID': session, 'X-Session-Token': token,
                            'X-Chunk-Sequence': str(seq)})
                    self.assertEqual(status, 200)
                    result = json.loads(body)
                    if result['status'] not in {'score_only'}:
                        continue
                    event = result['window_trace'][-1]
                    start = live.STATE.builder.emitted_starts[-1]
                    expected = self.detector.score_window(offline[start:start + 7936])
                    self.assertAlmostEqual(result['raw_score'], expected, delta=1e-4)
                    self.assertEqual(result['threshold'], self.detector.threshold)
                    self.assertAlmostEqual(event['raw_score'], expected, delta=1e-4)
                    self.assertIsNone(event.get('raw_vote'))
                    self.assertEqual(event['votes'], live.STATE.smoother.values)
                    self.assertIsNotNone(event['window_rms_dbfs'])
                    self.assertEqual(result['status'], 'score_only')
                    self.assertIsNone(result['drone'])
                    scored += 1
                self.assertGreaterEqual(scored, 3)

    def test_health_static_scope_and_session_ownership(self):
        status, body = self.request("/api/health")
        self.assertEqual(status, 200)
        health = json.loads(body)
        self.assertEqual(health["seed"], 42)
        self.assertEqual(health["model_id"], "CNN_V4.0")
        self.assertEqual(health["instance_id"], "release-test-instance")
        self.assertEqual(health["checkpoint_sha256"], self.detector.artifact_sha256)
        for path in ('/live/index.html', '/live/app.js'):
            with urllib.request.urlopen(self.base + path, timeout=5) as response:
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.assertEqual(self.request("/results/models/30/metadata.json")[0], 404)
        self.assertEqual(self.request("/live/audio-capture-worklet.js")[0], 200)
        status, _ = self.request("/api/session/start", method="POST",
                                 body=b'{"session_id":"foreign"}',
                                 headers={"Content-Type": "application/json", "Origin": "http://evil.example"})
        self.assertEqual(status, 403)

        start = json.dumps({"session_id": "release-test"}).encode()
        status, body = self.request("/api/session/start", method="POST", body=start,
                                    headers={"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        owner = json.loads(body)["owner_token"]
        status, _ = self.request("/api/session/start", method="POST",
                                 body=json.dumps({"session_id": "second-tab"}).encode(),
                                 headers={"Content-Type": "application/json"})
        self.assertEqual(status, 409)

        audio = np.zeros(22050, dtype="<f4").tobytes()
        audio_headers = {"Content-Type": "application/octet-stream", "X-Sample-Rate": "44100",
                         "X-Session-ID": "release-test", "X-Session-Token": owner,
                         "X-Chunk-Sequence": "0"}
        status, body = self.request("/api/audio", method="POST", body=audio, headers=audio_headers)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "buffering")
        status, _ = self.request("/api/audio", method="POST", body=audio, headers=audio_headers)
        self.assertEqual(status, 400)
        status, body = self.request("/api/audio", method="POST", body=audio,
                                    headers={**audio_headers, "X-Chunk-Sequence": "2"})
        recovered = json.loads(body)
        self.assertEqual(status, 200)
        self.assertTrue(recovered["recovered"])
        self.assertIn("sequence_gap", recovered["recovery_reasons"])
        self.assertEqual(recovered["status"], "buffering")
        status, _ = self.request("/api/audio", method="POST", body=audio,
                                 headers={**audio_headers, "X-Chunk-Sequence": "2"})
        self.assertEqual(status, 400)
        status, _ = self.request("/api/session/stop", method="POST",
                                 body=json.dumps({"session_id": "release-test", "owner_token": owner}).encode(),
                                 headers={"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        status, _ = self.request("/api/audio", method="POST", body=audio,
                                 headers={**audio_headers, "X-Chunk-Sequence": "1"})
        self.assertEqual(status, 400)

        stale = live.STATE.claim("expired-session")
        live.STATE.owner_touched = time.monotonic() - live.OWNER_LEASE_SECONDS - 1
        replacement = live.STATE.claim("replacement-session")
        self.assertNotEqual(stale, replacement)
        self.assertEqual(live.STATE.owner_session, "replacement-session")


if __name__ == "__main__":
    unittest.main()
