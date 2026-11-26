"""V4 activation checks: no test corpus access or threshold selection."""
import hashlib
import http.client
import json
from pathlib import Path
import sys
import threading
from unittest.mock import patch

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import live_detector as live
from dronedet.inference import load_detector
from dronedet.features import standardize, mel_power, transform


@pytest.fixture(scope="module")
def detector():
    return load_detector(ROOT)


def test_checkpoint_and_training_stats_are_the_selected_v4_run(detector):
    assert detector.model_id == "CNN_V4.0"
    assert detector.threshold is None
    assert not detector.model.training
    assert detector.health_metadata()["parameter_count"] == 23585
    assert detector.health_metadata()["threshold_selected"] is False
    assert detector.artifact_sha256 == "f4372da59644395c174514944aa6387901ed483f64c137fd1ae4e31c83092049"
    base = ROOT.parents[1]
    with np.load(base / "storage/caches/corpus-002/processed/spectrograms/norm_stats_global.npz") as stats:
        for key in ("mean", "std"):
            np.testing.assert_array_equal(detector.stats[key], stats[key])


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_live_scores_match_training_frontend_at_browser_rates(detector, rate):
    source = np.random.default_rng(42).normal(0, .05, 4 * rate).astype(np.float32)
    offline = standardize(source, rate, detector.config)
    state = live.LiveState(detector.config, detector.stats, detector.model, None)
    token = state.claim("parity")
    count = 0
    for sequence, at in enumerate(range(0, len(source), rate // 2)):
        result = state.process(source[at:at + rate // 2], rate, "parity", sequence,
                               token, require_owner=True)
        for event in result.get("window_trace", []):
            assert event["status"] == "score_only"
            start = state.builder.emitted_starts[count]
            expected = transform(mel_power(offline[start:start + 7936], detector.config),
                                 detector.config, detector.stats)
            with torch.inference_mode():
                score = detector.model(torch.from_numpy(expected[None])).item()
            assert event["raw_score"] == pytest.approx(score, abs=1e-4)
            assert event["threshold"] is None and event["drone"] is None
            assert event["votes"] == []
            count += 1
    assert count >= 6


def test_gap_resets_stream_and_nan_model_output_is_rejected(detector):
    state = live.LiveState(detector.config, detector.stats, detector.model, None)
    block = np.ones(7936, np.float32) * .02
    state.process(block, 16000, "gap", 0)
    result = state.process(block, 16000, "gap", 2)
    assert result["recovered"]
    assert state.builder.emitted_starts == [0]
    class InvalidModel:
        def __call__(self, x):
            return torch.tensor([float("nan")])
    invalid = live.LiveState(detector.config, detector.stats, InvalidModel(), None)
    with pytest.raises(ValueError, match="finite"):
        invalid.process(block, 16000, "invalid", 0)


def test_changed_registry_binding_is_rejected():
    from dronedet import inference
    original = inference._sha256
    with patch.object(inference, "_sha256", side_effect=lambda path:
                      "0" * 64 if Path(path).name == "live_detector.json" else original(path)):
        with pytest.raises(ValueError, match="registry"):
            load_detector(ROOT)


def test_http_scores_without_threshold_and_preserves_session_ownership(detector):
    live.MODEL_INFO = detector.health_metadata()
    live.INSTANCE_ID = "v4-test"
    live.STATE = live.LiveState(detector.config, detector.stats, detector.model, None)
    server = live.ThreadingHTTPServer(("127.0.0.1", 0), live.Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    port = server.server_port
    token = live.STATE.claim("http")
    try:
        for sequence in range(3):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            conn.request("POST", "/api/audio", body=(np.ones(8000, "<f4") * .02).tobytes(),
                         headers={"Host": f"localhost:{port}", "Origin": f"http://localhost:{port}",
                                  "Content-Type": "application/octet-stream", "X-Sample-Rate": "16000",
                                  "X-Session-ID": "http", "X-Session-Token": token,
                                  "X-Chunk-Sequence": str(sequence)})
            response = conn.getresponse()
            assert response.status == 200
            result = json.loads(response.read())
            assert result["status"] == "score_only"
            assert result["threshold"] is None and result["drone"] is None
            conn.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
