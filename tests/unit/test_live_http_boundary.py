"""HTTP boundary regressions for the local microphone service."""
import http.client
from pathlib import Path
import sys
import threading
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import live_detector as live


@pytest.fixture
def server():
    with patch.object(live, "MODEL_INFO", {}), patch.object(live, "INSTANCE_ID", "boundary"), \
         patch.object(live.Handler, "request_timeout_seconds", .15):
        service = live.ThreadingHTTPServer(("127.0.0.1", 0), live.Handler)
        worker = threading.Thread(target=service.serve_forever, daemon=True)
        worker.start()
        yield service
        service.shutdown()
        service.server_close()
        worker.join()


def request(server, host, path="/api/health", method="GET", body=None, extra=None):
    port = server.server_port
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    headers = {"Host": host, **(extra or {})}
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    result = response.status, response.read(), response.getheader("Content-Type")
    conn.close()
    return result


def test_loopback_host_allowlist(server):
    port = server.server_port
    for host in (f"localhost:{port}", f"127.0.0.1:{port}"):
        assert request(server, host)[0] == 200
    for host in (f"evil.example:{port}", "localhost", f"localhost:{port + 1}"):
        assert request(server, host)[0] == 403


def test_matching_origin_required_for_post(server):
    port = server.server_port
    with patch.object(live, "STATE") as state:
        state.claim.return_value = "owner"
        for alias in ("localhost", "127.0.0.1"):
            status, _, _ = request(server, f"{alias}:{port}", "/api/session/start", "POST",
                b'{"session_id":"boundary"}', {"Origin": f"http://{alias}:{port}"})
            assert status == 200
        assert request(server, f"localhost:{port}", "/api/session/start", "POST",
            b'{"session_id":"boundary"}', {"Origin": "https://evil.example"})[0] == 403


def test_worklet_has_explicit_javascript_mime(server):
    from http.server import SimpleHTTPRequestHandler
    with patch.object(SimpleHTTPRequestHandler, "extensions_map", {".js": "text/plain"}):
        status, _, mime = request(server, f"localhost:{server.server_port}", "/live/audio-capture-worklet.js")
    assert status == 200 and mime == "text/javascript"


def test_invalid_rate_is_rejected_before_body_read(server):
    port = server.server_port
    status, data, _ = request(server, f"localhost:{port}", "/api/audio", "POST", b"",
        {"Origin": f"http://localhost:{port}", "Content-Type": "application/octet-stream",
         "Content-Length": "4", "X-Sample-Rate": "0", "X-Session-ID": "boundary",
         "X-Chunk-Sequence": "0"})
    assert status == 400 and b"sample rate" in data


def test_incomplete_audio_body_times_out(server):
    port = server.server_port
    status, _, _ = request(server, f"localhost:{port}", "/api/audio", "POST", b"",
        {"Origin": f"http://localhost:{port}", "Content-Type": "application/octet-stream",
         "Content-Length": "4", "X-Sample-Rate": "16000", "X-Session-ID": "boundary",
         "X-Chunk-Sequence": "0"})
    assert status == 400


def test_json_serializer_refuses_nonfinite_values():
    handler = object.__new__(live.Handler)
    with pytest.raises(ValueError):
        handler._json(200, {"score": float("nan")})
