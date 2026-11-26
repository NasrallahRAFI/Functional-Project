"""Start and supervise the local CNN service, opening the browser only when ready."""
from __future__ import annotations

import json
import ctypes
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from frontend_identity import FRONTEND_FILES, frontend_sha256, frontend_sha256_from_sources

BASE_URL = "http://127.0.0.1:8766"
URL = f"{BASE_URL}/live/index.html"
EXPECTED_MODEL = json.loads((ROOT / "config/live_detector.json").read_text(encoding="utf-8"))
EXPECTED_FRONTEND_SHA256 = frontend_sha256(ROOT / "live")


def attach_kill_on_close_job(process: subprocess.Popen) -> object:
    """Tie the owned child lifetime to this launcher process on Windows."""
    if os.name != "nt":
        return None
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = ctypes.c_void_p
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())

    class BasicLimit(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32),
                    ("SchedulingClass", ctypes.c_uint32)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in
                    ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                     "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class ExtendedLimit(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BasicLimit), ("IoInfo", IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    limits = ExtendedLimit()
    limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        kernel.CloseHandle(job)
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.AssignProcessToJobObject(job, ctypes.c_void_p(process._handle)):
        kernel.CloseHandle(job)
        raise ctypes.WinError(ctypes.get_last_error())
    return job
def health() -> dict | None:
    try:
        with urllib.request.urlopen(BASE_URL + "/api/health", timeout=1.5) as response:
            if response.status != 200:
                return None
            return json.loads(response.read(16_384))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def frontend_matches(value: dict | None) -> bool:
    if not value:
        return False
    reported = value.get("frontend_sha256")
    if reported is not None:
        return reported == EXPECTED_FRONTEND_SHA256
    # Older same-CNN servers may not report a build digest. Hash their actual
    # served files before reuse; cache headers are bypassed by urllib here.
    try:
        sources = {}
        for name in FRONTEND_FILES:
            with urllib.request.urlopen(f"{BASE_URL}/live/{name}", timeout=2) as response:
                if response.status != 200:
                    return False
                sources[name] = response.read()
        return frontend_sha256_from_sources(sources) == EXPECTED_FRONTEND_SHA256
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def is_ours(value: dict | None) -> bool:
    return bool(value and value.get("status") == "ready"
                and value.get("protocol_version") == 2
                and (value.get("instance_id") is None or isinstance(value.get("instance_id"), str))
                and value.get("application_id") == "local-drone-detector"
                and value.get("model_id") == EXPECTED_MODEL["model_id"]
                and value.get("model_version") == EXPECTED_MODEL["model_version"]
                and value.get("variant") == EXPECTED_MODEL["variant"]
                and value.get("seed") == EXPECTED_MODEL["seed"]
                and value.get("checkpoint_sha256") == EXPECTED_MODEL["model_sha256"]
                and frontend_matches(value)
                and value.get("threshold") == EXPECTED_MODEL["threshold"]
                and value.get("score_kind") == "raw_logit"
                and value.get("policy_id") == EXPECTED_MODEL["policy_id"]
                and value.get("stage", EXPECTED_MODEL.get("stage")) == EXPECTED_MODEL.get("stage", value.get("stage"))
                and value.get("validation_eligible", True) == EXPECTED_MODEL.get("validation_eligible", True)
                and value.get("experimental_activation", False) == EXPECTED_MODEL.get("experimental_activation", False)
                and value.get("validation_guard_failures", []) == EXPECTED_MODEL.get("validation_guard_failures", [])
                and value.get("test_access", False) == EXPECTED_MODEL.get("test_access", False))


def open_monitor() -> None:
    try:
        opened = webbrowser.open(URL, new=2)
    except webbrowser.Error:
        opened = False
    if not opened:
        print(f"Open the local monitor manually: {URL}")


def port_is_listening() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 8766), timeout=0.25):
            return True
    except OSError:
        return False


def main() -> int:
    existing = health()
    if existing is not None:
        if not is_ours(existing):
            same_cnn = (existing.get("application_id") == "local-drone-detector"
                        and existing.get("model_id") == EXPECTED_MODEL["model_id"]
                        and existing.get("checkpoint_sha256") == EXPECTED_MODEL["model_sha256"])
            if same_cnn and existing.get("frontend_sha256") != EXPECTED_FRONTEND_SHA256:
                print("A V4 service with a different or unverifiable frontend build is already running on port 8766. Close that service, then run start_detector.bat again.", file=sys.stderr)
            else:
                print("Port 8766 is serving another application or CNN build. Close it and retry.", file=sys.stderr)
            return 2
        print("The verified CNN service is already running.")
        open_monitor()
        return 0
    if port_is_listening():
        print("Port 8766 is occupied by an unrecognized application. Close it and retry.", file=sys.stderr)
        return 2

    instance_id = uuid.uuid4().hex
    child = subprocess.Popen([sys.executable, str(ROOT / "scripts/live_detector.py"),
                              "--instance-id", instance_id], cwd=ROOT)
    try:
        job = attach_kill_on_close_job(child)
    except Exception:
        child.terminate()
        child.wait()
        raise
    deadline = time.monotonic() + 90
    try:
        while time.monotonic() < deadline:
            if child.poll() is not None:
                print(f"Detector exited during startup with code {child.returncode}.", file=sys.stderr)
                return child.returncode or 1
            value = health()
            if value is not None:
                if not is_ours(value) or value.get("instance_id") != instance_id:
                    print("Port 8766 became occupied by another application.", file=sys.stderr)
                    return 2
                print("Verified CNN is ready; opening the local monitor.")
                open_monitor()
                return child.wait()
            time.sleep(0.4)
        print("Detector did not become ready within 90 seconds.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if os.name == "nt" and job:
            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(job)


if __name__ == "__main__":
    raise SystemExit(main())
