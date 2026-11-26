from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import call, patch

import scripts.launch_detector as launcher


def ready_health():
    selected = launcher.EXPECTED_MODEL
    return {
        "status": "ready", "protocol_version": 2,
        "instance_id": "test-instance",
        "application_id": "local-drone-detector", "model_id": selected["model_id"],
        "model_version": selected["model_version"], "stage": selected.get("stage", "V2"),
        "variant": selected["variant"],
        "seed": selected["seed"], "threshold": selected["threshold"],
        "validation_eligible": selected.get("validation_eligible", True),
        "experimental_activation": selected.get("experimental_activation", False),
        "validation_guard_failures": selected.get("validation_guard_failures", []),
        "test_access": selected.get("test_access", False),
        "checkpoint_sha256": launcher.EXPECTED_MODEL["model_sha256"], "score_kind": "raw_logit",
    }


class LauncherFlowTests(unittest.TestCase):
    @patch.object(launcher, "open_monitor")
    @patch.object(launcher, "health", return_value=ready_health())
    def test_matching_running_service_opens_without_spawning_a_duplicate(self, _health, open_monitor):
        with patch.object(launcher.subprocess, "Popen") as popen:
            self.assertEqual(launcher.main(), 0)
        open_monitor.assert_called_once_with()
        popen.assert_not_called()

    @patch.object(launcher, "open_monitor")
    @patch.object(launcher, "health", return_value={"application_id": "unrelated"})
    def test_port_collision_does_not_open_or_spawn(self, _health, open_monitor):
        with patch.object(launcher.subprocess, "Popen") as popen:
            self.assertEqual(launcher.main(), 2)
        open_monitor.assert_not_called()
        popen.assert_not_called()

    def test_new_service_opens_only_after_matching_health(self):
        events = []

        class Child:
            def __init__(self): self.returncode = 0; self.finished = False
            def poll(self): return self.returncode if self.finished else None
            def wait(self, timeout=None): events.append("wait"); self.finished = True; return 0
            def terminate(self): self.returncode = -15; self.finished = True
            def kill(self): self.returncode = -9; self.finished = True

        with patch.object(launcher, "health", side_effect=[None, ready_health()]), \
             patch.object(launcher.subprocess, "Popen", return_value=Child()) as popen, \
             patch.object(launcher, "port_is_listening", return_value=False), \
             patch.object(launcher, "attach_kill_on_close_job", return_value=None), \
             patch.object(launcher, "open_monitor", side_effect=lambda: events.append("open")), \
             patch.object(launcher.uuid, "uuid4", return_value=SimpleNamespace(hex="test-instance")), \
             patch.object(launcher.time, "sleep"):
            self.assertEqual(launcher.main(), 0)
        self.assertEqual(events, ["open", "wait"])
        self.assertEqual(popen.call_args.args[0][0], launcher.sys.executable)
        self.assertFalse(popen.call_args.kwargs.get("shell", False))

    def test_unrecognized_listener_is_reported_without_starting_child(self):
        with patch.object(launcher, "health", return_value=None), \
             patch.object(launcher, "port_is_listening", return_value=True), \
             patch.object(launcher.subprocess, "Popen") as popen:
            self.assertEqual(launcher.main(), 2)
        popen.assert_not_called()

    def test_wrong_checkpoint_identity_is_not_attached(self):
        health = ready_health()
        health["checkpoint_sha256"] = "0" * 64
        self.assertFalse(launcher.is_ours(health))

    def test_different_model_or_experimental_status_is_not_attached(self):
        health = ready_health()
        health["model_id"] = "unselected-model"
        self.assertFalse(launcher.is_ours(health))
        health = ready_health()
        health["experimental_activation"] = not health["experimental_activation"]
        self.assertFalse(launcher.is_ours(health))

    def test_failed_browser_open_prints_the_local_url(self):
        output = StringIO()
        with patch.object(launcher.webbrowser, "open", return_value=False), redirect_stdout(output):
            launcher.open_monitor()
        self.assertIn(launcher.URL, output.getvalue())


if __name__ == "__main__":
    unittest.main()
