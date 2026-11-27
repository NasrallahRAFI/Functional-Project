"""Exercise the legacy Windows entry point without starting a model or browser."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.name == "nt", "Windows batch routing")
class LegacyEntrypointTests(unittest.TestCase):
    def run_entrypoint(self, target_exit=None):
        with tempfile.TemporaryDirectory(prefix="detector routing ") as directory:
            root = Path(directory)
            (root / "project.layout.json").write_text("{}", encoding="ascii")
            legacy = root / "Functional Project"
            legacy.mkdir()
            source = Path(__file__).resolve().parents[2] / "start_detector.bat"
            entrypoint = legacy / source.name
            entrypoint.write_bytes(source.read_bytes())
            if target_exit is not None:
                target = root / "workspaces/study-v4/start_detector.bat"
                target.parent.mkdir(parents=True)
                target.write_text(
                    f"@echo off\necho V4_TARGET_REACHED\nexit /b {target_exit}\n",
                    encoding="ascii",
                )
            return subprocess.run(
                [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", str(entrypoint)],
                cwd=root, capture_output=True, text=True, timeout=10,
            )

    def test_legacy_path_calls_v4_from_a_different_working_directory(self):
        result = self.run_entrypoint(0)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("V4_TARGET_REACHED", result.stdout)
        self.assertIn("8766/live/index.html", result.stdout)

    def test_v4_failure_is_propagated(self):
        result = self.run_entrypoint(7)
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertIn("V4_TARGET_REACHED", result.stdout)

    def test_missing_v4_fails_without_falling_back_to_v3(self):
        result = self.run_entrypoint()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("V4 launcher is missing", result.stdout)
        self.assertNotIn("Python 3 is required", result.stdout)


if __name__ == "__main__":
    unittest.main()
