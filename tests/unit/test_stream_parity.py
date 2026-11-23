import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "demo"))

from dronedet.config import load_config  # noqa: E402
from run_demo import check_stream_parity  # noqa: E402


class StreamingParityTests(unittest.TestCase):
    def test_g14_t1_t2_t3(self):
        result = check_stream_parity(load_config(ROOT / "config.yaml"))
        self.assertEqual(result["status"], "PASS", result)
        self.assertTrue(result["t1_16khz"]["exact"])
        self.assertTrue(all(item["pass"] for item in result["t2_48khz"].values()))
        self.assertTrue(result["t3_overflow_recovery"]["pass"])
        self.assertEqual(result["t3_overflow_recovery"]["fresh_samples_before_first_window"],
                         load_config(ROOT / "config.yaml").win_samples)


if __name__ == "__main__":
    unittest.main()
