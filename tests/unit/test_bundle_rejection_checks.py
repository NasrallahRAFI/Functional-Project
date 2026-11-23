import unittest
from pathlib import Path

from dronedet.bundle import run_bundle_rejection_checks
from dronedet.config import load_config


ROOT = Path(__file__).resolve().parents[2]


class BundleTamperContractTests(unittest.TestCase):
    def test_re_stamped_bundle_rejects_tampering(self):
        bundle = ROOT / "data/processed/golden/frontend_bundle"
        result = run_bundle_rejection_checks(bundle, load_config(ROOT / "config.yaml"))
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(value.get("pass", True) for value in result.values()
                            if isinstance(value, dict)))


if __name__ == "__main__":
    unittest.main()
