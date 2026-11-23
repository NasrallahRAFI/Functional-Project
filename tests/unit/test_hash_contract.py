import copy
import json
import unittest
from dataclasses import fields
from pathlib import Path

import numpy as np

from dronedet.config import config_hash, load_config, params_hash


ROOT = Path(__file__).resolve().parents[2]


class FrontendHashContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "config.yaml")

    @staticmethod
    def _changed(value):
        if value is None:
            return "changed"
        if isinstance(value, bool):
            return not value
        if isinstance(value, str):
            return value + "_changed"
        if isinstance(value, tuple):
            return (value[0] + 0.25, value[1])
        if isinstance(value, int):
            return value + 1
        if isinstance(value, float):
            return value + 0.125
        raise AssertionError(f"No mutation fixture for {type(value)}")

    def test_every_frontend_field_changes_config_hash(self):
        baseline = config_hash(self.config)
        changed_fields = []
        for field in fields(self.config):
            altered = copy.copy(self.config)
            object.__setattr__(altered, field.name, self._changed(getattr(self.config, field.name)))
            self.assertNotEqual(baseline, config_hash(altered), field.name)
            changed_fields.append(field.name)
        self.assertTrue({"pcen_time_constant", "pcen_gain", "pcen_bias", "pcen_power",
                         "pcen_eps", "pcen_input_scale"}.issubset(changed_fields))

    def test_masked_statistics_change_bundle_hash_but_not_parameter_hash(self):
        with np.load(ROOT / "data/processed/spectrograms/norm_stats_global.npz",
                     allow_pickle=False) as archive:
            stats = {"mean": archive["mean"].copy(), "std": archive["std"].copy()}
        baseline_params = params_hash(self.config)
        baseline_bundle = config_hash(self.config, stats)
        changed = {key: value.copy() for key, value in stats.items()}
        changed["mean"][0] = np.nextafter(changed["mean"][0], np.float32(np.inf))
        self.assertNotEqual(baseline_bundle, config_hash(self.config, changed))
        self.assertEqual(baseline_params, params_hash(self.config))

    def test_statistics_hash_uses_explicit_little_endian_float32(self):
        values = np.asarray([1.25, -2.5], dtype=np.float64)
        expected = config_hash(self.config, {"mean": values})
        self.assertEqual(expected, config_hash(self.config, {"mean": values.astype("<f4")}))

    def test_canonical_hash_is_stable(self):
        self.assertEqual(config_hash(self.config), config_hash(copy.copy(self.config)))
        payload = self.config.__dict__
        self.assertEqual(json.dumps(payload, sort_keys=True, separators=(",", ":")),
                         json.dumps(dict(reversed(list(payload.items()))), sort_keys=True,
                                    separators=(",", ":")))


if __name__ == "__main__":
    unittest.main()
