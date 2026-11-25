import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

from dronedet.config import (cache_content_hash, config_hash, config_from_mapping,
                             decision_hash, legacy_config_hash, load_config,
                             normalization_lineage_hash, params_hash,
                             tabular_content_hash)
from dronedet.features import tabular
from scripts.run_frontend_closeout import git_is_ancestor, stale_bound_files


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

    def test_hash_fields_follow_their_declared_domains(self):
        power_fields = {"sr", "channel", "n_fft", "win_length", "hop", "window", "center",
                        "n_mels", "fmin", "fmax", "mel_scale", "mel_norm", "power", "L",
                        "resample_quality"}
        decision_fields = {"decision_interval_s", "smoothing_windows"}
        for field in fields(self.config):
            altered = copy.copy(self.config)
            object.__setattr__(altered, field.name, self._changed(getattr(self.config, field.name)))
            if field.name in power_fields:
                self.assertNotEqual(cache_content_hash(self.config), cache_content_hash(altered), field.name)
                self.assertNotEqual(tabular_content_hash(self.config), tabular_content_hash(altered), field.name)
            else:
                self.assertEqual(cache_content_hash(self.config), cache_content_hash(altered), field.name)
                self.assertEqual(tabular_content_hash(self.config), tabular_content_hash(altered), field.name)
            if field.name in decision_fields:
                self.assertEqual(params_hash(self.config), params_hash(altered), field.name)
                self.assertNotEqual(decision_hash(self.config), decision_hash(altered), field.name)
            elif field.name in {"passband_status", "measured_passband_hz", "passband_override_reason",
                                "manifest_seed"}:
                self.assertEqual(params_hash(self.config), params_hash(altered), field.name)
            else:
                self.assertNotEqual(params_hash(self.config), params_hash(altered), field.name)

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
        expected = legacy_config_hash(self.config, {"mean": values})
        self.assertEqual(expected, legacy_config_hash(self.config, {"mean": values.astype("<f4")}))

    def test_canonical_hash_is_stable(self):
        self.assertEqual(config_hash(self.config), config_hash(copy.copy(self.config)))
        payload = self.config.__dict__
        self.assertEqual(json.dumps(payload, sort_keys=True, separators=(",", ":")),
                         json.dumps(dict(reversed(list(payload.items()))), sort_keys=True,
                                    separators=(",", ":")))

    def test_numpy_scalars_match_python_scalars(self):
        altered = copy.copy(self.config)
        for name in ("sr", "n_fft", "hop", "n_mels", "manifest_seed"):
            object.__setattr__(altered, name, np.int64(getattr(self.config, name)))
        for name in ("fmin", "fmax", "power", "pcen_gain", "pcen_input_scale"):
            object.__setattr__(altered, name, np.float64(getattr(self.config, name)))
        object.__setattr__(altered, "gain_range_db", tuple(np.float64(x) for x in self.config.gain_range_db))
        self.assertEqual(config_hash(self.config), config_hash(altered))

    def test_frontend_parser_rejects_unknown_and_conflicting_keys(self):
        with self.assertRaisesRegex(ValueError, "unknown frontend keys"):
            config_from_mapping({"frontned": 2})
        with self.assertRaisesRegex(ValueError, "conflicting frontend aliases"):
            config_from_mapping({"sr": 16000, "sample_rate": 16000})

    def test_legacy_identity_survives_decision_section_migration(self):
        raw = __import__("yaml").safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        moved = copy.deepcopy(raw)
        moved["frontend"]["decision_interval_s"] = moved["frontend"].pop("decision_interval_s", 0.5)
        moved["frontend"]["smoothing_windows"] = moved["frontend"].pop("smoothing_windows", 3)
        cfg = config_from_mapping(moved)
        self.assertEqual(legacy_config_hash(self.config), legacy_config_hash(cfg))
        self.assertEqual(0.5, cfg.decision_interval_s)
        self.assertEqual(3, cfg.smoothing_windows)

    def test_statistics_lineage_has_domain_separator_and_policy(self):
        stats = {"mean": np.zeros(64, dtype=np.float32), "std": np.ones(64, dtype=np.float32)}
        self.assertNotEqual(normalization_lineage_hash(self.config, stats),
                            normalization_lineage_hash(self.config, stats, fit_policy="other"))

    def test_decision_identity_binds_selected_threshold(self):
        self.assertNotEqual(decision_hash(self.config, threshold=2.75),
                            decision_hash(self.config, threshold=2.8))

    def test_nonfinite_and_invalid_discrete_values_fail_at_config_load(self):
        for field, value in (("L", float("nan")), ("pcen_input_scale", 0),
                             ("pcen_gain", 1.2), ("pcen_power", 1.1),
                             ("smoothing_windows", 2.5), ("n_mels", 16.0),
                             ("decision_interval_s", float("inf"))):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(self.config, **{field: value})

    def test_tabular_capability_does_not_reject_a_valid_cnn_transform(self):
        cnn_only = replace(self.config, n_mels=16)
        self.assertGreater(cnn_only.validate_filterbank()["minimum_nonzero_bins"], 0)
        with self.assertRaisesRegex(ValueError, "84-column tabular"):
            tabular(np.ones((16, cnn_only.n_win), dtype=np.float32), cnn_only)

    def test_closeout_detects_stale_and_missing_bound_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "contract.md"
            target.write_text("approved", encoding="utf-8")
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            self.assertEqual([], stale_bound_files(root, {"contract.md": digest}))
            target.write_text("changed", encoding="utf-8")
            self.assertEqual(["contract.md"], stale_bound_files(root, {"contract.md": digest}))
            target.unlink()
            self.assertEqual(["contract.md"], stale_bound_files(root, {"contract.md": digest}))

    def test_closeout_accepts_unrelated_descendants_without_moving_freeze_tag(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)

            def git(*args: str) -> str:
                result = subprocess.run(["git", "-C", str(repo), *args], check=True,
                                        capture_output=True, text=True)
                return result.stdout.strip()

            git("init", "-b", "main")
            git("config", "user.name", "Test")
            git("config", "user.email", "test@example.invalid")
            (repo / "first.txt").write_text("first", encoding="utf-8")
            git("add", "first.txt")
            git("commit", "-m", "freeze")
            frozen = git("rev-parse", "HEAD")
            (repo / "second.txt").write_text("descendant", encoding="utf-8")
            git("add", "second.txt")
            git("commit", "-m", "unrelated later change")
            descendant = git("rev-parse", "HEAD")
            self.assertTrue(git_is_ancestor(repo, frozen, descendant))

            git("checkout", "--orphan", "unrelated")
            git("rm", "-rf", ".")
            (repo / "other.txt").write_text("other history", encoding="utf-8")
            git("add", "other.txt")
            git("commit", "-m", "independent history")
            independent = git("rev-parse", "HEAD")
            self.assertFalse(git_is_ancestor(repo, frozen, independent))


if __name__ == "__main__":
    unittest.main()
