import json
import hashlib
import shutil
import tempfile
import unittest
import sys
from dataclasses import asdict, replace
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dronedet.cache_metadata import file_sha256, sidecar_path
from dronedet.config import load_config, params_hash
from dronedet.features import (cache_split_power, cache_split_tabular, load_cache_index,
                               get_window, load_power_cache, load_tabular, mel_power,
                               standardize, tabular)

REPO = Path(__file__).resolve().parents[2]


class CacheSidecarTests(unittest.TestCase):
    def test_eval_windows_are_centered_and_rng_is_training_only(self):
        config = load_config(REPO / "config.yaml")
        cache = np.arange(64 * 100, dtype=np.float32).reshape(64, 100)
        row = pd.Series({"start_frame": 10, "n_frames": 40, "win_start_frame": 6})
        first = get_window(cache, row, config, split="val")
        second = get_window(cache, row, config, split="val")
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, cache[:, 16:44])
        with self.assertRaisesRegex(ValueError, "training only"):
            get_window(cache, row, config, np.random.default_rng(3), split="val")
        train1 = get_window(cache, row, config, np.random.default_rng(11), split="train")
        train2 = get_window(cache, row, config, np.random.default_rng(11), split="train")
        np.testing.assert_array_equal(train1, train2)
        selected_offsets = set()
        for seed in range(1000):
            window = get_window(cache, row, config, np.random.default_rng(seed), split="train")
            offset = int(np.flatnonzero(cache[0] == window[0, 0])[0]) - 10
            selected_offsets.add(offset)
        self.assertEqual(selected_offsets, set(range(13)))

    def test_invalid_window_spans_and_noncentered_eval_offsets_are_rejected(self):
        config = load_config(REPO / "config.yaml")
        cache = np.zeros((64, 100), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "canonical centre"):
            get_window(cache, pd.Series({"start_frame": 0, "n_frames": 40,
                                         "win_start_frame": 0}), config, split="val")
        with self.assertRaisesRegex(ValueError, "cache bounds"):
            get_window(cache, pd.Series({"start_frame": 70, "n_frames": 40,
                                         "win_start_frame": 6}), config, split="train")
    def test_missing_sidecar_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, cache = self._loader_paths(Path(folder))
            with self.assertRaisesRegex(ValueError, "sidecar missing"):
                load_power_cache(paths, "train")

    def test_tampered_sidecar_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, cache = self._loader_paths(Path(folder))
            sidecar_path(cache).write_text(json.dumps({
                "frontend_params_hash": "tampered",
                "train_manifest_sha256": hashlib.sha256(
                    (paths["splits"] / "train.csv").read_bytes()).hexdigest(),
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "parameter hash mismatch"):
                load_power_cache(paths, "train")

    def test_current_sidecar_is_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, cache = self._loader_paths(Path(folder))
            self._write_valid(paths)
            result = load_power_cache(paths, "train")
            self.assertEqual(result.shape, (64, 28))
            result._mmap.close()

    def test_d055_mode_and_decision_changes_do_not_invalidate_cached_content(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, _ = self._loader_paths(Path(folder))
            self._write_valid(paths)
            config_path = paths["root"] / "config.yaml"
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            raw["frontend"]["mode"] = "pcen"
            raw["frontend"]["pcen_gain"] = 0.7
            raw["decision"]["interval_s"] = 0.75
            raw["decision"]["smoothing_windows"] = 5
            config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            loaded = load_power_cache(paths, "train", mmap_mode=None)
            self.assertEqual(loaded.shape, (64, 28))
            self.assertEqual(load_tabular(paths, "train")["X"].shape, (1, 84))

    def test_power_and_tabular_writers_round_trip_without_renaming_or_stamping(self):
        for writer in ("wave", "power"):
            with self.subTest(writer=writer), tempfile.TemporaryDirectory() as folder:
                paths, cache = self._loader_paths(Path(folder))
                config = load_config(paths["root"] / "config.yaml")
                source = np.random.default_rng(19).normal(0, .1, 8000).astype(np.float32)
                expected_power = mel_power(standardize(source, 8000, config), config)
                if writer == "wave":
                    array, index = cache_split_power([("synthetic", source)], split="train",
                        output_dir=paths["spec"], config=config, paths=paths, source_rate=8000)
                    self.assertEqual(array.name, "mel_power_train.npy")
                    self.assertEqual(index.name, "index_train.csv")
                else:
                    archive = cache_split_tabular([("synthetic", expected_power)], split="train",
                        output_dir=paths["tab"], config=config, paths=paths)
                    self.assertEqual(archive.name, "train.npz")
                loaded = load_power_cache(paths, "train", mmap_mode=None)
                np.testing.assert_array_equal(loaded, expected_power)
                np.testing.assert_array_equal(load_tabular(paths, "train")["X"][0], tabular(expected_power, config))
                self.assertEqual(len(load_cache_index(paths, "train")), 1)
                self.assertTrue((paths["spec"] / "cache_train.commit.json").is_file())

    def test_nonfinite_and_negative_powers_are_rejected_even_with_consistent_checksums(self):
        for invalid in (np.nan, np.inf, -1.0):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as folder:
                paths, cache = self._loader_paths(Path(folder))
                self._write_valid(paths)
                values = np.load(cache, allow_pickle=False)
                values[0, 0] = invalid
                np.save(cache, values, allow_pickle=False)
                self._update_array_checksum(paths, cache)
                with self.assertRaisesRegex(ValueError, "non-finite or negative"):
                    load_power_cache(paths, "train")

    def test_index_bounds_and_center_offsets_are_rejected_after_checksum_verification(self):
        for field, value in (("start_frame", -1), ("n_frames", 29), ("win_start_frame", 1)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                paths, cache = self._loader_paths(Path(folder))
                self._write_valid(paths)
                path = paths["spec"] / "index_train.csv"
                index = pd.read_csv(path)
                index.loc[0, field] = value
                index.to_csv(path, index=False, lineterminator="\n")
                for metadata_path in (sidecar_path(cache), sidecar_path(paths["tab"] / "train.npz")):
                    metadata = json.loads(metadata_path.read_text())
                    metadata["index_sha256"] = file_sha256(path)
                    metadata_path.write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):
                    load_cache_index(paths, "train")

    def test_wrong_split_manifest_and_incomplete_transactions_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, cache = self._loader_paths(Path(folder))
            self._write_valid(paths)
            metadata_path = sidecar_path(cache)
            metadata = json.loads(metadata_path.read_text())
            metadata["split_manifest_sha256"] = "wrong"
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "split-manifest"):
                load_power_cache(paths, "train")
            metadata["split_manifest_sha256"] = file_sha256(paths["splits"] / "train.csv")
            metadata_path.write_text(json.dumps(metadata))
            (paths["tab"] / "train.npz").unlink()
            with self.assertRaisesRegex(ValueError, "transaction is incomplete"):
                load_power_cache(paths, "train")

    def test_transposed_cache_unknown_schema_and_missing_canonical_name_are_rejected(self):
        for change in ("transpose", "schema", "filename"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                paths, cache = self._loader_paths(Path(folder))
                self._write_valid(paths)
                if change == "transpose":
                    np.save(cache, np.load(cache, allow_pickle=False).T, allow_pickle=False)
                    self._update_array_checksum(paths, cache)
                elif change == "schema":
                    metadata_path = sidecar_path(cache)
                    metadata = json.loads(metadata_path.read_text())
                    metadata["schema_version"] = 999
                    metadata_path.write_text(json.dumps(metadata))
                else:
                    cache.rename(cache.with_name("train.npy"))
                with self.assertRaises(ValueError):
                    load_power_cache(paths, "train")

    def test_publication_failure_cleans_only_its_own_files(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, cache = self._loader_paths(Path(folder))
            sentinel = paths["spec"] / "owner.txt"
            sentinel.write_text("preserve me")
            import os
            real_link = os.link
            count = 0

            def failing_link(source, destination):
                nonlocal count
                count += 1
                if count == 3:
                    raise OSError("injected publication failure")
                real_link(source, destination)

            with patch("dronedet.cache_metadata.os.link", side_effect=failing_link):
                with self.assertRaisesRegex(OSError, "injected"):
                    self._write_valid(paths)
            self.assertFalse(cache.exists())
            self.assertFalse((paths["spec"] / "index_train.csv").exists())
            self.assertFalse((paths["spec"] / "cache_train.commit.json").exists())
            self.assertEqual(sentinel.read_text(), "preserve me")
            self.assertEqual(list(paths["spec"].glob("*.tmp")), [])

    def test_changed_feature_order_is_rejected_with_consistent_archive_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, cache = self._loader_paths(Path(folder))
            archive = self._write_valid(paths)
            with np.load(archive, allow_pickle=False) as source:
                values = {name: source[name] for name in source.files}
            values["feature_names"] = values["feature_names"][::-1]
            np.savez_compressed(archive, **values)
            self._update_array_checksum(paths, archive)
            with self.assertRaisesRegex(ValueError, "feature names"):
                load_tabular(paths, "train")

    def test_explicit_passband_override_is_recorded_and_unlogged_writes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, cache = self._loader_paths(Path(folder))
            config = replace(load_config(paths["root"] / "config.yaml"),
                             passband_status="interim", measured_passband_hz=None)
            original = yaml.safe_load((paths["root"] / "config.yaml").read_text())
            original["frontend"] = asdict(config)
            (paths["root"] / "config.yaml").write_text(yaml.safe_dump(original))
            source = np.random.default_rng(23).normal(0, .1, 16000).astype(np.float32)
            with self.assertRaisesRegex(ValueError, "measured.*override"):
                cache_split_power([("synthetic", source)], split="train", output_dir=paths["spec"],
                                  config=config, paths=paths, source_rate=16000)
            cache_split_power([("synthetic", source)], split="train", output_dir=paths["spec"],
                              config=config, paths=paths, source_rate=16000,
                              logged_override="synthetic compatibility fixture, not production")
            self.assertIn("synthetic", json.loads(sidecar_path(cache).read_text())["passband_override_reason"])
            load_power_cache(paths, "train", mmap_mode=None)

    def test_synthetic_test_split_requires_explicit_authorization_for_build_and_load(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, cache = self._loader_paths(Path(folder))
            config = load_config(paths["root"] / "config.yaml")
            manifest = pd.read_csv(paths["splits"] / "train.csv")
            manifest["split"] = "test"
            manifest.to_csv(paths["splits"] / "test.csv", index=False, lineterminator="\n")
            clips = [("synthetic", np.ones((64, 28), np.float32))]
            with self.assertRaises(PermissionError):
                cache_split_tabular(clips, split="test", output_dir=paths["tab"], config=config, paths=paths)
            cache_split_tabular(clips, split="test", output_dir=paths["tab"], config=config,
                                paths=paths, allow_test=True)
            with self.assertRaises(PermissionError):
                load_power_cache(paths, "test")
            loaded = load_power_cache(paths, "test", allow_test=True, mmap_mode=None)
            self.assertEqual(loaded.shape, (64, 28))

    @staticmethod
    def _update_array_checksum(paths, cache):
        metadata_path = sidecar_path(cache)
        metadata = json.loads(metadata_path.read_text())
        metadata["artifact_sha256"] = file_sha256(cache)
        metadata_path.write_text(json.dumps(metadata))
        marker_path = paths["spec"] / "cache_train.commit.json"
        marker = json.loads(marker_path.read_text())
        marker["artifact_sha256"][cache.name] = metadata["artifact_sha256"]
        marker_path.write_text(json.dumps(marker))

    @staticmethod
    def _write_valid(paths):
        config = load_config(paths["root"] / "config.yaml")
        return cache_split_tabular([("synthetic", np.ones((64, 28), np.float32))],
                    split="train", output_dir=paths["tab"], config=config, paths=paths)

    @staticmethod
    def _loader_paths(folder: Path):
        root = folder / "root"
        split = folder / "splits"
        spec = folder / "spec"
        tab = folder / "tab"
        root.mkdir()
        split.mkdir()
        spec.mkdir()
        tab.mkdir()
        shutil.copy2(REPO / "config.yaml", root / "config.yaml")
        pd.DataFrame([{"clip_id": "synthetic", "split": "train", "label": 0,
                       "group": "synthetic", "silent": False, "near_silent": False,
                       "partial_zero": False, "clipped_flag": False, "dc_flag": False}]).to_csv(
                           split / "train.csv", index=False, lineterminator="\n")
        cache = spec / "mel_power_train.npy"
        return {"root": root, "splits": split, "spec": spec, "tab": tab}, cache


if __name__ == "__main__":
    unittest.main()
