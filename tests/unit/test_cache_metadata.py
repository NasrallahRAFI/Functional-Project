import json
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from dronedet.cache_metadata import sidecar_path
from dronedet.config import load_config, params_hash
from dronedet.features import load_power_cache

REPO = Path(__file__).resolve().parents[2]


class CacheSidecarTests(unittest.TestCase):
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
            import numpy as np
            np.save(cache, np.zeros((64, 1), dtype=np.float32), allow_pickle=False)
            train_sha = hashlib.sha256((paths["splits"] / "train.csv").read_bytes()).hexdigest()
            sidecar_path(cache).write_text(json.dumps({
                "frontend_params_hash": params_hash(load_config(paths["root"] / "config.yaml")),
                "train_manifest_sha256": train_sha,
                "array_shape": [64, 1], "array_dtype": "float32",
            }), encoding="utf-8")
            result = load_power_cache(paths, "train")
            self.assertEqual(result.shape, (64, 1))
            result._mmap.close()

    @staticmethod
    def _loader_paths(folder: Path):
        root = folder / "root"
        split = folder / "splits"
        spec = folder / "spec"
        root.mkdir()
        split.mkdir()
        spec.mkdir()
        shutil.copy2(REPO / "config.yaml", root / "config.yaml")
        (split / "train.csv").write_text("clip_id\nsynthetic\n", encoding="utf-8")
        cache = spec / "mel_power_train.npy"
        return {"root": root, "splits": split, "spec": spec}, cache


if __name__ == "__main__":
    unittest.main()
