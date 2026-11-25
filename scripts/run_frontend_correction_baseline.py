"""Capture W0 evidence without opening test audio or test array payloads.

This command records observations, including known failures. Run it before
corrections; existing baseline reports are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "demo"))

from dronedet.bundle import verify_bundle
from dronedet.config import cache_content_hash, config_hash, load_project_config, params_hash
from dronedet.data_preparation import locate_dataset_root
from dronedet.features import (cache_split_power, cache_split_tabular, file_to_power,
                               load_cache_index, load_power_cache,
                               load_tabular, mel_power, tabular, to_db)
import run_demo as demo


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments], text=True).strip()


def callback_gap_probe(config) -> dict:
    """Exercise the actual live callback while its 32-block queue is full."""
    real_queue = queue.Queue
    holder, windows = {}, []
    block_size = 240

    class EndCapture(Exception):
        pass

    class DrivenQueue(real_queue):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.next_block = 34

        def get(self, *args, **kwargs):
            if self.empty():
                raise EndCapture()
            item = super().get(*args, **kwargs)
            if self.next_block < 90:
                index = self.next_block
                self.next_block += 1
                holder["callback"](ramp(index), block_size, None, False)
            return item

    def ramp(index):
        return (np.arange(index * block_size, (index + 1) * block_size,
                          dtype=np.float32) / 100000)[:, None]

    class FakeStream:
        def __init__(self, **kwargs):
            holder["callback"] = kwargs["callback"]

        def __enter__(self):
            for index in range(34):
                holder["callback"](ramp(index), block_size, None, False)
            return self

        def __exit__(self, *args):
            return False

    class Recorder(demo.LiveWindowBuilder):
        def push(self, *args, **kwargs):
            emitted = super().push(*args, **kwargs)
            windows.extend(emitted)
            return emitted

    fake_device = types.SimpleNamespace(
        query_devices=lambda **kwargs: {"default_samplerate": 16000}, InputStream=FakeStream)
    with patch.dict(sys.modules, {"sounddevice": fake_device}), \
            patch.object(demo.queue, "Queue", DrivenQueue), \
            patch.object(demo, "LiveWindowBuilder", Recorder), \
            patch.object(demo.LOG, "warning"), patch.object(demo.LOG, "info"):
        try:
            demo.live_mode(config)
        except EndCapture:
            pass
    jumps = [float(np.diff(window).max()) for window in windows]
    return {"block_samples": block_size, "dropped_blocks": 2,
            "windows": len(windows), "windows_spanning_gap": sum(x > 2e-5 for x in jumps),
            "largest_adjacent_step": max(jumps, default=0), "expected_step": 1e-5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/interim/frontend_correction_v1_baseline.json")
    parser.add_argument("--sample-count", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"baseline already exists: {arguments.output}")
    if arguments.sample_count < 1:
        parser.error("sample count must be positive")
    project = load_project_config(ROOT / "config.yaml")
    paths, config = project["P"], project["fe"]
    packages = {}
    for name in ("numpy", "pandas", "scipy", "librosa", "soundfile", "soxr",
                 "sounddevice", "torch", "pytest", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    files = [ROOT / "config.yaml", *paths["splits"].glob("*.csv"),
             paths["splits"] / "FROZEN.txt", *paths["spec"].glob("*.json"),
             *paths["spec"].glob("index_*.csv"), *paths["spec"].glob("norm*.npz"),
             *paths["golden"].rglob("*"), *(ROOT / "results/models").rglob("*.pt"),
             *(ROOT / "results/models").rglob("metadata.json")]
    for split in ("train", "val"):
        files.extend((paths["spec"] / f"mel_power_{split}.npy", paths["tab"] / f"{split}.npz"))
    # Hashing manifests is allowed; never decode test labels or array payloads.
    artifacts = {str(path.relative_to(ROOT)): sha256(path)
                 for path in sorted(set(files)) if path.is_file()}
    with np.load(paths["spec"] / "norm_stats_global.npz", allow_pickle=False) as archive:
        stats = {"mean": archive["mean"].copy(), "std": archive["std"].copy()}
    test_metadata = {}
    for directory in (paths["spec"], paths["tab"]):
        for path in directory.glob("*test*.meta.json"):
            test_metadata[str(path.relative_to(ROOT))] = json.loads(path.read_text())
    consumers = {}
    tokens = ("config_hash", "params_hash", "cache_content_hash", "statistics_sha256",
              "lineage_hash", "frontend_params_hash", "norm_stats_global")
    for directory in ("src", "scripts", "demo"):
        for path in (ROOT / directory).rglob("*.py"):
            lines = path.read_text(encoding="utf-8-sig").splitlines()
            matches = [{"line": i, "text": line.strip()} for i, line in enumerate(lines, 1)
                       if any(token in line for token in tokens)]
            if matches:
                consumers[str(path.relative_to(ROOT))] = {"sha256": sha256(path), "matches": matches}
    test_command = [sys.executable, "-m", "pytest", "tests/unit", "-q"]
    print("Running W0 unit baseline...", flush=True)
    tests = subprocess.run(test_command, cwd=ROOT, capture_output=True, text=True)
    rng = np.random.default_rng(20261125)
    raw = locate_dataset_root(paths["raw_root"]) / "Binary_Drone_Audio"
    samples = {}
    train_power = train_index = None
    for split in ("train", "val"):
        print(f"Recomputing {arguments.sample_count} {split} clips...", flush=True)
        power = load_power_cache(paths, split)
        index = load_cache_index(paths, split)
        features = load_tabular(paths, split)["X"]
        rows = []
        for position in sorted(rng.choice(len(index), arguments.sample_count, replace=False)):
            row = index.iloc[int(position)]
            fresh, _ = file_to_power(raw / str(row.clip_id), config)
            start, count = int(row.start_frame), int(row.n_frames)
            previous = power[:, start:start + count]
            rows.append({"clip_id": str(row.clip_id), "position": int(position),
                         "start_frame": start, "n_frames": count,
                         "max_power_abs_difference": float(np.max(np.abs(fresh - previous))),
                         "max_tabular_abs_difference": float(np.max(np.abs(tabular(fresh, config) - features[int(position)])))})
        samples[split] = rows
        if split == "train":
            train_power, train_index = power, index
    # Deliberately reproduce the superseded notebook formula for diagnosis,
    # independently of the production helper corrected after this baseline.
    old_values = np.concatenate([to_db(train_power[:, int(row.start_frame):
        int(row.start_frame) + int(row.n_frames)], config)
        for row in train_index[train_index.stats_eligible].itertuples()], axis=1)
    mean = old_values.mean(axis=1, dtype=np.float64)
    std = old_values.std(axis=1, dtype=np.float64)
    frame_count = old_values.shape[1]
    probes = {"old_notebook_fit": {"max_mean_abs_difference": float(np.max(abs(mean - stats["mean"]))),
                                   "max_std_abs_difference": float(np.max(abs(std - stats["std"]))),
                                   "hash": config_hash(config, {"mean": mean, "std": std}),
                                   "frame_count": frame_count},
              "unaltered_bundle": verify_bundle(paths["golden"] / "frontend_bundle", config),
              "callback_gap": callback_gap_probe(config)}
    with tempfile.TemporaryDirectory(prefix="frontend_w0_") as temporary:
        temporary = Path(temporary)
        target = temporary / "bundle"
        shutil.copytree(paths["golden"] / "frontend_bundle", target)
        expected = target / "golden/golden16_db.npy"
        values = np.load(expected, allow_pickle=False)
        values.flat[0] = np.nan
        np.save(expected, values, allow_pickle=False)
        try:
            verify_bundle(target, config)
        except ValueError as exc:
            probes["nan_golden"] = {"accepted": False, "error": str(exc)}
        else:
            probes["nan_golden"] = {"accepted": True}
        audio = rng.normal(0, .1, config.win_samples * 2).astype(np.float32)
        try:
            array, index = cache_split_power([("synthetic", audio)], split="train",
                                             output_dir=temporary, config=config)
            tab = cache_split_tabular([("synthetic", mel_power(audio, config))], split="train",
                                      output_dir=temporary, config=config)
        except ValueError as exc:
            probes["writer_contract"] = {"legacy_call_rejected": True, "error": str(exc),
                "note": "canonical writer/loader round trips are checked by the correction suite"}
        else:
            with np.load(tab, allow_pickle=False) as archive:
                keys = archive.files
            probes["writer_contract"] = {"power_filename": array.name,
                                          "power_shape": list(np.load(array).shape),
                                          "index_filename": index.name, "tabular_filename": tab.name,
                                          "tabular_keys": keys,
                                          "has_power_sidecar": Path(str(array) + ".meta.json").exists()}
    report = {"stage": "W0_BASELINE", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "platform": platform.platform(), "packages": packages,
              "branch": git("branch", "--show-current"), "commit": git("rev-parse", "HEAD"),
              "remotes": git("remote", "-v"), "status": git("status", "--porcelain"),
              "freeze_commit": git("rev-parse", "frontend-core-frozen^{commit}"),
              "resolved_config": config.__dict__, "n_win": config.n_win,
              "win_samples": config.win_samples, "parameters_hash": params_hash(config),
              "statistics_bearing_hash": config_hash(config, stats),
              "power_content_hash": cache_content_hash(config),
              "installed_stats": {key: {"values": value.tolist(), "shape": list(value.shape),
                                         "dtype": str(value.dtype)} for key, value in stats.items()},
              "artifact_sha256": artifacts, "test_sidecars_metadata_only": test_metadata,
              "identity_consumers": consumers, "unit_baseline": {"command": test_command,
                   "returncode": tests.returncode, "stdout": tests.stdout, "stderr": tests.stderr},
              "sample_seed": 20261125, "recomputations": samples, "probes": probes,
              "test_audio_or_array_payloads_opened": False}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"record": str(arguments.output), "unit_returncode": tests.returncode,
                      "parameters_hash": report["parameters_hash"], "probes": probes}, indent=2))
    if tests.returncode:
        raise SystemExit(tests.returncode)


if __name__ == "__main__":
    main()
