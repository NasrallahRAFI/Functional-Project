"""Train exactly one fixed V4 A0 CNN with seed 42; never open the test split."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE_ROOT.parents[1]
RUN_ID = "cnn_a0_seed42"
RUN_ROOT = PROJECT_ROOT / "runs" / "study-v4" / RUN_ID
CACHE_ROOT = PROJECT_ROOT / "storage" / "caches" / "corpus-002" / "processed"
SPLIT_ROOT = PROJECT_ROOT / "storage" / "splits" / "split-002"

sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from dronedet.config import load_config
from dronedet.features import (
    cache_split_power,
    fit_cached_normalizer,
    load_cache_index,
    load_power_cache,
    load_wave,
    to_db,
)
from dronedet.models.cnn import build_cnn, parameter_count
from dronedet.training import train_torch_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def cache_paths(run_root: Path) -> dict[str, Path]:
    return {
        "root": run_root,
        "splits": SPLIT_ROOT,
        "spec": CACHE_ROOT / "spectrograms",
        "tab": CACHE_ROOT / "features_tabular",
        "raw_root": PROJECT_ROOT / "storage" / "sources",
    }


def cache_one_split(split: str, config, paths: dict[str, Path]) -> tuple[Path, Path]:
    if split not in {"train", "val"}:
        raise ValueError("V4 single-model training may cache only train and validation")
    manifest = pd.read_csv(SPLIT_ROOT / f"{split}.csv", keep_default_na=False)
    if not manifest.split.eq(split).all():
        raise RuntimeError(f"{split} manifest contains a foreign split row")

    def clips():
        for row in manifest[["clip_id", "file"]].itertuples(index=False):
            yield str(row.clip_id), load_wave(Path(row.file), config)

    return cache_split_power(
        clips(), split=split, output_dir=paths["spec"], config=config,
        paths=paths, source_rate=config.sr, allow_test=False,
    )


def make_validation_tensor(power: np.ndarray, index: pd.DataFrame,
                           mean: np.ndarray, std: np.ndarray, config) -> np.ndarray:
    starts = index.start_frame.to_numpy(np.int64) + index.win_start_frame.to_numpy(np.int64)
    windows = np.stack([
        power[:, int(start):int(start) + config.n_win] for start in starts
    ]).astype(np.float32)
    return np.ascontiguousarray(
        ((to_db(windows, config) - mean[None, :, None]) / std[None, :, None])[:, None],
        dtype=np.float32,
    )


def train() -> None:
    resume_preparation = False
    if RUN_ROOT.exists():
        status_path = RUN_ROOT / "run_status.json"
        old_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        if (old_status.get("status") != "STOP"
                or any((RUN_ROOT / name).exists() for name in
                       ("cnn_v4_a0_seed42.pt", "history.json", "freeze.json"))):
            raise FileExistsError(f"refusing to start a second CNN run in {RUN_ROOT}")
        with (RUN_ROOT / "attempts.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(old_status, allow_nan=False) + "\n")
        resume_preparation = True
    else:
        if (CACHE_ROOT / "spectrograms").exists() or (CACHE_ROOT / "features_tabular").exists():
            raise FileExistsError(f"refusing to overwrite V4 cache artifacts at {CACHE_ROOT}")
        source_config = yaml.safe_load((WORKSPACE_ROOT / "config.yaml").read_text(encoding="utf-8"))
        run_config = json.loads(json.dumps(source_config))
        run_config["paths"] = {"local": {"raw_root": str(PROJECT_ROOT / "storage" / "sources")}}
        RUN_ROOT.mkdir(parents=True)
        (RUN_ROOT / "config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    write_json(RUN_ROOT / "run_status.json", {
        "status": "PREPARING_RESUME" if resume_preparation else "PREPARING",
        "run_id": RUN_ID, "seed": 42,
        "model_count": 1, "test_access": False,
    })

    config = load_config(RUN_ROOT / "config.yaml")
    paths = cache_paths(RUN_ROOT)
    for split in ("train", "val"):
        commit = paths["spec"] / f"cache_{split}.commit.json"
        if commit.is_file():
            print(f"reusing the complete canonical {split} cache from the preserved preparation attempt", flush=True)
        else:
            if resume_preparation:
                raise RuntimeError(f"cannot safely resume: {split} cache transaction is missing")
            print(f"building canonical {split} cache", flush=True)
            cache_one_split(split, config, paths)

    from dronedet.cache_metadata import file_sha256

    train_rows = pd.read_csv(SPLIT_ROOT / "train.csv", keep_default_na=False)
    val_rows = pd.read_csv(SPLIT_ROOT / "val.csv", keep_default_na=False)
    train_index = load_cache_index(paths, "train")
    val_index = load_cache_index(paths, "val")
    train_power = load_power_cache(paths, "train", mmap_mode="r")
    val_power = load_power_cache(paths, "val", mmap_mode="r")

    print("fitting D047 normalization on stats_eligible training rows only", flush=True)
    normalization_rows = train_rows.copy()
    # V4's ``file`` column is the absolute source audio path; cache spans and
    # the shared normalizer use clip_id as the power-map identity.
    normalization_rows["file"] = normalization_rows.clip_id
    normalizer = fit_cached_normalizer(
        train_power, train_index, normalization_rows.to_dict("records"), config
    )
    np.savez_compressed(
        paths["spec"] / "norm_stats_global.npz",
        mean=normalizer.mean.astype(np.float32),
        std=normalizer.std.astype(np.float32),
        clips_excluded=np.asarray(normalizer.excluded_count, dtype=np.int64),
    )
    normalization_record = {
        "fit_policy": "train_stats_eligible_per_band_floor_mask_v1",
        "fit_record": normalizer.fit_record,
        "excluded_clips": normalizer.excluded_count,
        "train_manifest_sha256": sha256(SPLIT_ROOT / "train.csv"),
        "normalization_file_sha256": sha256(paths["spec"] / "norm_stats_global.npz"),
        "test_audio_read": False,
    }
    write_json(paths["spec"] / "normalization.json", normalization_record)

    mean = normalizer.mean.astype(np.float32)
    std = normalizer.std.astype(np.float32)
    val_x = make_validation_tensor(val_power, val_index, mean, std, config)
    y_train = train_rows.label.to_numpy(np.float32)
    y_val = val_rows.label.to_numpy(np.float32)
    if val_x.shape != (len(val_rows), 1, config.n_mels, config.n_win):
        raise RuntimeError(f"validation shape mismatch: {val_x.shape}")
    if not np.isfinite(val_x).all():
        raise RuntimeError("validation tensor contains non-finite values")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = build_cnn(
        (1, config.n_mels, config.n_win), channels=(16, 32, 64),
        kernel_size=3, pool=(2, 2), head="global_average", dropout=0.3,
    )
    if parameter_count(model) != 23585:
        raise RuntimeError("D048 A0 CNN parameter count changed")

    starts = train_index.start_frame.to_numpy(np.int64)
    frame_counts = train_index.n_frames.to_numpy(np.int64)
    labels = train_rows.label.to_numpy(np.float32)

    def training_batch(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        selected_starts = np.asarray([
            starts[i] + rng.integers(0, frame_counts[i] - config.n_win + 1)
            for i in indices
        ], dtype=np.int64)
        windows = np.stack([
            train_power[:, int(start):int(start) + config.n_win]
            for start in selected_starts
        ]).astype(np.float32)
        # Power scales by the square of the waveform gain; this matches the
        # D049 random waveform-gain augmentation without clipping or per-clip normalization.
        gains_db = rng.uniform(-12.0, 12.0, len(indices)).astype(np.float32)
        windows *= np.power(10.0, gains_db / 10.0, dtype=np.float32)[:, None, None]
        db = to_db(windows, config)
        return np.ascontiguousarray(
            ((db - mean[None, :, None]) / std[None, :, None])[:, None],
            dtype=np.float32,
        )

    write_json(RUN_ROOT / "run_status.json", {
        "status": "TRAINING", "run_id": RUN_ID, "seed": seed,
        "model_count": 1, "test_access": False,
    })
    print("training one D048 A0 CNN (seed 42, CPU); validation is used only by the fixed D049 early-stop recipe", flush=True)
    started = time.perf_counter()
    result = train_torch_model(
        model, training_batch, val_x, y_train, y_val,
        seed=seed, batch_size=64, learning_rate=1e-3, weight_decay=1e-2,
        max_epochs=100, patience=8, device="cpu", expected_threads=1,
        lineage_hash=file_sha256(paths["spec"] / "mel_power_train.npy"),
    )
    model = result["model"].cpu().eval()
    checkpoint = RUN_ROOT / "cnn_v4_a0_seed42.pt"
    torch.save({
        "state_dict": result["state_dict"],
        "architecture": {"channels": [16, 32, 64], "kernel_size": 3,
                         "pool": [2, 2], "head": "global_average", "dropout": 0.3},
        "parameter_count": parameter_count(model),
        "input_shape": [1, config.n_mels, config.n_win],
        "seed": seed, "best_epoch": result["best_epoch"],
        "best_validation_loss": result["best_validation_loss"],
    }, checkpoint)
    (RUN_ROOT / "history.json").write_text(
        json.dumps(result["history"], indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    record = {
        "status": "FROZEN_BEFORE_TEST", "run_id": RUN_ID,
        "run_count": 1, "model_count": 1, "training_seed": seed,
        "architecture_decision": "D048 A0; selected before this single run; no architecture or hyperparameter search",
        "recipe": "D049 fixed CPU recipe, with owner's one-seed constraint",
        "architecture": {"channels": [16, 32, 64], "kernel_size": 3,
                         "pool": [2, 2], "head": "global_average", "dropout": 0.3},
        "parameter_count": parameter_count(model),
        "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-2,
        "loss": "BCEWithLogitsLoss with training-set positive weight",
        "batch_size": 64, "scheduler": "ReduceLROnPlateau factor=0.5 patience=3",
        "early_stopping": {"criterion": "validation loss", "patience": 8,
                           "maximum_epochs": 100, "restored_best_state": True},
        "training_window": "one fresh random frame-aligned 0.5s window per clip per epoch",
        "gain_augmentation_db": [-12.0, 12.0],
        "best_epoch": result["best_epoch"],
        "best_validation_loss": result["best_validation_loss"],
        "epochs_run": len(result["history"]),
        "training_seconds": time.perf_counter() - started,
        "torch_threads": torch.get_num_threads(),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "train_manifest_sha256": sha256(SPLIT_ROOT / "train.csv"),
        "validation_manifest_sha256": sha256(SPLIT_ROOT / "val.csv"),
        "test_manifest_opened": False, "test_audio_opened": False,
        "threshold_selected": False, "test_scored": False,
        "model_searches": 0, "training_runs": 1,
        "normalization_record": normalization_record,
        "cache_power_hashes": {
            split: sha256(paths["spec"] / f"mel_power_{split}.npy")
            for split in ("train", "val")
        },
    }
    write_json(RUN_ROOT / "freeze.json", record)
    write_json(RUN_ROOT / "run_status.json", {
        "status": "PASS", "run_id": RUN_ID, "seed": seed,
        "model_count": 1, "test_access": False,
    })
    print(json.dumps({"status": "PASS", "run_id": RUN_ID,
                      "best_epoch": result["best_epoch"],
                      "best_validation_loss": result["best_validation_loss"],
                      "epochs_run": len(result["history"]),
                      "checkpoint_sha256": record["checkpoint_sha256"]}, indent=2), flush=True)


if __name__ == "__main__":
    try:
        train()
    except BaseException as exc:
        if RUN_ROOT.exists():
            write_json(RUN_ROOT / "run_status.json", {
                "status": "STOP", "run_id": RUN_ID,
                "error_type": type(exc).__name__, "error": str(exc),
                "test_access": False,
            })
        raise
