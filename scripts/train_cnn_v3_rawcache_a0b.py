"""Run A0b from a raw-WAV rebuilt full-clip cache and the matched V2 recipe."""
from __future__ import annotations

import argparse
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
from torch import nn
import yaml

from dronedet.config import config_from_mapping
from dronedet.features import to_db
from dronedet.metrics import choose_f1_threshold, metrics_at_threshold, operating_threshold
from dronedet.models.cnn import build_cnn, parameter_count


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def make_tensors(power: np.ndarray, starts: np.ndarray, means: np.ndarray, stds: np.ndarray,
                 fe, gains: np.ndarray | None = None) -> np.ndarray:
    batch = np.stack([power[:, int(s):int(s) + fe.n_win] for s in starts]).astype(np.float32)
    if gains is not None:
        batch *= np.power(10.0, gains / 10.0).astype(np.float32)[:, None, None]
    db = to_db(batch, fe)
    batch = (db - means[None, :, None]) / stds[None, :, None]
    return np.ascontiguousarray(batch[:, None, :, :], dtype=np.float32)


def train(config_path: Path) -> None:
    cp = config_path.resolve()
    doc = yaml.safe_load(cp.read_text(encoding="utf-8"))
    fe = config_from_mapping(doc["frontend"])
    root = cp.parent.parent
    data = root / doc["output_root"] / "data"
    out = root / "results" / "cnn_v3" / "experiments" / "A0b_rawcache_seed42"
    ready = json.loads((data / "DATASET_READY.json").read_text(encoding="utf-8"))
    if ready.get("status") != "READY" or ready.get("representation") != "cnn_mel_power":
        raise RuntimeError("CNN V2 dataset is not ready")
    sys.path.insert(0, str(root))
    from scripts.prepare_cnn_v2 import verify as verify_dataset
    verify_dataset(cp)
    if out.exists() and not (out / "checkpoint_epoch.pt").is_file():
        if any(out.iterdir()):
            raise FileExistsError(f"training output already exists without a resumable checkpoint: {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(doc["threads"]))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(42); np.random.seed(42); random.seed(42)

    train_manifest = pd.read_csv(data / "manifests/train.csv", keep_default_na=False)
    val_manifest = pd.read_csv(data / "manifests/val.csv", keep_default_na=False)
    train_index = pd.read_csv(data / "cache/index_train.csv")
    val_index = pd.read_csv(data / "cache/index_val.csv")
    raw_cache_path = root / "results/cnn_v3/prepared/raw_power_train.npy"
    raw_cache_report = json.loads((raw_cache_path.parent / "raw_train_cache_report.json").read_text(encoding="utf-8"))
    if raw_cache_report.get("status") != "PASS" or not raw_cache_report.get("bitwise_equal"):
        raise RuntimeError("raw-derived full-clip train power cache failed frozen V2 parity")
    train_power = np.load(raw_cache_path, mmap_mode="r", allow_pickle=False)
    val_power = np.load(data / "cache/mel_power_val.npy", mmap_mode="r", allow_pickle=False)
    stats = np.load(data / "normalization_stats.npz", allow_pickle=False)
    mean, std = stats["mean"].astype(np.float32), stats["std"].astype(np.float32)
    y_train = train_manifest.label.to_numpy(dtype=np.float32)
    y_val = val_manifest.label.to_numpy(dtype=np.float32)
    if len(train_manifest) != 7607 or len(val_manifest) != 1510 or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise RuntimeError("CNN V2 training data or statistics failed the frozen contract")
    val_starts = val_index.center_start_frame.to_numpy(dtype=np.int64) + val_index.start_frame.to_numpy(dtype=np.int64)
    val_x = make_tensors(val_power, val_starts, mean, std, fe)
    if val_x.shape != (1510, 1, 64, 28) or not np.isfinite(val_x).all():
        raise RuntimeError("validation tensors are invalid")

    model = build_cnn((1, 64, 28), channels=(16, 32, 64, 64), kernel_size=3,
                      pool=(2, 2), head="global_average", dropout=0.3)
    if parameter_count(model) != 60641:
        raise RuntimeError("CNN V2 architecture parameter count changed")
    n_pos, n_neg = int((y_train == 1).sum()), int((y_train == 0).sum())
    if (n_pos, n_neg) != (967, 6640):
        raise RuntimeError("train class counts do not match approved split")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    epoch_limit, patience = 100, 8
    state_path = out / "checkpoint_epoch.pt"
    history_path = out / "history.json"
    history: list[dict] = []
    best_loss, best_epoch, stale, start_epoch = float("inf"), 0, 0, 0
    best_state = None
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("config_sha256") != sha256(cp) or state.get("dataset_ready_sha256") != sha256(data / "DATASET_READY.json"):
            raise RuntimeError("checkpoint identities differ; refusing an unsafe resume")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"]); np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"])
        history = state["history"]; best_loss = state["best_loss"]
        best_epoch = state["best_epoch"]; stale = state["stale"]
        best_state = state["best_state"]; start_epoch = state["next_epoch"]
        print(f"Resuming the same CNN V2 run at epoch {start_epoch + 1}.", flush=True)

    started = time.perf_counter()
    stop_reason = "maximum_epochs"
    for epoch in range(start_epoch, epoch_limit):
        model.train()
        epoch_seed = 42 * 1_000_003 + epoch
        rng, order_rng = np.random.default_rng(epoch_seed), np.random.default_rng(epoch_seed ^ 0x5EED)
        order = order_rng.permutation(len(y_train))
        loss_sum = 0.0
        for offset in range(0, len(order), 64):
            ix = order[offset:offset + 64]
            offsets = np.array([int(rng.integers(0, int(r["n_frames"]) - fe.n_win + 1))
                                for r in train_index.iloc[ix].to_dict("records")], dtype=np.int64)
            starts = train_index.iloc[ix].start_frame.to_numpy(dtype=np.int64) + offsets
            gains = rng.uniform(-12.0, 12.0, len(ix))
            xb = torch.from_numpy(make_tensors(train_power, starts, mean, std, fe, gains))
            yb = torch.from_numpy(y_train[ix])
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb); loss = criterion(logits, yb); loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * len(ix)
        train_loss = loss_sum / len(y_train)
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(val_x)).cpu().numpy().astype(np.float64)
            val_loss = float(criterion(torch.from_numpy(val_logits.astype(np.float32)), torch.from_numpy(y_val)).item())
        scheduler.step(val_loss)
        row = {"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss,
               "learning_rate": float(optimizer.param_groups[0]["lr"])}
        history.append(row)
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        snapshot = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(),
                    "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                    "history": history, "best_loss": best_loss, "best_epoch": best_epoch,
                    "best_state": best_state, "stale": stale, "next_epoch": epoch + 1,
                    "config_sha256": sha256(cp), "dataset_ready_sha256": sha256(data / "DATASET_READY.json")}
        tmp = state_path.with_suffix(".pt.tmp"); torch.save(snapshot, tmp); os.replace(tmp, state_path)
        history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(f"epoch={epoch + 1} train_loss={train_loss:.6f} val_loss={val_loss:.6f} best_epoch={best_epoch}", flush=True)
        if stale >= patience:
            stop_reason = "validation_loss_patience_8"
            break
    if best_state is None:
        raise RuntimeError("no best CNN state is available")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        val_logits = model(torch.from_numpy(val_x)).cpu().numpy().astype(np.float64)
    threshold = choose_f1_threshold(y_val.astype(int), val_logits)
    val_metrics = metrics_at_threshold(y_val.astype(int), val_logits, float(threshold["threshold"]))
    far_thresholds = {str(far): {"threshold": operating_threshold(y_val.astype(int), val_logits, far),
                                 "metrics": metrics_at_threshold(y_val.astype(int), val_logits,
                                     operating_threshold(y_val.astype(int), val_logits, far))}
                      for far in (0.01, 0.05)}
    checkpoint_path = out / "cnn_v3_a0b_rawcache_best.pt"
    temp = checkpoint_path.with_suffix(".pt.tmp")
    torch.save({"state_dict": best_state, "architecture": {"channels": [16, 32, 64, 64],
                "kernel_size": 3, "pool": [2, 2], "head": "global_average", "dropout": 0.3},
                "parameter_count": 60641, "seed": 42, "best_epoch": best_epoch}, temp)
    os.replace(temp, checkpoint_path)
    freeze = {"status": "FROZEN_BEFORE_TEST", "model": "single depth_4 CNN", "checkpoint": checkpoint_path.name,
              "checkpoint_sha256": sha256(checkpoint_path), "dataset_ready_sha256": sha256(data / "DATASET_READY.json"),
              "train_manifest_sha256": sha256(data / "manifests/train.csv"), "val_manifest_sha256": sha256(data / "manifests/val.csv"),
              "normalization_sha256": sha256(data / "normalization.json"), "config_sha256": sha256(cp),
              "architecture": {"channels": [16, 32, 64, 64], "kernel_size": 3, "pool": [2, 2], "head": "global_average", "dropout": 0.3},
              "parameter_count": 60641, "seed": 42, "best_epoch": best_epoch,
              "best_validation_loss": best_loss, "stop_reason": stop_reason,
              "validation_threshold": threshold, "validation_metrics": val_metrics,
              "validation_far_operating_points": far_thresholds,
              "score_space": "raw_logit", "validation_size": len(y_val),
              "test_disclosure": "V1 benchmark reused; not an independent unseen test",
              "training_seconds": time.perf_counter() - started, "torch_version": torch.__version__,
              "threads": torch.get_num_threads(), "raw_train_cache_sha256": sha256(raw_cache_path),
              "raw_train_cache_report_sha256": sha256(raw_cache_path.parent / "raw_train_cache_report.json"),
              "training_input_source": "full-clip mel power rebuilt from all raw training WAVs; bitwise equal to frozen V2 cache"}
    (out / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "FROZEN_BEFORE_TEST", "checkpoint": str(checkpoint_path),
                      "best_epoch": best_epoch, "best_validation_loss": best_loss,
                      "threshold": threshold, "parameters": 60641}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    train(parser.parse_args().config)


if __name__ == "__main__":
    main()
