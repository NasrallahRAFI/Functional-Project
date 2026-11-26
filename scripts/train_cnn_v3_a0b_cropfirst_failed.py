"""Train preregistered A0b: V2 recipe with raw waveform frontend and clean stopping."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

from dronedet.config import config_from_mapping
from dronedet.features import fit_length, load_wave, mel_power, to_db, transform
from dronedet.metrics import choose_f1_threshold, metrics_at_threshold
from dronedet.models.cnn import build_cnn


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def cached_val(data: Path, fe, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    ix = pd.read_csv(data / "cache/index_val.csv")
    power = np.load(data / "cache/mel_power_val.npy", mmap_mode="r", allow_pickle=False)
    out = []
    for row in ix.itertuples(index=False):
        start = int(row.start_frame) + int(row.center_start_frame)
        p = power[:, start : start + fe.n_win]
        db = to_db(p, fe)
        out.append(((db - mean[:, None]) / std[:, None])[None, ...])
    return np.ascontiguousarray(np.stack(out), dtype=np.float32)


def train(config_path: Path) -> None:
    cp = config_path.resolve()
    doc = yaml.safe_load(cp.read_text(encoding="utf-8"))
    root = cp.parent.parent
    fe = config_from_mapping(doc["frontend"])
    data = root / "results/cnn_v2/data"
    out = root / "results/cnn_v3/experiments/A0b_raw_seed42"
    out.mkdir(parents=True, exist_ok=False)
    raw_root = Path(__import__("os").environ.get("DRONE_AUDIO_DATASET_ROOT", root / doc["raw_root"])).resolve()
    train_manifest = pd.read_csv(data / "manifests/train.csv", keep_default_na=False)
    val_manifest = pd.read_csv(data / "manifests/val.csv", keep_default_na=False)
    stats = np.load(data / "normalization_stats.npz", allow_pickle=False)
    mean, std = stats["mean"].astype(np.float32), stats["std"].astype(np.float32)
    train_index = pd.read_csv(data / "cache/index_train.csv")
    val_x = torch.from_numpy(cached_val(data, fe, mean, std))
    y_train = train_manifest.label.to_numpy(dtype=np.float32)
    y_val = val_manifest.label.to_numpy(dtype=np.float32)
    torch.set_num_threads(int(doc["threads"]))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    model = build_cnn((1, 64, 28), channels=(16, 32, 64, 64), kernel_size=3,
                      pool=(2, 2), head="global_average", dropout=0.3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(6640 / 967, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    history = []
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    started = time.perf_counter()
    for epoch in range(100):
        model.train()
        epoch_seed = 42 * 1_000_003 + epoch
        rng = np.random.default_rng(epoch_seed)
        order_rng = np.random.default_rng(epoch_seed ^ 0x5EED)
        order = order_rng.permutation(len(y_train))
        loss_sum = 0.0
        for start in range(0, len(order), 64):
            ids = order[start : start + 64]
            batch_rows = train_index.iloc[ids].to_dict("records")
            offsets = np.array([int(rng.integers(0, int(r["n_frames"]) - fe.n_win + 1))
                                for r in batch_rows], dtype=np.int64)
            gains = rng.uniform(-12.0, 12.0, len(ids))
            views = []
            for row_i, offset, gain in zip(ids, offsets, gains):
                clip_id = str(train_manifest.iloc[int(row_i)].clip_id)
                audio = fit_length(load_wave(raw_root / "Binary_Drone_Audio" / Path(*clip_id.split("/")), fe), fe)
                audio = audio * np.float32(10.0 ** (float(gain) / 20.0))
                start_sample = int(offset) * fe.hop
                window = audio[start_sample : start_sample + fe.win_samples]
                p = mel_power(window, fe)
                views.append(transform(p, fe, {"mean": mean, "std": std}))
            xb = torch.from_numpy(np.ascontiguousarray(np.stack(views), dtype=np.float32))
            yb = torch.from_numpy(y_train[ids])
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
        train_loss = loss_sum / len(y_train)
        model.eval()
        with torch.inference_mode():
            val_logits = model(val_x).cpu().numpy().astype(np.float64)
            val_loss = float(criterion(torch.from_numpy(val_logits.astype(np.float32)), torch.from_numpy(y_val)).item())
        scheduler.step(val_loss)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss,
                        "learning_rate": float(optimizer.param_groups[0]["lr"])})
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1}, out / "checkpoint_epoch.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(f"epoch={epoch+1} train_loss={train_loss:.6f} val_loss={val_loss:.6f} best_epoch={best_epoch}", flush=True)
        if stale >= 8:
            break
    if best_state is None:
        raise RuntimeError("no eligible checkpoint was selected")
    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        val_logits = model(val_x).cpu().numpy().astype(np.float64)
    threshold = choose_f1_threshold(y_val.astype(int), val_logits)
    metrics = metrics_at_threshold(y_val.astype(int), val_logits, float(threshold["threshold"]))
    checkpoint = out / "cnn_v3_a0b_best.pt"
    torch.save({"state_dict": best_state, "architecture": {"channels": [16,32,64,64],
                "kernel_size": 3, "pool": [2,2], "head": "global_average", "dropout": 0.3},
                "parameter_count": 60641, "seed": 42, "best_epoch": best_epoch}, checkpoint)
    record = {"status": "FROZEN_BEFORE_TEST", "run_id": "A0b_raw_seed42", "parent": "A0a_cached_seed42",
              "change": "raw waveform loader at matched frontend; no channel augmentation",
              "seed": 42, "best_epoch": best_epoch, "best_validation_loss": best_loss,
              "stop_reason": "validation_loss_patience_8", "validation_threshold": threshold,
              "validation_metrics": metrics, "checkpoint_sha256": sha(checkpoint),
              "config_sha256": sha(cp), "dataset_ready_sha256": sha(data / "DATASET_READY.json"),
              "train_manifest_sha256": sha(data / "manifests/train.csv"),
              "val_manifest_sha256": sha(data / "manifests/val.csv"),
              "training_seconds": time.perf_counter() - started, "test_access": False}
    (out / "A0b_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    train(Path(__file__).resolve().parents[1] / "config/cnn_v3.yaml")
