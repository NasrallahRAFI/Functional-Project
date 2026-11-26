"""Raw-waveform CNN V3 dataset with locked split access and matched V2 crops."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..features import fit_length, load_wave, mel_power, transform
from ..splits import guard_test_access
from .augment import ChannelAugment, stable_rng


def source_seed(seed: int, epoch: int, clip_id: str, draw_id: int = 0) -> int:
    payload = f"{seed}\0{epoch}\0{clip_id}\0{draw_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def frame_aligned_crop(audio: np.ndarray, n_frames: int, hop: int, rng: np.random.Generator, *, training: bool) -> tuple[np.ndarray, int]:
    """Select a valid hop-aligned V2-equivalent window from a fixed-length clip."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1 or n_frames <= 0 or hop <= 0 or x.size < n_frames:
        raise ValueError("audio must contain a valid mono crop and frame geometry")
    available_samples = x.size - n_frames
    available_frames = available_samples // hop
    start_frame = int(rng.integers(0, available_frames + 1)) if training else available_frames // 2
    start = start_frame * hop
    return x[start : start + n_frames].copy(), start_frame


class RawWaveformDataset(Dataset):
    """Load only train/validation rows; test access is rejected at construction."""

    def __init__(
        self,
        *,
        split: str,
        raw_root: str | Path,
        manifest: pd.DataFrame,
        config,
        mean: np.ndarray,
        std: np.ndarray,
        seed: int = 42,
        augment: ChannelAugment | None = None,
        channel_enabled: bool = False,
        v2_gain_db: tuple[float, float] | None = (-12.0, 12.0),
        epoch: int = 0,
    ) -> None:
        guard_test_access(split, allow_test=False)
        if split not in {"train", "val"}:
            raise ValueError("V3 raw-waveform dataset supports train and validation only")
        if "clip_id" not in manifest or "label" not in manifest:
            raise ValueError("manifest must contain clip_id and label")
        if "split" in manifest and not (manifest["split"].astype(str) == split).all():
            raise ValueError("manifest contains rows outside the requested split")
        if split == "val" and channel_enabled:
            raise ValueError("validation is a fixed evaluation set; use its separate suite builder")
        self.split = split
        self.training = split == "train"
        self.raw_root = Path(raw_root).resolve()
        self.audio_root = self.raw_root / "Binary_Drone_Audio"
        self.rows = manifest.reset_index(drop=True).copy()
        self.config = config
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.shape != (config.n_mels,) or self.std.shape != self.mean.shape or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("normalization statistics do not match the frontend")
        self.seed = int(seed)
        self.augment = augment
        self.channel_enabled = bool(channel_enabled)
        self.v2_gain_db = v2_gain_db
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def __getitem__(self, item: int | tuple[int, int]):
        if isinstance(item, tuple):
            i, draw_id = int(item[0]), int(item[1])
        else:
            i, draw_id = int(item), 0
        if not 0 <= i < len(self.rows) or draw_id < 0:
            raise IndexError("dataset index or draw id is outside its valid range")
        row = self.rows.iloc[i]
        clip_id = str(row.clip_id)
        audio = load_wave(self.audio_root / Path(*clip_id.split("/")), self.config)
        audio = fit_length(audio, self.config)
        rng = stable_rng(self.seed, self.epoch, clip_id, draw_id)
        augmentation_record = {"branch": "clean", "clip_id": clip_id, "epoch": self.epoch, "draw_id": draw_id}
        if self.training and self.channel_enabled:
            if self.augment is None:
                raise RuntimeError("channel augmentation enabled without a ChannelAugment")
            audio, augmentation_record = self.augment(audio, rng, label=int(row.label))
            augmentation_record.update({"clip_id": clip_id, "epoch": self.epoch, "draw_id": draw_id})
        elif self.training and self.v2_gain_db is not None:
            gain_db = float(rng.uniform(*self.v2_gain_db))
            audio = np.asarray(audio * (10.0 ** (gain_db / 20.0)), dtype=np.float32)
            augmentation_record.update({"branch": "v2_gain", "gain_db": gain_db})
        # V2 sampled a whole-clip mel frame offset. Selecting an aligned raw
        # crop before mel extraction is the registered V3 input path.
        window, start_frame = frame_aligned_crop(
            audio, self.config.win_samples, self.config.hop, rng, training=self.training
        )
        power = mel_power(window, self.config)
        model_input = transform(power, self.config, {"mean": self.mean, "std": self.std})
        if model_input.shape != (1, self.config.n_mels, self.config.n_win) or not np.isfinite(model_input).all():
            raise RuntimeError("raw-waveform frontend emitted an invalid model tensor")
        return {
            "x": torch.from_numpy(np.ascontiguousarray(model_input)),
            "y": torch.tensor(float(row.label), dtype=torch.float32),
            "clip_id": clip_id,
            "group": str(row.get("group", "")),
            "start_frame": start_frame,
            "augmentation": augmentation_record,
        }


def cache_tensor_for_offset(power: np.ndarray, start_frame: int, config, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Reference cached-power tensor at an explicit frame offset for A0b parity."""
    p = np.asarray(power, dtype=np.float32)
    start_frame = int(start_frame)
    if p.ndim != 2 or p.shape[0] != config.n_mels or start_frame < 0 or start_frame + config.n_win > p.shape[1]:
        raise ValueError("cached mel power does not contain the requested window")
    if config.mode != "global_stats":
        raise ValueError("A0 raw/cache parity currently requires V2 global_stats mode")
    m = np.asarray(mean, dtype=np.float32); s = np.asarray(std, dtype=np.float32)
    if m.shape != (config.n_mels,) or s.shape != m.shape or np.any(s <= 0):
        raise ValueError("invalid normalization vectors")
    window = p[:, start_frame : start_frame + config.n_win]
    db = 10.0 * np.log10(np.maximum(window, config.amin))
    return ((db - m[:, None]) / s[:, None])[None, ...].astype(np.float32)
