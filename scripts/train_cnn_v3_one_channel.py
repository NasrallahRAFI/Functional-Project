"""Run the single registered CNN V3 combined-channel experiment."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from train_cnn_v3_rung import train


if __name__ == "__main__":
    train("ONE_CHANNEL", 42, ROOT / "config/cnn_v3_one_channel.yaml")
