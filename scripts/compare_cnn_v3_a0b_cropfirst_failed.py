"""Compare raw A0b and cached A0a at their shared frozen V2 threshold."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
import yaml

from dronedet.config import config_from_mapping
from dronedet.features import to_db
from dronedet.metrics import metrics_at_threshold
from dronedet.models.cnn import build_cnn


def main():
    root = Path(__file__).resolve().parents[1]
    doc = yaml.safe_load((root / "config/cnn_v3.yaml").read_text(encoding="utf-8"))
    fe = config_from_mapping(doc["frontend"])
    data = root / "results/cnn_v2/data"
    ix = pd.read_csv(data / "cache/index_val.csv")
    y = pd.read_csv(data / "manifests/val.csv", keep_default_na=False).label.to_numpy(dtype=int)
    power = np.load(data / "cache/mel_power_val.npy", mmap_mode="r", allow_pickle=False)
    stats = np.load(data / "normalization_stats.npz", allow_pickle=False)
    starts = ix.start_frame.to_numpy(dtype=int) + ix.center_start_frame.to_numpy(dtype=int)
    batch = np.stack([power[:, s:s+fe.n_win] for s in starts]).astype(np.float32)
    x = (to_db(batch, fe) - stats["mean"][None, :, None]) / stats["std"][None, :, None]
    x = torch.from_numpy(np.ascontiguousarray(x[:, None, ...], dtype=np.float32))
    model = build_cnn((1,64,28), channels=(16,32,64,64), kernel_size=3, pool=(2,2), head="global_average", dropout=.3)
    payload = torch.load(root / "results/cnn_v3/experiments/A0b_raw_seed42/cnn_v3_a0b_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"]); model.eval()
    with torch.inference_mode(): logits = model(x).numpy().astype(float)
    report = metrics_at_threshold(y, logits, 2.823960542678833)
    report.update({"run_id":"A0b_raw_seed42", "model_sha256":__import__("hashlib").sha256((root / "results/cnn_v3/experiments/A0b_raw_seed42/cnn_v3_a0b_best.pt").read_bytes()).hexdigest(),"test_access":False})
    out = root / "results/cnn_v3/experiments/A0b_raw_seed42/A0b_frozen_v2_threshold_comparison.json"
    out.write_text(json.dumps(report, indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
