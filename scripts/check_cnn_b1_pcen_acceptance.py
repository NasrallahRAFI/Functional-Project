"""Check the registered PCEN gain limit on every stored eligible train window."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dronedet.config import load_config, params_hash
from dronedet.features import load_cache_index, load_power_cache, centre_window
from dronedet.normalization import pcen_gain_check


def main() -> int:
    root = Path.cwd()
    config = load_config(root / "config.yaml")
    if config.mode != "global_stats":
        raise RuntimeError("PCEN acceptance must begin from the frozen project config")
    cpu_count = os.cpu_count()
    if cpu_count != 4:
        raise RuntimeError(f"registered run requires the recorded four logical CPUs; found {cpu_count}")
    torch.set_num_threads(cpu_count)
    paths = {"root": root, "spec": root / "data/processed/spectrograms",
             "splits": root / "data/processed/splits"}
    manifest_path = paths["splits"] / "train.csv"
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    rows = pd.read_csv(manifest_path, keep_default_na=False)
    index = load_cache_index(paths, "train")
    power = load_power_cache(paths, "train", mmap_mode="r")
    if not np.array_equal(rows.clip_id.to_numpy(), index.clip_id.to_numpy()):
        raise RuntimeError("train manifest and cache index differ")
    if "stats_eligible" not in index:
        raise RuntimeError("train cache index is missing stored stats_eligible")
    eligible = index[index.stats_eligible.astype(bool)]
    if len(eligible) != 7519:
        raise RuntimeError(f"expected 7,519 stored eligible train clips, got {len(eligible)}")
    windows = [centre_window(power[:, int(row.start_frame):int(row.start_frame)+int(row.n_frames)], config)
               for row in eligible.itertuples(index=False)]
    result = pcen_gain_check(windows, config)
    result.update({"status": "pass" if result.get("status") == "pass" else "fail",
                   "eligible_train_windows": len(windows),
                   "excluded_train_clips": len(index) - len(windows),
                   "eligibility_source": "stored index_train.csv stats_eligible",
                   "manifest_sha256": manifest_sha,
                   "frontend_params_hash": params_hash(config),
                   "acceptance_rule": "max mean absolute PCEN change at -12/+12 dB <= 0.2 baseline std",
                   "device": "cpu", "cpu_threads": cpu_count})
    output = root / "results/tables/cnn_b1_pcen_acceptance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
