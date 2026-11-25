"""Measure streaming and concatenate-based masked-statistics memory on train cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dronedet.config import load_project_config  # noqa: E402
from dronedet.features import load_cache_index, load_power_cache, to_db  # noqa: E402
from dronedet.normalization import GLOBAL_STATS_POLICY, Normalizer  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare():
    project = load_project_config(ROOT / "config.yaml")
    paths, config = project["P"], project["fe"]
    manifest = pd.read_csv(paths["splits"] / "train.csv", keep_default_na=False)
    index = load_cache_index(paths, "train")
    if len(manifest) != len(index):
        raise ValueError("train manifest and cache index row counts differ")
    cache = load_power_cache(paths, "train", mmap_mode="r")
    rows = manifest.to_dict("records")
    power_by_file = {}
    for position, row in enumerate(rows):
        stored = index.iloc[position]
        clip = str(row.get("clip_id", row.get("file", "")))
        if not clip or clip != str(stored["clip_id"]):
            raise ValueError("train manifest and cache index order differs")
        row["split"] = "train"
        row["stats_eligible"] = bool(stored["stats_eligible"])
        start, count = int(stored["start_frame"]), int(stored["n_frames"])
        power_by_file[clip] = cache[:, start:start + count]
    return paths, config, rows, power_by_file, cache


def _legacy_concat_fit(rows, power_by_file, config):
    """Recreate the pre-W11 full float64 dB-list plus concatenation algorithm."""
    values = []
    excluded = 0
    for row in rows:
        if not row["stats_eligible"]:
            excluded += 1
            continue
        values.append(to_db(power_by_file[row["clip_id"]], config).astype(np.float64))
    joined = np.concatenate(values, axis=1)
    floor_threshold = 10.0 * np.log10(config.amin) + 1e-6
    mean = np.empty(config.n_mels, dtype=np.float64)
    std = np.empty(config.n_mels, dtype=np.float64)
    counts = np.empty(config.n_mels, dtype=np.int64)
    for band in range(config.n_mels):
        cells = joined[band, joined[band] > floor_threshold]
        if cells.size == 0:
            raise ValueError(f"mel band {band} has no non-floor cells")
        mean[band] = cells.mean(dtype=np.float64)
        std[band] = cells.std(dtype=np.float64)
        counts[band] = cells.size
    return mean.astype(np.float32), std.astype(np.float32), excluded, int(joined.shape[1]), counts.tolist()


def run_variant(mode: str) -> dict:
    if mode not in {"legacy_concat", "streaming"}:
        raise ValueError(f"unsupported mode: {mode}")
    paths, config, rows, power_by_file, cache = _prepare()
    process = psutil.Process(os.getpid())
    base_rss = process.memory_info().rss
    peak_rss = [base_rss]
    stop = threading.Event()

    def sample_memory():
        while not stop.wait(0.005):
            try:
                peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
            except psutil.Error:
                return

    sampler = threading.Thread(target=sample_memory, name="rss-sampler", daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        if mode == "legacy_concat":
            mean, std, excluded, frame_count, counts = _legacy_concat_fit(rows, power_by_file, config)
        else:
            fitted = Normalizer.fit(rows, config, power_by_file)
            mean, std = fitted.mean, fitted.std
            excluded = fitted.excluded_count
            frame_count = int(fitted.fit_record["frame_count"])
            counts = fitted.fit_record["retained_cells_per_band"]
    finally:
        elapsed = time.perf_counter() - started
        peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
        stop.set()
        sampler.join(timeout=1.0)
    original_path = paths["spec"] / "norm_stats_global.npz"
    with np.load(original_path, allow_pickle=False) as original:
        mean_error = float(np.max(np.abs(mean - original["mean"]), initial=0.0))
        std_error = float(np.max(np.abs(std - original["std"]), initial=0.0))
        exact = bool(np.array_equal(mean, original["mean"]) and np.array_equal(std, original["std"]))
    if getattr(cache, "_mmap", None) is not None:
        cache._mmap.close()
    return {"mode": mode, "status": "PASS" if exact and excluded == 88 else "FAIL",
            "eligible_clips": len(rows) - excluded, "excluded_clips": excluded,
            "frame_count": frame_count, "seconds": elapsed,
            "base_rss_bytes": base_rss, "peak_rss_bytes": peak_rss[0],
            "incremental_peak_rss_bytes": max(0, peak_rss[0] - base_rss),
            "mean_max_abs_error": mean_error, "std_max_abs_error": std_error,
            "statistics_exact": exact, "retained_cells_per_band": counts,
            "fit_policy": GLOBAL_STATS_POLICY, "train_manifest_sha256": _sha256(paths["splits"] / "train.csv"),
            "statistics_sha256": _sha256(original_path),
            "test_audio_or_array_payloads_opened": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("legacy_concat", "streaming", "compare"), default="compare")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "compare":
        results = {}
        for mode in ("legacy_concat", "streaming"):
            run = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--mode", mode],
                                 cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
            if run.returncode:
                print(run.stdout)
                print(run.stderr, file=sys.stderr)
                return run.returncode
            results[mode] = json.loads(run.stdout)
        a, b = results["legacy_concat"], results["streaming"]
        report = {"schema_version": 1, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "status": "PASS" if a["statistics_exact"] and b["statistics_exact"]
                  and np.array_equal(a["retained_cells_per_band"], b["retained_cells_per_band"]) else "FAIL",
                  "platform": sys.platform, "python": sys.version,
                  "benchmark_source_sha256": _sha256(Path(__file__).resolve()),
                  "normalization_source_sha256": _sha256(ROOT / "src" / "dronedet" / "normalization.py"),
                  "results": results,
                  "interpretation": "Separate subprocess RSS peaks compare the previous concatenate allocation pattern with the streaming implementation; values are valid only for this host and cache. Both variants must reproduce installed statistics exactly."}
    else:
        report = run_variant(args.mode)
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
