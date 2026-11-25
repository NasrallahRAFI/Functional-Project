"""Measure legacy and fixed-ring allocation cost at the configured stream cadence."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from demo.run_demo import Ring  # noqa: E402
from dronedet.config import load_config  # noqa: E402


class LegacyRing:
    """Recreate the previous concatenate-and-trim ring implementation."""
    def __init__(self, size: int):
        self.size = size
        self.buffer = np.empty(0, dtype=np.float32)

    @property
    def data(self) -> np.ndarray:
        return self.buffer.copy()

    def push(self, values: np.ndarray) -> None:
        self.buffer = np.concatenate((self.buffer, values))[-self.size:]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(kind: str, seconds: int, seed: int) -> dict:
    config = load_config(ROOT / "config.yaml")
    ring = Ring(config.win_samples) if kind == "fixed" else LegacyRing(config.win_samples)
    sample_count = config.sr * seconds
    rng = np.random.default_rng(seed)
    audio = rng.standard_normal(sample_count).astype(np.float32)
    # Match the live builder's hop cadence and snapshot each decision window.
    cadence_hops = max(1, int(round(config.decision_interval_s * config.sr / config.hop)))
    cadence = cadence_hops * config.hop
    process = psutil.Process()
    base_rss = process.memory_info().rss
    started = time.perf_counter()
    snapshots = []
    windows = 0
    for position in range(0, sample_count, config.hop):
        piece = audio[position:position + config.hop]
        if kind == "fixed":
            ready = ring.push(piece)
        else:
            ring.push(piece)
            ready = ring.buffer.size >= config.win_samples
        if ready and position + len(piece) >= config.win_samples:
            if (position + len(piece) - config.win_samples) % cadence < config.hop:
                snap = ring.data
                windows += 1
                if windows == 1 or windows == int(round(seconds / config.decision_interval_s)):
                    snapshots.append(hashlib.sha256(snap.tobytes()).hexdigest())
    elapsed = time.perf_counter() - started
    del audio
    return {"implementation": kind, "sample_rate": config.sr, "window_samples": config.win_samples,
            "hop_samples": config.hop, "decision_interval_seconds": config.decision_interval_s,
            "duration_seconds": seconds, "windows_emitted": windows, "runtime_seconds": elapsed,
            "base_rss_bytes": base_rss, "end_rss_bytes": process.memory_info().rss,
            "rss_observation_note": "process RSS sampled at start/end; throughput and repeated allocations measured directly",
            "snapshot_sha256": snapshots}


def main() -> int:
    seconds = 60
    seed = 20261125
    legacy = measure("legacy", seconds, seed)
    fixed = measure("fixed", seconds, seed)
    report = {"schema_version": 1, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "platform": sys.platform, "python": sys.version,
              "benchmark_source_sha256": sha256(Path(__file__).resolve()),
              "ring_source_sha256": sha256(ROOT / "demo" / "run_demo.py"),
              "status": "PASS" if legacy["windows_emitted"] == fixed["windows_emitted"]
              and legacy["snapshot_sha256"] == fixed["snapshot_sha256"] else "FAIL",
              "results": {"legacy": legacy, "fixed": fixed},
              "runtime_speedup_fixed_over_legacy": legacy["runtime_seconds"] / fixed["runtime_seconds"],
              "interpretation": "Synthetic one-minute 16 kHz stream, hop/window/cadence taken from project config. Snapshot hashes verify the emitted windows sampled at the first and last cadence. Timing is host-specific; RSS here includes the preallocated synthetic input and is not a high-water peak measurement."}
    out = ROOT / "data" / "interim" / "frontend_correction_v1_ring_benchmark.json"
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
