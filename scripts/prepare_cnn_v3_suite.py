"""Freeze deterministic validation channel views without reading test audio."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from dronedet.config import config_from_mapping
from dronedet.features import fit_length, load_wave
from dronedet.cnn_v3.augment import AugmentConfig, ChannelAugment, stable_rng
from dronedet.splits import guard_test_access


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build(config_path: Path) -> None:
    config_path = config_path.resolve()
    project = config_path.parent.parent
    doc = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fe = config_from_mapping(doc["frontend"])
    raw_value = __import__("os").environ.get("DRONE_AUDIO_DATASET_ROOT", doc["raw_root"])
    raw_root = Path(raw_value).expanduser()
    if not raw_root.is_absolute():
        raw_root = project / raw_root
    raw_root = raw_root.resolve()
    output = project / doc["output_root"] / "validation_suite"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen validation suite: {output}")
    val = pd.read_csv(project / "results/cnn_v2/data/manifests/val.csv", keep_default_na=False)
    if "test" in set(val.split.astype(str)):
        raise RuntimeError("validation manifest contains test rows")
    guard_test_access("val", allow_test=False)
    severity = doc["evaluation"]
    specs = {
        "mild": {"gain_db": (-12.0, -3.0), "noise_snr_db": tuple(severity["mild"]["noise_snr_db"]),
                 "rt60_s": tuple(severity["mild"]["rt60_s"]), "hp_hz": tuple(severity["mild"]["hp_hz"]),
                 "lp_hz": tuple(severity["mild"]["lp_hz"]), "p_eq": 0.25, "p_soft_clip": 0.1},
        "moderate": {"gain_db": (-20.0, 0.0), "noise_snr_db": tuple(severity["moderate"]["noise_snr_db"]),
                     "rt60_s": tuple(severity["moderate"]["rt60_s"]), "hp_hz": tuple(severity["moderate"]["hp_hz"]),
                     "lp_hz": tuple(severity["moderate"]["lp_hz"]), "p_eq": 0.5, "p_soft_clip": 0.2},
        "severe": {"gain_db": (-30.0, -6.0), "noise_snr_db": tuple(severity["severe"]["noise_snr_db"]),
                   "rt60_s": tuple(severity["severe"]["rt60_s"]), "hp_hz": tuple(severity["severe"]["hp_hz"]),
                   "lp_hz": tuple(severity["severe"]["lp_hz"]), "p_eq": 0.75, "p_soft_clip": 0.3},
    }
    output.mkdir(parents=True)
    rows = []
    for sev, spec in specs.items():
        out_dir = output / sev
        out_dir.mkdir()
        aug = ChannelAugment(config=AugmentConfig(
            sample_rate=fe.sr, p_clean_recipe=0.0, p_band_limit=1.0,
            p_eq=spec["p_eq"], p_soft_clip=spec["p_soft_clip"],
            gain_db=spec["gain_db"], p_reverb=1.0,
            rt60_s=spec["rt60_s"], drr_db=(-5.0, 10.0),
            p_noise=1.0, noise_snr_db=spec["noise_snr_db"],
            floor_dbfs=(-75.0, -60.0), minimum_positive_snr_db=0.0,
        ))
        for row in val.to_dict("records"):
            clip_id = str(row["clip_id"])
            audio = fit_length(load_wave(raw_root / "Binary_Drone_Audio" / Path(*clip_id.split("/")), fe), fe)
            rng = stable_rng(20261126, 0, f"{sev}/{clip_id}")
            rendered, params = aug(audio, rng, label=int(row["label"]))
            target = out_dir / (hashlib.sha256(clip_id.encode()).hexdigest()[:20] + ".wav")
            # PCM16 is the frozen evaluation format; it approximates microphone capture
            # and makes the actual evaluated bytes content-addressable.
            sf.write(target, rendered, fe.sr, subtype="PCM_16")
            rows.append({"severity": sev, "clip_id": clip_id, "group": str(row["group"]),
                         "label": int(row["label"]), "source_split": "val",
                         "source_pcm_sha1": str(row["pcm_sha1"]), "parameters": params,
                         "path": target.relative_to(output).as_posix(), "sha256": sha256(target),
                         "size": target.stat().st_size})
    manifest = {"status": "FROZEN", "schema_version": 1, "source": "CNN V2 validation manifest",
                "seed_root": 20261126, "sample_rate": fe.sr,
                "weights": {"clean": 0.5, "mild": 1/6, "moderate": 1/6, "severe": 1/6},
                "validation_manifest_sha256": sha256(project / "results/cnn_v2/data/manifests/val.csv"),
                "config_sha256": sha256(config_path), "examples": rows,
                "count_per_condition": {s: sum(r["severity"] == s for r in rows) for s in specs}}
    (output / "SUITE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "SUITE_MANIFEST.sha256").write_text(sha256(output / "SUITE_MANIFEST.json") + "  SUITE_MANIFEST.json\n", encoding="ascii")
    print(json.dumps({"status": "FROZEN", "suite": str(output), "counts": manifest["count_per_condition"],
                      "manifest_sha256": sha256(output / "SUITE_MANIFEST.json")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/cnn_v3.yaml"))
    args = parser.parse_args()
    build(args.config)


if __name__ == "__main__":
    main()
