"""Build the registered B2 front-end variants and fit them on local CPU."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dronedet.config import cache_content_hash, config_hash, load_config, params_hash
from dronedet.features import (centre_window, fix_length, get_window, load_audio,
                               mel_power, standardize, to_db)
from dronedet.ledger import append_run, sha256_file, write_run_report
from dronedet.metrics import select_variant
from dronedet.models.cnn import build_cnn, parameter_count
from dronedet.normalization import stress_filter
from dronedet.training import train_torch_model
from scripts.run_cnn_b1_local import _batch_transform, _evaluate, CONDITIONS, SEEDS, FILTERS


ROOT = Path.cwd().resolve()
VARIANTS = [
    {"name": "n_mels_40", "overrides": {"n_mels": 40}},
    {"name": "n_mels_96", "overrides": {"n_mels": 96}},
    {"name": "nfft512_hop128_mels40", "overrides": {"n_fft": 512, "win_length": 512, "hop": 128, "n_mels": 40}},
    {"name": "nfft2048_hop512", "overrides": {"n_fft": 2048, "win_length": 2048, "hop": 512}},
    {"name": "mel_scale_slaney", "overrides": {"mel_scale": "slaney"}},
]


def _manifest_rows(split: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/processed/splits" / f"{split}.csv", keep_default_na=False)


def _raw_binary_root(raw_root: Path) -> Path:
    candidates = (raw_root / "Binary_Drone_Audio",
                  raw_root / "DroneAudioDataset-master" / "Binary_Drone_Audio")
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Binary_Drone_Audio missing under {raw_root}")


def _variant_dir(config) -> Path:
    path = ROOT / "data/processed/spectrograms" / cache_content_hash(config)[:16]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_split(split: str, manifest: pd.DataFrame, config, raw_binary: Path,
                 folder: Path) -> tuple[np.ndarray, pd.DataFrame]:
    cache_path = folder / f"mel_power_{split}.npy"
    index_path = folder / f"index_{split}.csv"
    record_path = folder / "variant_cache.json"
    cache_hash = cache_content_hash(config)
    manifest_sha = hashlib.sha256((ROOT / "data/processed/splits" / f"{split}.csv").read_bytes()).hexdigest()
    if cache_path.is_file() and index_path.is_file() and record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("cache_content_hash") != cache_hash or record.get(f"{split}_manifest_sha256") != manifest_sha:
            raise RuntimeError(f"variant cache provenance mismatch for {split}: {cache_hash}")
        expected_sha = record.get(f"{split}_array_sha256")
        if not expected_sha or sha256_file(cache_path) != expected_sha:
            raise RuntimeError(f"variant cache file hash mismatch for {split}: {cache_path}")
        index = pd.read_csv(index_path, keep_default_na=False)
        cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        if list(cache.shape) != record.get(f"{split}_array_shape") or str(cache.dtype) != record.get(f"{split}_array_dtype"):
            raise RuntimeError(f"variant cache header mismatch for {split}")
        if not np.array_equal(manifest.clip_id.to_numpy(), index.clip_id.to_numpy()):
            raise RuntimeError(f"variant {split} cache order mismatch")
        return cache, index
    if cache_path.exists() or index_path.exists():
        raise RuntimeError(f"partial variant cache exists for {split}; refusing to overwrite: {folder}")
    rows, powers, cursor = [], [], 0
    for source in manifest.itertuples(index=False):
        audio, sr = load_audio(raw_binary / str(source.clip_id), config)
        audio = standardize(audio, sr, config)
        fixed, length_status = fix_length(audio, config)
        if not fixed.size:
            raise RuntimeError(f"{source.clip_id}: invalid length {length_status}")
        power = mel_power(fixed, config)
        centre = centre_window(power, config)
        all_zero = bool(getattr(source, "all_zero", getattr(source, "silent", False)))
        near = bool(getattr(source, "near_silent", False))
        crop_zero = bool(np.all(centre == 0.0))
        eligible = not (all_zero or near or crop_zero)
        rows.append({"clip_id": str(source.clip_id), "start_frame": cursor,
                     "n_frames": int(power.shape[1]),
                     "win_start_frame": int((power.shape[1] - config.n_win) // 2),
                     "silent": all_zero, "near_silent": near,
                     "crop_all_zero": crop_zero, "stats_eligible": eligible,
                     "length_status": length_status})
        powers.append(power)
        cursor += power.shape[1]
    index = pd.DataFrame(rows)
    cache = np.concatenate(powers, axis=1).astype(np.float32, copy=False)
    temp_cache = cache_path.with_suffix(".npy.tmp")
    with temp_cache.open("wb") as stream:
        np.save(stream, cache, allow_pickle=False)
    os.replace(temp_cache, cache_path)
    index.to_csv(index_path, index=False)
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
    else:
        record = {"cache_content_hash": cache_hash,
                  "variant_params_hash": params_hash(config),
                  "train_manifest_sha256": hashlib.sha256((ROOT / "data/processed/splits/train.csv").read_bytes()).hexdigest(),
                  "libraries": {"numpy": np.__version__, "librosa": version("librosa"),
                                "soxr": version("soxr"), "soundfile": version("soundfile")}}
    record[f"{split}_manifest_sha256"] = manifest_sha
    record[f"{split}_array_shape"] = list(cache.shape)
    record[f"{split}_array_dtype"] = str(cache.dtype)
    record[f"{split}_array_sha256"] = sha256_file(cache_path)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return cache, index


def _fit_stats(cache: np.ndarray, index: pd.DataFrame, config) -> tuple[dict, int]:
    eligible = index[index.stats_eligible.astype(bool)]
    if len(eligible) == 0:
        raise RuntimeError("variant has no stats-eligible train clips")
    n_mels = config.n_mels
    sums = np.zeros(n_mels, dtype=np.float64)
    sums2 = np.zeros(n_mels, dtype=np.float64)
    counts = np.zeros(n_mels, dtype=np.int64)
    floor_db = 10.0 * np.log10(config.amin) + 1e-6
    for row in eligible.itertuples(index=False):
        power = cache[:, int(row.start_frame):int(row.start_frame) + int(row.n_frames)]
        db = to_db(power, config)
        for band in range(n_mels):
            values = db[band, db[band] > floor_db].astype(np.float64)
            sums[band] += values.sum(dtype=np.float64)
            sums2[band] += np.square(values).sum(dtype=np.float64)
            counts[band] += values.size
    if np.any(counts == 0):
        raise RuntimeError("variant statistics have a mel band with no above-floor train cells")
    mean = sums / counts
    variance = np.maximum(sums2 / counts - mean * mean, 0.0)
    std = np.maximum(np.sqrt(variance), 1e-6)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}, int(len(index) - len(eligible))


def _filtered_val(folder: Path, val_manifest: pd.DataFrame, config,
                  raw_binary: Path) -> dict[str, np.ndarray]:
    stored = {}
    for key in FILTERS:
        array_path = folder / f"validation_{key}.npy"
        if array_path.is_file():
            arr = np.load(array_path, mmap_mode="r", allow_pickle=False)
            if arr.shape != (len(val_manifest), config.n_mels, config.n_win):
                raise RuntimeError(f"filtered validation cache shape mismatch: {array_path.name}")
            stored[key] = arr
            continue
        powers = []
        for source in val_manifest.itertuples(index=False):
            audio, sr = load_audio(raw_binary / str(source.clip_id), config)
            audio = standardize(audio, sr, config)
            audio = stress_filter(audio, config.sr, key)
            fixed, status = fix_length(audio, config)
            if not fixed.size:
                raise RuntimeError(f"{source.clip_id}: invalid under {key}: {status}")
            powers.append(centre_window(mel_power(fixed, config), config))
        arr = np.stack(powers).astype(np.float32)
        temp = array_path.with_suffix(".npy.tmp")
        with temp.open("wb") as stream:
            np.save(stream, arr, allow_pickle=False)
        os.replace(temp, array_path)
        stored[key] = np.load(array_path, mmap_mode="r", allow_pickle=False)
    return stored


def _fit_variant_seed(seed: int, name: str, config, stats: dict, train_cache: np.ndarray,
                      train_index: pd.DataFrame, y_train: np.ndarray,
                      val_powers: dict[str, np.ndarray], y_val: np.ndarray,
                      lineage_hash: str) -> dict:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    params = {"stage": "B2", "variant": name, "front_end": params_hash(config),
              "lineage_hash": lineage_hash, "mode": "global_stats", "gain_augmentation": True}
    config_id = hashlib.sha256(json.dumps(params, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    run_id = f"{config_id}_seed{seed}"
    folder = ROOT / "results/models/30"
    folder.mkdir(parents=True, exist_ok=True)
    artifact = folder / f"fit_{run_id}.pt"
    ledger_path = ROOT / "results/tables/runs_30.csv"
    score_path = ROOT / "results/tables/val_scores_30.npz"
    if artifact.is_file() and ledger_path.is_file() and score_path.is_file():
        ledger = pd.read_csv(ledger_path, keep_default_na=False)
        hit = ledger[(ledger.configuration_id == config_id) & (ledger.seed.astype(str) == str(seed))]
        with np.load(score_path, allow_pickle=False) as archive:
            has_score = run_id in archive.files
            saved = archive[run_id].astype(np.float64) if has_score else None
        if len(hit) and has_score and hit.iloc[-1].artefact_sha256 == sha256_file(artifact):
            payload = torch.load(artifact, map_location="cpu", weights_only=True)
            model = build_cnn((1, config.n_mels, config.n_win)); model.load_state_dict(payload["state_dict"])
            result = _evaluate(model, val_powers, y_val, config, stats)
            if not np.allclose(result["validation_scores"]["gain_+0db"], saved, rtol=0, atol=1e-6):
                raise RuntimeError(f"resume score mismatch for {run_id}")
            result.update({"seed": seed, "variant": name, "seconds": float(hit.iloc[-1].seconds),
                           "artifact": str(artifact), "run_id": run_id, "resumed": True,
                           "parameter_count": parameter_count(model)})
            return result
    model = build_cnn((1, config.n_mels, config.n_win))
    def train_batch(indices, rng):
        windows = np.stack([get_window(train_cache, train_index.iloc[int(i)], config,
                                       rng=rng, split="train") for i in indices]).astype(np.float32)
        gains = rng.uniform(config.gain_range_db[0], config.gain_range_db[1], size=len(indices))
        windows *= np.power(10.0, gains / 10.0).astype(np.float32)[:, None, None]
        return _batch_transform(windows, config, stats)
    val_base = _batch_transform(val_powers["unfiltered"], config, stats)
    fit = train_torch_model(model, train_batch, val_base, y_train, y_val, seed=seed,
                            batch_size=64, learning_rate=1e-3, weight_decay=1e-2,
                            max_epochs=100, patience=8, device="cpu", expected_threads=4,
                            lineage_hash=lineage_hash)
    model = fit["model"]
    result = _evaluate(model, val_powers, y_val, config, stats)
    temporary = artifact.with_suffix(".pt.tmp")
    torch.save({"state_dict": fit["state_dict"], "lineage_hash": lineage_hash,
                "seed": seed, "stage": "B2", "variant": name}, temporary)
    os.replace(temporary, artifact)
    row = {"notebook": 30, "stage": "B2", "configuration_id": config_id,
           "seed": seed, "validation_f1": result["base_metrics"]["f1"],
           "validation_roc_auc": result["base_metrics"]["roc_auc"],
           "threshold": result["threshold"], "seconds": fit["training_seconds"],
           "parameter_count": parameter_count(model), "lineage_hash": lineage_hash,
           "device": "cpu", "thread_count": 4, "bit_reproducible": not bool(fit["warnings"]),
           "warnings": json.dumps(fit["warnings"]),
           "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "artefact_path": str(artifact.relative_to(ROOT)), "artefact_sha256": sha256_file(artifact),
           "run_id": run_id}
    append_run(ROOT, 30, row, result["validation_scores"]["gain_+0db"])
    result.update({"seed": seed, "variant": name, "seconds": fit["training_seconds"],
                   "artifact": str(artifact), "run_id": run_id, "resumed": False,
                   "parameter_count": parameter_count(model), "best_epoch": fit["best_epoch"],
                   "warnings": fit["warnings"]})
    return result


def main() -> int:
    if os.cpu_count() != 4:
        raise RuntimeError(f"registered CNN runs expect four logical CPUs; found {os.cpu_count()}")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    base = load_config(ROOT / "config.yaml")
    if base.mode != "global_stats" or params_hash(base) != \
            "97544883461de11a1ee51cc17d7028431d6d0ea8f175effc8e63bc18595f28fc":
        raise RuntimeError("B2 must start from the tagged project front end")
    b1 = json.loads((ROOT / "results/tables/run_report_30_local_b1.json").read_text(encoding="utf-8"))
    if b1.get("selection", {}).get("mode") != "global_stats":
        raise RuntimeError("B2 winner differs from this B1 runner's required starting mode")
    base_candidate = next(x for x in b1["mode_candidates"] if x["mode"] == "global_stats")
    train_manifest, val_manifest = _manifest_rows("train"), _manifest_rows("val")
    raw_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw_binary = _raw_binary_root(Path(raw_cfg["paths"]["local"]["raw_root"]).resolve())
    y_train = train_manifest.label.to_numpy(dtype=np.float32)
    y_val = val_manifest.label.to_numpy(dtype=np.int8)
    results = []
    variant_values = []
    for spec in VARIANTS:
        cfg = replace(base, **spec["overrides"], mode="global_stats")
        folder = _variant_dir(cfg)
        print(json.dumps({"event": "variant_cache_start", "name": spec["name"],
                          "cache_content_hash": cache_content_hash(cfg), "folder": str(folder)},), flush=True)
        train_cache, train_index = _build_split("train", train_manifest, cfg, raw_binary, folder)
        val_cache, val_index = _build_split("val", val_manifest, cfg, raw_binary, folder)
        if not np.array_equal(train_index.clip_id.to_numpy(), train_manifest.clip_id.to_numpy()):
            raise RuntimeError(f"{spec['name']}: train order mismatch")
        if not np.array_equal(val_index.clip_id.to_numpy(), val_manifest.clip_id.to_numpy()):
            raise RuntimeError(f"{spec['name']}: validation order mismatch")
        stats, excluded = _fit_stats(train_cache, train_index, cfg)
        lineage = config_hash(cfg, stats)
        val_power = np.stack([get_window(val_cache, row, cfg, split="val")
                              for row in val_index.itertuples(index=False)]).astype(np.float32)
        val_powers = {"unfiltered": val_power, **_filtered_val(folder, val_manifest, cfg, raw_binary)}
        stats_doc = {"mean": stats["mean"].tolist(), "std": stats["std"].tolist(),
                     "clips_excluded": excluded, "stats_eligible_clips": int(len(train_index)-excluded),
                     "lineage_hash": lineage, "cache_content_hash": cache_content_hash(cfg)}
        (folder / "normalization_stats_global.json").write_text(json.dumps(stats_doc, indent=2)+"\n", encoding="utf-8")
        np.savez_compressed(folder / "norm_stats_global.npz", mean=stats["mean"].astype("<f4"),
                            std=stats["std"].astype("<f4"), clips_excluded=np.asarray(excluded, np.int64))
        seed_rows = []
        for seed in SEEDS:
            print(json.dumps({"event": "fit_start", "variant": spec["name"], "seed": seed}), flush=True)
            seed_rows.append(_fit_variant_seed(seed, spec["name"], cfg, stats, train_cache,
                                               train_index, y_train, val_powers, y_val, lineage))
        zero = [r["condition_f1"]["gain_+0db"] for r in seed_rows]
        worst = [r["worst_condition_f1"] for r in seed_rows]
        candidate = {"variant": spec["name"], "overrides": spec["overrides"],
                     "cache_content_hash": cache_content_hash(cfg), "lineage_hash": lineage,
                     "statistics_excluded_clips": excluded,
                     "mean_f1_0db": float(np.mean(zero)),
                     "sd_f1_0db": float(np.std(zero, ddof=1)),
                     "score": float(np.mean(worst)), "sd_f1": float(np.std(worst, ddof=1)),
                     "per_seed_f1_0db": zero, "per_seed_worst_condition_f1": worst,
                     "condition_f1_by_seed": [r["condition_f1"] for r in seed_rows]}
        variant_values.append(candidate)
        results.append({"candidate": candidate,
                        "seeds": [{k:v for k,v in r.items() if k != "validation_scores"} for r in seed_rows]})
        del train_cache, val_cache, val_power, val_powers

    selection = select_variant(base_candidate, variant_values)
    combination = None
    status = "PASS"
    if len(selection["accepted"]) >= 2:
        merged: dict[str, object] = {}
        conflicts = []
        for candidate in selection["accepted"]:
            for key, value in candidate["overrides"].items():
                if key in merged and merged[key] != value:
                    conflicts.append({"key": key, "values": [merged[key], value],
                                      "variants": [candidate["variant"]]})
                merged[key] = value
        if conflicts:
            status = "STOP_CONFLICTING_ACCEPTED_VARIANTS"
            combination = {"status": status, "conflicts": conflicts,
                           "accepted_variants": [x["variant"] for x in selection["accepted"]]}
        else:
            name = "combination_" + "_".join(x["variant"] for x in selection["accepted"])
            cfg = replace(base, **merged, mode="global_stats")
            folder = _variant_dir(cfg)
            train_cache, train_index = _build_split("train", train_manifest, cfg, raw_binary, folder)
            val_cache, val_index = _build_split("val", val_manifest, cfg, raw_binary, folder)
            stats, excluded = _fit_stats(train_cache, train_index, cfg)
            lineage = config_hash(cfg, stats)
            val_power = np.stack([get_window(val_cache, row, cfg, split="val")
                                  for row in val_index.itertuples(index=False)]).astype(np.float32)
            val_powers = {"unfiltered": val_power,
                          **_filtered_val(folder, val_manifest, cfg, raw_binary)}
            seed_rows = []
            for seed in SEEDS:
                print(json.dumps({"event": "fit_start", "variant": name, "seed": seed}), flush=True)
                seed_rows.append(_fit_variant_seed(seed, name, cfg, stats, train_cache,
                                                   train_index, y_train, val_powers, y_val, lineage))
            combo_zero = [r["condition_f1"]["gain_+0db"] for r in seed_rows]
            combo_worst = [r["worst_condition_f1"] for r in seed_rows]
            combo_candidate = {"variant": name, "overrides": merged,
                               "cache_content_hash": cache_content_hash(cfg),
                               "lineage_hash": lineage, "statistics_excluded_clips": excluded,
                               "mean_f1_0db": float(np.mean(combo_zero)),
                               "sd_f1_0db": float(np.std(combo_zero, ddof=1)),
                               "score": float(np.mean(combo_worst)),
                               "sd_f1": float(np.std(combo_worst, ddof=1)),
                               "per_seed_f1_0db": combo_zero,
                               "per_seed_worst_condition_f1": combo_worst,
                               "condition_f1_by_seed": [r["condition_f1"] for r in seed_rows]}
            best_single = selection["accepted"][0]
            passes_combo = (combo_candidate["score"] - best_single["score"] > best_single["sd_f1"]
                            and combo_candidate["score"] - best_single["score"] >= 0.01)
            combination = {"status": "accepted" if passes_combo else "rejected",
                           "merged_overrides": merged, "candidate": combo_candidate,
                           "seeds": [{k: v for k, v in r.items() if k != "validation_scores"}
                                     for r in seed_rows],
                           "best_single_variant": best_single["variant"],
                           "best_single_score": best_single["score"],
                           "best_single_sd": best_single["sd_f1"],
                           "passes_registered_rule": passes_combo}
            if passes_combo:
                selection["winner"] = combo_candidate
            if len(combination["seeds"]) == 3:
                del train_cache, val_cache, val_power, val_powers
    report = {"stage": "B2", "status": status, "test_reads": 0,
              "base_mode": "global_stats", "base_candidate": base_candidate,
              "conditions": list(CONDITIONS), "threshold_rule": "one threshold per seed at unfiltered 0 dB, held fixed for all conditions",
              "variant_candidates": variant_values, "selection": selection,
              "variant_runs": results, "combination": combination,
              "train_clips": len(train_manifest), "validation_clips": len(val_manifest),
              "raw_binary_root": str(raw_binary), "device": "cpu", "torch_threads": 4}
    path, _ = write_run_report(ROOT, "30_local_b2", report)
    print(json.dumps({"event": "B2_complete", "status": status,
                      "accepted": [x["variant"] for x in selection["accepted"]],
                      "winner": selection["winner"].get("variant", "global_stats"),
                      "combination": combination, "report": str(path)}, indent=2, default=str), flush=True)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
