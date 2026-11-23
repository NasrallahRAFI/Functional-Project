"""Run the registered CNN B1 mode comparison and no-gain control locally."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from dronedet.config import config_hash, load_config, params_hash
from dronedet.features import (centre_window, fix_length, get_window, load_audio,
                               load_cache_index, load_power_cache, mel_power,
                               standardize, to_db)
from dronedet.ledger import append_run, sha256_file, write_run_report
from dronedet.metrics import choose_f1_threshold, select_mode
from dronedet.models.cnn import build_cnn, parameter_count
from dronedet.normalization import pcen_transform, stress_filter
from dronedet.training import train_torch_model


ROOT = Path.cwd().resolve()
CPU_COUNT = os.cpu_count()
SEEDS = (42, 43, 44)
GAINS = (-12, -6, 0, 6, 12)
FILTERS = ("highpass_150", "highpass_300", "lowpass_5000")
CONDITIONS = ("gain_-12db", "gain_-6db", "gain_0db", "gain_+6db",
              "gain_+12db", *FILTERS)


def _batch_transform(power: np.ndarray, config, stats: dict[str, float | np.ndarray]) -> np.ndarray:
    x = np.asarray(power, dtype=np.float32)
    if config.mode == "global_stats":
        out = (to_db(x, config) - np.asarray(stats["mean"], np.float32)[None, :, None]) / \
              np.asarray(stats["std"], np.float32)[None, :, None]
    elif config.mode == "per_window_mean":
        db = to_db(x, config)
        out = (db - db.mean(axis=(1, 2), keepdims=True)) / float(stats["std"])
    elif config.mode == "pcen":
        magnitude = np.sqrt(x) * np.float32(config.pcen_input_scale)
        time_frames = config.pcen_time_constant * config.sr / config.hop
        b = float((np.sqrt(1.0 + 4.0 * time_frames**2) - 1.0) /
                  (2.0 * time_frames**2))
        zi = ((1.0 - b) * magnitude.mean(axis=-1, keepdims=True)).astype(np.float32)
        pcen = librosa.pcen(magnitude, sr=config.sr, hop_length=config.hop,
                            time_constant=config.pcen_time_constant, gain=config.pcen_gain,
                            bias=config.pcen_bias, power=config.pcen_power, eps=config.pcen_eps,
                            b=b, zi=zi, axis=-1)
        out = (np.asarray(pcen, np.float32) - float(stats["mean"])) / float(stats["std"])
    else:
        raise ValueError(f"unsupported B1 mode {config.mode}")
    return np.asarray(out[:, None, :, :], dtype=np.float32)


def _fit_mode_stats(mode: str, eligible_power: np.ndarray, config) -> dict:
    if mode == "global_stats":
        with np.load(ROOT / "data/processed/spectrograms/norm_stats_global.npz",
                     allow_pickle=False) as archive:
            return {"mean": archive["mean"].astype(np.float32),
                    "std": archive["std"].astype(np.float32),
                    "clips_excluded": int(archive["clips_excluded"])}
    cfg = replace(config, mode=mode)
    count = 0
    mean = 0.0
    m2 = 0.0
    for start in range(0, len(eligible_power), 128):
        batch = eligible_power[start:start + 128]
        if mode == "per_window_mean":
            values = to_db(batch, cfg)
            values = values - values.mean(axis=(1, 2), keepdims=True)
        elif mode == "pcen":
            values = _batch_transform(batch, cfg, {"mean": 0.0, "std": 1.0})[:, 0]
        else:
            raise ValueError(mode)
        flat = values.astype(np.float64, copy=False).ravel()
        n = flat.size
        batch_mean = float(flat.mean())
        batch_m2 = float(np.square(flat - batch_mean).sum(dtype=np.float64))
        delta = batch_mean - mean
        total = count + n
        m2 += batch_m2 + delta * delta * count * n / total
        mean += delta * n / total
        count = total
    std = float(np.sqrt(m2 / count))
    if not np.isfinite(std) or std <= 0:
        raise RuntimeError(f"invalid {mode} training statistics")
    return {"mean": float(mean), "std": std, "clips_excluded": 88,
            "fit_cells": int(count), "fit_clips": int(len(eligible_power))}


def _validation_windows(config, val_index: pd.DataFrame, val_cache: np.ndarray,
                        val_manifest: pd.DataFrame, raw_root: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    y = val_manifest.label.to_numpy(dtype=np.int8)
    windows = np.stack([get_window(val_cache, row, config, split="val")
                        for row in val_index.itertuples(index=False)]).astype(np.float32)
    outputs = {"unfiltered": windows}
    # The local config points one directory above the extracted archive root.
    binary_root = raw_root / "Binary_Drone_Audio"
    if not binary_root.is_dir():
        nested = raw_root / "DroneAudioDataset-master" / "Binary_Drone_Audio"
        if nested.is_dir():
            binary_root = nested
        else:
            raise FileNotFoundError(f"Binary_Drone_Audio not found under configured raw root {raw_root}")
    for kind in FILTERS:
        filtered = []
        for row in val_manifest.itertuples(index=False):
            path = binary_root / str(row.clip_id)
            audio, sr = load_audio(path, config)
            audio = standardize(audio, sr, config)
            audio = stress_filter(audio, config.sr, kind)
            fixed, status = fix_length(audio, config)
            if not fixed.size:
                raise RuntimeError(f"validation clip {row.clip_id} became invalid under {kind}: {status}")
            filtered.append(centre_window(mel_power(fixed, config), config))
        outputs[kind] = np.stack(filtered).astype(np.float32)
    return outputs, y


def _condition_powers(windows: np.ndarray, condition: str) -> np.ndarray:
    if condition == "unfiltered":
        return windows
    if condition in FILTERS:
        raise ValueError("filtered condition windows must be selected from their own front-end output")
    gain_db = int(condition.removeprefix("gain_").removesuffix("db"))
    return windows * np.float32(10.0 ** (gain_db / 10.0))


def _scores(model, powers: np.ndarray, config, stats: dict, *, batch_size: int = 96) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(powers), batch_size):
            transformed = _batch_transform(powers[start:start + batch_size], config, stats)
            logits = model(torch.from_numpy(transformed)).cpu().numpy().astype(np.float64)
            parts.append(logits)
    return np.concatenate(parts)


def _f1_at(y: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    return float(f1_score(y, scores >= threshold, zero_division=0))


def _evaluate(model, powers: dict[str, np.ndarray], y: np.ndarray, config, stats: dict) -> dict:
    result = {}
    for gain in GAINS:
        key = f"gain_{gain:+d}db"
        values = _scores(model, _condition_powers(powers["unfiltered"], key), config, stats)
        result[key] = values
    for key in FILTERS:
        result[key] = _scores(model, powers[key], config, stats)
    base = result["gain_+0db"]
    selected = choose_f1_threshold(y, base)
    f1s = {key: _f1_at(y, values, selected["threshold"]) for key, values in result.items()}
    return {"threshold": selected["threshold"], "base_metrics": selected,
            "condition_f1": f1s, "worst_condition_f1": min(f1s.values()),
            "validation_scores": result}


def _load_a0_state(seed: int, config_id: str) -> dict:
    path = ROOT / "results/models/30" / f"fit_{config_id}_seed{seed}.pt"
    with torch.serialization.safe_globals([]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["lineage_hash"] != "327b5e0c54632f3c7e40533d2d0b521faa876cbaf58b8dd8c5bfa57fe9f6b8f0":
        raise RuntimeError(f"A0 seed {seed} has an unexpected statistics lineage")
    model = build_cnn((1, 64, 28))
    model.load_state_dict(payload["state_dict"])
    return model


def _train_seed(seed: int, mode: str, use_gain: bool, config, stats: dict,
                train_index: pd.DataFrame, train_cache: np.ndarray, y_train: np.ndarray,
                val_powers: dict[str, np.ndarray], y_val: np.ndarray,
                lineage_hash: str) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    input_shape = (1, config.n_mels, config.n_win)
    model = build_cnn(input_shape)
    config_id = hashlib.sha256(json.dumps(
        {"stage": "B1_control" if not use_gain else "B1", "architecture": "A0",
         "mode": mode, "gain_augmentation": use_gain, "lineage_hash": lineage_hash},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    run_id = f"{config_id}_seed{seed}"
    artifact = ROOT / "results/models/30" / f"fit_{run_id}.pt"
    ledger_path = ROOT / "results/tables/runs_30.csv"
    scores_path = ROOT / "results/tables/val_scores_30.npz"
    if ledger_path.is_file() and scores_path.is_file() and artifact.is_file():
        ledger = pd.read_csv(ledger_path, keep_default_na=False)
        hit = ledger[(ledger.configuration_id == config_id) & (ledger.seed.astype(str) == str(seed))]
        with np.load(scores_path, allow_pickle=False) as archive:
            stored = run_id in archive.files
        if len(hit) and hit.iloc[-1].artefact_sha256 == sha256_file(artifact) and stored:
            with np.load(scores_path, allow_pickle=False) as archive:
                base_scores = archive[run_id].astype(np.float64)
            payload = torch.load(artifact, map_location="cpu", weights_only=True)
            model.load_state_dict(payload["state_dict"])
            eval_result = _evaluate(model, val_powers, y_val, config, stats)
            if not np.allclose(eval_result["validation_scores"]["gain_+0db"], base_scores,
                               rtol=0, atol=1e-6):
                raise RuntimeError(f"resumed B1 validation scores differ for {run_id}")
            eval_result.update({"seed": seed, "mode": mode, "seconds": float(hit.iloc[-1].seconds),
                                "artifact": str(artifact), "run_id": run_id, "resumed": True,
                                "parameter_count": parameter_count(model)})
            return eval_result
    def train_batch(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        power = np.stack([get_window(train_cache, train_index.iloc[int(i)], config,
                                     rng=rng, split="train") for i in indices]).astype(np.float32)
        if use_gain:
            db_gain = rng.uniform(config.gain_range_db[0], config.gain_range_db[1], len(indices))
            power *= np.power(10.0, db_gain / 10.0).astype(np.float32)[:, None, None]
        return _batch_transform(power, config, stats)
    val_base = _batch_transform(val_powers["unfiltered"], config, stats)
    fit = train_torch_model(model, train_batch, val_base,
                            y_train, y_val, seed=seed, batch_size=64, learning_rate=1e-3,
                            weight_decay=1e-2, max_epochs=100, patience=8, device="cpu",
                            expected_threads=CPU_COUNT, lineage_hash=lineage_hash)
    started = time.perf_counter()
    eval_result = _evaluate(fit["model"], val_powers, y_val, config, stats)
    elapsed = float(fit["training_seconds"])
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_suffix(".pt.tmp")
    torch.save({"state_dict": fit["state_dict"], "lineage_hash": lineage_hash,
                "seed": seed, "mode": mode, "stage": "B1" if use_gain else "B1_control"}, temporary)
    os.replace(temporary, artifact)
    run_row = {"notebook": 30, "stage": "B1" if use_gain else "B1_control",
               "configuration_id": config_id, "seed": seed,
               "validation_f1": eval_result["base_metrics"]["f1"],
               "validation_roc_auc": eval_result["base_metrics"]["roc_auc"],
               "threshold": eval_result["threshold"], "seconds": elapsed,
               "parameter_count": parameter_count(fit["model"]), "lineage_hash": lineage_hash,
               "device": "cpu", "thread_count": CPU_COUNT,
               "bit_reproducible": not bool(fit["warnings"]),
               "warnings": json.dumps(fit["warnings"]),
               "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                        cwd=ROOT, text=True).strip(),
               "artefact_path": str(artifact.relative_to(ROOT)),
               "artefact_sha256": sha256_file(artifact), "run_id": run_id}
    append_run(ROOT, 30, run_row, eval_result["validation_scores"]["gain_+0db"])
    eval_result.update({"seed": seed, "mode": mode, "seconds": elapsed,
                        "artifact": str(artifact), "run_id": run_id, "resumed": False,
                        "parameter_count": parameter_count(fit["model"]),
                        "best_epoch": fit["best_epoch"], "warnings": fit["warnings"]})
    return eval_result


def main() -> int:
    if CPU_COUNT != 4:
        raise RuntimeError(f"registered CNN run expects four logical CPUs; found {CPU_COUNT}")
    torch.set_num_threads(CPU_COUNT)
    torch.use_deterministic_algorithms(True, warn_only=True)
    config = load_config(ROOT / "config.yaml")
    if config.mode != "global_stats" or params_hash(config) != \
            "97544883461de11a1ee51cc17d7028431d6d0ea8f175effc8e63bc18595f28fc":
        raise RuntimeError("B1 must start from the tagged project front end")
    acceptance = json.loads((ROOT / "results/tables/cnn_b1_pcen_acceptance.json").read_text())
    if acceptance.get("status") != "pass" or acceptance.get("eligible_train_windows") != 7519:
        raise RuntimeError("PCEN acceptance artifact is missing or not production-faithful")
    p = {"root": ROOT, "spec": ROOT / "data/processed/spectrograms",
         "splits": ROOT / "data/processed/splits"}
    train_manifest = pd.read_csv(p["splits"] / "train.csv", keep_default_na=False)
    val_manifest = pd.read_csv(p["splits"] / "val.csv", keep_default_na=False)
    train_index = load_cache_index(p, "train")
    val_index = load_cache_index(p, "val")
    train_cache = load_power_cache(p, "train", mmap_mode="r")
    val_cache = load_power_cache(p, "val", mmap_mode="r")
    if not np.array_equal(train_manifest.clip_id.to_numpy(), train_index.clip_id.to_numpy()):
        raise RuntimeError("train manifest/cache order mismatch")
    if not np.array_equal(val_manifest.clip_id.to_numpy(), val_index.clip_id.to_numpy()):
        raise RuntimeError("validation manifest/cache order mismatch")
    eligible = train_index[train_index.stats_eligible.astype(bool)]
    if len(eligible) != 7519:
        raise RuntimeError(f"stored stats_eligible count mismatch: {len(eligible)}")
    eligible_power = np.stack([centre_window(
        train_cache[:, int(row.start_frame):int(row.start_frame)+int(row.n_frames)], config)
        for row in eligible.itertuples(index=False)]).astype(np.float32)
    # Use the actual extracted dataset root corresponding to the locally configured raw root.
    configured_raw = Path(r"C:\Users\Nasrallah\Downloads\DroneAudioDataset-master")
    val_powers, y_val = _validation_windows(config, val_index, val_cache, val_manifest,
                                             configured_raw)
    y_train = train_manifest.label.to_numpy(dtype=np.float32)
    print(json.dumps({"event": "B1_inputs_ready", "eligible_train_stats_clips": len(eligible),
                      "train_clips": len(train_manifest), "validation_clips": len(val_manifest),
                      "conditions": list(CONDITIONS), "test_reads": 0,
                      "filtered_validation_windows": {k: len(v) for k, v in val_powers.items()}}, indent=2), flush=True)
    mode_results = {}
    mode_stats = {}
    for mode in ("global_stats", "per_window_mean", "pcen"):
        mode_config = replace(config, mode=mode)
        stats = _fit_mode_stats(mode, eligible_power, mode_config)
        mode_stats[mode] = stats
        lineage = config_hash(mode_config, stats)
        if mode == "global_stats":
            stage_a_id = hashlib.sha256(json.dumps(
                {"stage": "A", "architecture": "A0", "mode": "global_stats",
                 "gain_augmentation": True, "lineage_hash": lineage},
                sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            rows = []
            for seed in SEEDS:
                model = _load_a0_state(seed, stage_a_id)
                score = _evaluate(model, val_powers, y_val, mode_config, stats)
                score.update({"seed": seed, "mode": mode, "seconds": None,
                              "run_id": f"{stage_a_id}_seed{seed}", "resumed": True,
                              "parameter_count": parameter_count(model),
                              "artifact": str(ROOT / "results/models/30" /
                                              f"fit_{stage_a_id}_seed{seed}.pt")})
                rows.append(score)
            mode_results[mode] = rows
        else:
            rows = []
            for seed in SEEDS:
                print(json.dumps({"event": "fit_start", "mode": mode, "seed": seed},),
                      flush=True)
                rows.append(_train_seed(seed, mode, True, mode_config, stats, train_index,
                                        train_cache, y_train, val_powers, y_val, lineage))
            mode_results[mode] = rows

    candidate_summary = []
    for mode, rows in mode_results.items():
        zero = [r["condition_f1"]["gain_+0db"] for r in rows]
        worst = [r["worst_condition_f1"] for r in rows]
        candidate_summary.append({"mode": mode, "mean_f1_0db": float(np.mean(zero)),
                                  "sd_f1_0db": float(np.std(zero, ddof=1)),
                                  "score": float(np.mean(worst)),
                                  "sd_f1": float(np.std(worst, ddof=1)),
                                  "per_seed_f1_0db": zero,
                                  "per_seed_worst_condition_f1": worst,
                                  "condition_f1_by_seed": [r["condition_f1"] for r in rows]})
    selection = select_mode(candidate_summary)
    control_config = replace(config, mode="global_stats")
    global_stats = mode_stats["global_stats"]
    control_lineage = config_hash(control_config, global_stats)
    controls = []
    for seed in SEEDS:
        controls.append(_train_seed(seed, "global_stats", False, control_config, global_stats,
                                    train_index, train_cache, y_train, val_powers, y_val,
                                    control_lineage))
    fit_times = [r["seconds"] for mode in mode_results.values() for r in mode
                 if r["seconds"] is not None]
    report = {"stage": "B1", "status": "PASS", "test_reads": 0,
              "frontend_params_hash": params_hash(config),
              "stats_eligible_train_clips": len(eligible), "excluded_train_clips": 88,
              "pcen_acceptance": acceptance,
              "conditions": list(CONDITIONS), "threshold_rule": "one threshold per seed at unfiltered 0 dB, held fixed for all conditions",
              "mode_candidates": candidate_summary, "selection": selection,
              "mode_fit_results": {mode: [{k: v for k, v in row.items()
                  if k not in {"validation_scores"}} for row in rows]
                  for mode, rows in mode_results.items()},
              "b1_control_no_gain": [{k: v for k, v in row.items()
                                      if k not in {"validation_scores"}} for row in controls],
              "logstft": "not run (deferred by the closeout)",
              "time": {"mean_seconds_per_new_fit": float(np.mean(fit_times)) if fit_times else None,
                       "max_seconds_per_new_fit": float(np.max(fit_times)) if fit_times else None},
              "device": "cpu", "torch_threads": CPU_COUNT,
              "validation_raw_root_used": str(configured_raw),
              "filtered_conditions_filter_whole_clip_before_frontend": True}
    path, _ = write_run_report(ROOT, "30_local_b1", report)
    print(json.dumps({"event": "B1_complete", "selection": selection,
                      "candidates": candidate_summary,
                      "control_worst": [r["worst_condition_f1"] for r in controls],
                      "report": str(path)}, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
