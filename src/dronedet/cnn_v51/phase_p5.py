"""V5.1 train/validation power caches and stateless power augmentation."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from scipy import signal
from scipy.signal import periodogram

from dronedet.features import load_wave, mel_centres
from dronedet.normalization import Normalizer
from dronedet.config import cache_content_hash, transform_hash

from .artifacts import canonical_json_bytes, contract_columns, write_csv
from .identity import standardized_sample_sha256
from .randomness import canonical_key, integer, keyed_sha1, ranked, uniform, uniform_range

CONTRACT_PATH = Path(__file__).resolve().parents[3] / "docs" / "v5.1" / "DATA_PREPARATION_CNN_V5_1_CONTRACT.json"
_RNG_PURPOSES = {
    "aug_gain", "aug_mix_flag", "aug_donor_group", "aug_donor_clip", "aug_bed_gain",
    "aug_snr", "aug_response_flag", "aug_hp", "aug_lp_flag", "aug_lp", "aug_tilt",
    "aug_ripple", "aug_floor_flag", "aug_floor_rms", "crop_base", "crop_donor",
}
_MIX_PAIR_PARALLEL_THRESHOLD = 1_000_000


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _band_power(audio: np.ndarray, sr: int) -> float:
    f, density = periodogram(np.asarray(audio, dtype=np.float64), fs=sr, window="boxcar",
                             detrend=False, return_onesided=True, scaling="density")
    return float(np.sum(density[(f >= 100.0) & (f <= 4000.0)] * (sr / len(audio)), dtype=np.float64))


def _rms_dbfs(audio: np.ndarray) -> float:
    x = np.asarray(audio, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x * x, dtype=np.float64)))
    return -240.0 if rms == 0.0 else 20.0 * math.log10(rms)


def _mix_pair_rank(pair: tuple[dict[str, Any], dict[str, Any]]) -> tuple[str, str]:
    key = [str(pair[0]["clip_id"]), str(pair[1]["clip_id"]), 0]
    return keyed_sha1("mix_check", key), canonical_key(key)


def _top_mix_pairs_for_bases(args):
    """Return at most 200 ranked pairs for one bounded base-row chunk."""
    bases, donors = args
    pairs = ((base, donor) for base in bases for donor in donors
             if str(base["group_key"]) != str(donor["group_key"]))
    return heapq.nsmallest(200, pairs, key=_mix_pair_rank)


def _a_check_pairs(train_rows: Iterable[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows = sorted((dict(row) for row in train_rows
                   if row.get("split") == "train" and row.get("kept") is True),
                  key=lambda r: str(r["clip_id"]))
    donors = sorted((row for row in rows if int(row["label"]) == 0
                     and row.get("negative_stratum") in {"N1", "N5", "N6", "N7"}),
                    key=lambda r: str(r["clip_id"]))

    # Rank every eligible pair exactly, but bound each worker to a small base
    # chunk and its best 200. The global top 200 must be among these local top
    # 200 lists, so merging them preserves the contract's full ordering.
    common_groups = {str(row["group_key"]) for row in rows} & {str(row["group_key"]) for row in donors}
    eligible_count = len(rows) * len(donors) - sum(
        sum(str(row["group_key"]) == group for row in rows)
        * sum(str(row["group_key"]) == group for row in donors)
        for group in common_groups
    )
    workers = min(8, os.cpu_count() or 1, len(rows))
    if workers <= 1 or eligible_count < _MIX_PAIR_PARALLEL_THRESHOLD:
        return _top_mix_pairs_for_bases((rows, donors))

    chunk_size = (len(rows) + workers - 1) // workers
    chunks = [(rows[start:start + chunk_size], donors)
              for start in range(0, len(rows), chunk_size)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        local_top = [pair for result in pool.map(_top_mix_pairs_for_bases, chunks)
                     for pair in result]
    return heapq.nsmallest(200, local_top, key=_mix_pair_rank)


def build_p5_cache(
    train_rows: Iterable[dict[str, Any]], val_rows: Iterable[dict[str, Any]],
    source_inventory_rows: Iterable[dict[str, Any]], *, sealed_source_file_ids: Iterable[str],
    train_manifest_sha256: str, val_manifest_sha256: str, frozen_membership_sha256: str, config,
    path_resolver: Callable[[dict[str, Any]], str | Path], loader=load_wave,
) -> dict[str, Any]:
    """Re-derive permitted train/val powers; never opens test/OOD files."""
    manifests = [dict(row) for row in train_rows] + [dict(row) for row in val_rows]
    if not manifests or any(r.get("split") not in {"train", "val"} or r.get("kept") is not True for r in manifests):
        raise ValueError("P5 requires retained frozen train and validation rows")
    if len({str(r["clip_id"]) for r in manifests}) != len(manifests):
        raise ValueError("P5 manifest clip IDs must be unique")
    sealed = {str(value) for value in sealed_source_file_ids}
    inventory_rows = [dict(row) for row in source_inventory_rows]
    inventory = {str(row["source_file_id"]): row for row in inventory_rows}
    if len(inventory) != len(inventory_rows):
        raise ValueError("P5 source inventory contains duplicate files")
    by_file: dict[str, list[dict[str, Any]]] = {}
    for row in manifests:
        source_id = str(row["source_file_id"])
        if source_id in sealed:
            raise PermissionError("P5 refuses to open a source represented in frozen test/OOD")
        source = inventory.get(source_id)
        if source is None or any(row.get(k) != source.get(k) for k in ("source", "source_revision", "file_sha256")):
            raise ValueError(f"P5 manifest/inventory mismatch: {row['clip_id']}")
        by_file.setdefault(source_id, []).append(row)

    outputs: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    caches: dict[str, tuple[np.ndarray, list[dict[str, Any]]]] = {}
    lookup: list[dict[str, Any]] = []
    train_manifests = [dict(row) for row in manifests if row["split"] == "train"]
    val_manifests = [dict(row) for row in manifests if row["split"] == "val"]
    a_pairs = _a_check_pairs(train_manifests)
    a_ids = {str(row["clip_id"]) for pair in a_pairs for row in pair}
    a_waveforms: dict[str, np.ndarray] = {}
    manifest_hash = {"train": train_manifest_sha256, "val": val_manifest_sha256}
    for split in ("train", "val"):
        if len(manifest_hash[split]) != 64 or any(c not in "0123456789abcdef" for c in manifest_hash[split]):
            raise ValueError(f"P5 {split} manifest hash must be lowercase SHA-256")
    for source_id in sorted(by_file):
        source = inventory[source_id]
        path = Path(path_resolver(source))
        if _hash_file(path) != str(source["file_sha256"]):
            raise ValueError(f"P5 raw file hash mismatch: {source_id}")
        audio = np.asarray(loader(path, config), dtype=np.float32)
        if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
            raise ValueError(f"P5 loader returned invalid waveform: {source_id}")
        for row in sorted(by_file[source_id], key=lambda r: str(r["clip_id"])):
            start, count = int(row["start_sample_16k"]), int(row["n_samples"])
            if row["sample_rate"] != config.sr or start < 0 or count <= 0 or start + count > audio.size:
                raise ValueError(f"P5 waveform coordinates invalid: {row['clip_id']}")
            waveform = np.ascontiguousarray(audio[start:start + count], dtype=np.float32)
            if standardized_sample_sha256(waveform) != row["sample_sha256"]:
                raise ValueError(f"P5 frozen sample hash mismatch: {row['clip_id']}")
            # Apply unchanged V4 fixed-length handling to the frozen clip samples.
            from dronedet.features import fix_length, mel_power
            fixed, _ = fix_length(waveform, config)
            if not fixed.size:
                raise ValueError(f"P5 frozen clip is below frontend minimum: {row['clip_id']}")
            crop_power = mel_power(fixed, config)
            split_name = str(row["split"])
            outputs[split_name].append({"clip_id": str(row["clip_id"]), "split": split_name,
                                        "manifest_sha256": manifest_hash[split_name],
                                        "sample_sha256": str(row["sample_sha256"]), "power": crop_power,
                                        "rms_window_full_dbfs": _rms_dbfs(fixed)})
            if split_name == "train":
                if str(row["clip_id"]) in a_ids:
                    a_waveforms[str(row["clip_id"])] = fixed.copy()
                frames = crop_power.shape[1]
                if frames < config.n_win:
                    raise ValueError("P5 feature cache has fewer than n_win frames")
                # Crop offsets are the contract's frame-aligned window starts on fixed audio.
                crop_len = int(config.win_samples)
                for frame in range(frames - config.n_win + 1):
                    sample_start = frame * config.hop
                    window = fixed[sample_start:sample_start + crop_len]
                    if window.size != crop_len:
                        continue
                    lookup.append({"clip_id": str(row["clip_id"]), "manifest_sha256": manifest_hash["train"],
                                   "sample_sha256": str(row["sample_sha256"]), "crop_start_frame": frame,
                                   "crop_start_sample_fixed_waveform": sample_start,
                                   "band_100_4000_power": _band_power(window, config.sr),
                                   "rms_window_full_dbfs": _rms_dbfs(window)})
            del waveform, fixed
        del audio

    for split in ("train", "val"):
        outputs[split].sort(key=lambda r: r["clip_id"])
        caches[split] = pack_power_cache(outputs[split], config.n_mels)
    train_items = {item["clip_id"]: item["power"] for item in outputs["train"]}
    norm = Normalizer.fit(train_manifests, config, train_items)
    lookup.sort(key=lambda row: (row["clip_id"], row["crop_start_frame"]))
    a_check = a_check_mix_approximation(train_manifests, a_waveforms, config=config)
    noise_template = generate_noise_template(config)
    return {"caches": caches, "normalizer": norm, "band_lookup": lookup,
            "train_rows": train_manifests, "val_rows": val_manifests,
            "a_check": a_check, "noise_template": noise_template,
            "bindings": {"frozen_membership_sha256": frozen_membership_sha256,
                         "train_manifest_sha256": train_manifest_sha256,
                         "val_manifest_sha256": val_manifest_sha256,
                         "transform_sha256": transform_hash(config),
                         "cache_content_sha256": cache_content_hash(config)}}


def pack_power_cache(items: list[dict[str, Any]], n_mels: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not items:
        raise ValueError("cannot pack an empty split power cache")
    items = sorted(items, key=lambda r: r["clip_id"])
    cursor, powers, index = 0, [], []
    for item in items:
        p = np.asarray(item["power"], dtype=np.float32)
        if p.ndim != 2 or p.shape[0] != n_mels or p.shape[1] < 28 or not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError(f"invalid cached mel power: {item['clip_id']}")
        index.append({key: item[key] for key in ("clip_id", "split", "manifest_sha256", "sample_sha256")} |
                     {"start_frame": cursor, "n_frames": int(p.shape[1])})
        powers.append(p)
        cursor += p.shape[1]
    return np.concatenate(powers, axis=1), index


def _aug_key(epoch: int, draw_index: int, clip_id: str, parameter: str, component: int = 0) -> list[Any]:
    if epoch < 0 or draw_index < 0 or component < 0 or not clip_id or not parameter:
        raise ValueError("augmentation key components are invalid")
    return [int(epoch), int(draw_index), str(clip_id), str(parameter), int(component)]


def generate_noise_template(config) -> dict[str, Any]:
    """Build the Box-Muller keyed 16 kHz/64 s synthetic frontend floor template."""
    n_samples = config.sr * 64
    if n_samples % 2:
        raise ValueError("synthetic-noise Box-Muller sequence requires an even sample count")
    wave = np.empty(n_samples, dtype=np.float64)
    for pair_index in range(n_samples // 2):
        u0 = uniform("synthetic_noise", [pair_index, 0])
        u1 = uniform("synthetic_noise", [pair_index, 1])
        radius = math.sqrt(-2.0 * math.log(u0))
        angle = 2.0 * math.pi * u1
        wave[2 * pair_index] = radius * math.cos(angle)
        wave[2 * pair_index + 1] = radius * math.sin(angle)
    wave = wave.astype(np.float32)
    wave -= np.float32(wave.mean(dtype=np.float64))
    wave /= np.float32(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
    from dronedet.features import mel_power
    matrix = mel_power(wave, config)
    template = matrix.mean(axis=1, dtype=np.float64).astype(np.float32)
    raw_hash = hashlib.sha256(np.ascontiguousarray(wave, dtype="<f4").tobytes()).hexdigest()
    array_hash = hashlib.sha256(np.ascontiguousarray(template, dtype="<f4").tobytes()).hexdigest()
    return {"template": template, "generation": "synthetic_noise_box_muller_keyed_sha1_v1",
            "source_sha256": raw_hash, "array_sha256": array_hash,
            "sample_count": int(wave.size), "rms": _rms_dbfs(wave),
            "keys": {"purpose": "synthetic_noise", "pair_index_start": 0,
                     "pair_index_end_exclusive": n_samples // 2, "components": [0, 1]}}


def _response_curve(config, hp: float, lp: float) -> np.ndarray:
    centers = mel_centres(config).astype(np.float64)
    gain = np.ones_like(centers)
    if hp > 0:
        sos = signal.butter(2, hp, btype="highpass", fs=config.sr, output="sos")
        _, response = signal.sosfreqz(sos, worN=centers, fs=config.sr)
        gain *= np.abs(response) ** 2
    if lp > 0:
        sos = signal.butter(2, lp, btype="lowpass", fs=config.sr, output="sos")
        _, response = signal.sosfreqz(sos, worN=centers, fs=config.sr)
        gain *= np.abs(response) ** 2
    return gain.astype(np.float32)


def choose_mix_donor(train_rows: Iterable[dict[str, Any]], *, base_row: dict[str, Any],
                     epoch: int, draw_index: int) -> dict[str, Any] | None:
    """Select a train negative uniformly by eligible file group, then by clip."""
    base = dict(base_row)
    allowed_strata = {"N1", "N5", "N6", "N7"} if int(base.get("label", -1)) == 1 else None
    candidates = [dict(row) for row in train_rows if row.get("split") == "train"
                  and row.get("kept") is True and int(row.get("label", -1)) == 0
                  and row.get("negative_stratum") is not None
                  and (allowed_strata is None or row.get("negative_stratum") in allowed_strata)
                  and str(row.get("group_key")) != str(base.get("group_key"))]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        groups.setdefault(str(row["group_key"]), []).append(row)
    group_ids = sorted(groups)
    if not group_ids:
        return None
    key = _aug_key(int(epoch), int(draw_index), str(base["clip_id"]), "donor")
    group_id = group_ids[integer("aug_donor_group", key, len(group_ids))]
    clips = sorted(groups[group_id], key=lambda row: str(row["clip_id"]))
    return clips[integer("aug_donor_clip", key, len(clips))]


def crop_lookup_value(lookup_rows: Iterable[dict[str, Any]], *, clip_id: str,
                      crop_start_frame: int, manifest_sha256: str,
                      sample_sha256: str) -> dict[str, Any]:
    """Retrieve a crop scalar only when all cache/manifest identities agree."""
    matches = [dict(row) for row in lookup_rows if str(row["clip_id"]) == str(clip_id)
               and int(row["crop_start_frame"]) == int(crop_start_frame)]
    if len(matches) != 1:
        raise ValueError("crop lookup must have exactly one matching clip/frame row")
    row = matches[0]
    if row["manifest_sha256"] != manifest_sha256 or row["sample_sha256"] != sample_sha256:
        raise ValueError("crop lookup manifest/sample identity mismatch")
    return row


def a_check_mix_approximation(train_rows: Iterable[dict[str, Any]], waveforms: dict[str, np.ndarray],
                              *, config, pair_count: int = 200) -> dict[str, Any]:
    """Run the fixed deterministic 200-pair 0 dB waveform/power mix audit."""
    rows = [dict(row) for row in train_rows if row.get("split") == "train" and row.get("kept") is True]
    by_id = {str(row["clip_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("A-chk train clip IDs must be unique")
    if pair_count != 200:
        raise ValueError("A-chk is fixed at exactly 200 pairs")
    selected = _a_check_pairs(rows)
    if len(selected) != pair_count:
        return {"state": "FAIL", "pair_count": len(selected), "required_pairs": pair_count,
                "reason": "fewer than 200 eligible unique-group train crop pairs", "acceptance_max_mean_db": 3.0}
    selected_ids = {str(row["clip_id"]) for pair in selected for row in pair}
    if not selected_ids.issubset(waveforms):
        raise ValueError("A-chk is missing waveform data for its fixed selected pairs")
    from dronedet.features import fix_length, mel_power, to_db
    errors: list[float] = []
    pair_means: list[float] = []
    for pair_index, (base, donor) in enumerate(selected):
        b = np.asarray(waveforms[str(base["clip_id"])], dtype=np.float32)
        d = np.asarray(waveforms[str(donor["clip_id"])], dtype=np.float32)
        b, _ = fix_length(b, config)
        d, _ = fix_length(d, config)
        if not b.size or not d.size or b.size < config.win_samples or d.size < config.win_samples:
            raise ValueError("A-chk pair contains an ineligible frontend-length clip")
        bframes = 1 + (b.size - config.n_fft) // config.hop
        dframes = 1 + (d.size - config.n_fft) // config.hop
        key_b = _aug_key(0, pair_index, str(base["clip_id"]), "start")
        key_d = _aug_key(0, pair_index, str(donor["clip_id"]), "start")
        bstart = integer("crop_base", key_b, bframes - config.n_win + 1) * config.hop
        dstart = integer("crop_donor", key_d, dframes - config.n_win + 1) * config.hop
        bclip = b[bstart:bstart + config.win_samples]
        dclip = d[dstart:dstart + config.win_samples]
        bp, dp = _band_power(bclip, config.sr), _band_power(dclip, config.sr)
        if bp <= 0 or dp <= 0:
            return {"state": "FAIL", "pair_count": pair_count, "checked_pairs": pair_index,
                    "failed_pair": [str(base["clip_id"]), str(donor["clip_id"])],
                    "reason": "fixed selected A-chk crop has zero band power; pair list was not replaced",
                    "acceptance_max_mean_db": 3.0}
        donor_gain = uniform_range("aug_bed_gain", _aug_key(0, pair_index, str(base["clip_id"]), "db"), -30., 0.)
        dgain_amp = 10 ** (donor_gain / 20.)
        # Nominal 0 dB band SNR: scale the base to the gained donor's measured band power.
        base_scale = math.sqrt(dp * dgain_amp * dgain_amp / bp)
        waveform_mix = bclip * np.float32(base_scale) + dclip * np.float32(dgain_amp)
        base_power, donor_power = mel_power(bclip, config), mel_power(dclip, config)
        power_mix = base_power * np.float32(base_scale * base_scale) + donor_power * np.float32(dgain_amp * dgain_amp)
        db_wave = to_db(mel_power(waveform_mix, config), config).astype(np.float64)
        db_power = to_db(power_mix, config).astype(np.float64)
        pair_error = np.abs(db_wave - db_power)
        pair_means.append(float(pair_error.mean()))
        errors.extend(pair_error.reshape(-1).tolist())
    mean_error = float(np.mean(errors, dtype=np.float64))
    return {"state": "PASS" if mean_error <= 3.0 else "FAIL", "pair_count": pair_count,
            "mean_absolute_db_error": mean_error,
            "per_pair_error_quantiles_db": np.quantile(pair_means, [0., .25, .5, .75, .95, 1.], method="linear").tolist(),
            "pair_mean_error_db": pair_means, "acceptance_max_mean_db": 3.0}


def write_p5_artifacts(output_directory: str | Path, result: dict[str, Any], *, config) -> dict[str, Any]:
    """Persist unique local caches and bound P5 metadata after the caller's gates."""
    target = Path(output_directory)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite P5 attempt directory: {target}")
    target.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for split in ("train", "val"):
        power, index = result["caches"][split]
        cache_path, index_path = target / f"power_{split}.npy", target / f"power_{split}_index.json"
        np.save(cache_path, np.asarray(power, dtype=np.float32), allow_pickle=False)
        index_bytes = canonical_json_bytes(index) + b"\n"
        index_path.write_bytes(index_bytes)
        hashes[cache_path.name] = _hash_file(cache_path)
        hashes[index_path.name] = hashlib.sha256(index_bytes).hexdigest()
    lookup_hash = write_csv(target / "train_window_band_power.csv", contract_columns("train_window_band_power.csv"), result["band_lookup"])
    hashes["train_window_band_power.csv"] = lookup_hash
    norm_path = target / "normalizer.json"
    norm_bytes = canonical_json_bytes(result["normalizer"].state_dict()) + b"\n"
    norm_path.write_bytes(norm_bytes)
    hashes[norm_path.name] = hashlib.sha256(norm_bytes).hexdigest()
    template = result.get("noise_template") or generate_noise_template(config)
    template_path = target / "augmentation_noise_template.npz"
    np.savez_compressed(template_path, power=np.asarray(template["template"], dtype=np.float32),
                        generation=np.asarray(template["generation"]),
                        source_sha256=np.asarray(template["source_sha256"]),
                        array_sha256=np.asarray(template["array_sha256"]))
    hashes[template_path.name] = _hash_file(template_path)
    summary = {"study_id": "study-v5-1", "contract_sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
               **result["bindings"],
               "train_rows": len(result["train_rows"]), "val_rows": len(result["val_rows"]),
               "train_crop_rows": len(result["band_lookup"]), "normalizer": result["normalizer"].fit_record,
               "noise_template": {k: v for k, v in template.items() if k != "template"},
               "A_chk": result.get("a_check", {"state": "NOT_RUN_NO_AUTHORIZED_FROZEN_TRAIN"}),
               "B6": result.get("b6", {"state": "NOT_RUN_NO_P4_WEIGHTS"}), "artifact_sha256": hashes}
    summary_bytes = canonical_json_bytes(summary) + b"\n"
    (target / "p5_summary.json").write_bytes(summary_bytes)
    hashes["p5_summary.json"] = hashlib.sha256(summary_bytes).hexdigest()
    gate_states = {str(summary["A_chk"].get("state", "NOT_RUN")), str(summary["B6"].get("state", "NOT_RUN"))}
    state = "PASS" if gate_states == {"PASS"} else "FAIL" if "FAIL" in gate_states else "NOT_RUN"
    return {"state": state,
            "artifact_sha256": hashes, "summary": summary}


def augment_power(power: np.ndarray, *, row: dict[str, Any], epoch: int, draw_index: int,
                  config, training: bool, donor_power: np.ndarray | None = None,
                  donor_row: dict[str, Any] | None = None,
                  base_crop_frame: int | None = None, donor_crop_frame: int | None = None,
                  base_band_power: float | None = None, donor_band_power: float | None = None,
                  noise_template: np.ndarray | None = None) -> dict[str, Any]:
    """Apply A5/A2/A1/A3/A4 to a cached power crop; eval is strictly identity."""
    p = np.asarray(power, dtype=np.float32)
    if p.ndim != 2 or p.shape[0] != config.n_mels or p.shape[1] < config.n_win or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("augmentation requires finite nonnegative cached mel power")
    if not training:
        start = (p.shape[1] - config.n_win) // 2
        return {"power": p[:, start:start + config.n_win].copy(), "record": {"augmentation": "none", "evaluation": True,
                "crop_base": start}}
    epoch, draw_index, clip_id = int(epoch), int(draw_index), str(row["clip_id"])
    maximum = p.shape[1] - config.n_win
    crop = integer("crop_base", _aug_key(epoch, draw_index, clip_id, "start"), maximum + 1) if base_crop_frame is None else int(base_crop_frame)
    if not 0 <= crop <= maximum:
        raise ValueError("base crop offset is outside the cached clip")
    x = p[:, crop:crop + config.n_win].copy()
    label = int(row["label"])
    mix_p = .6 if label == 1 else .4
    mixed, mix_reason, donor_id = False, "probability_not_selected", None
    do_mix = uniform("aug_mix_flag", _aug_key(epoch, draw_index, clip_id, "flag")) < mix_p
    if do_mix:
        if donor_power is None or donor_band_power is None or base_band_power is None:
            mix_reason = "no_eligible_donor_or_bound_band_power"
        elif base_band_power <= 0 or donor_band_power <= 0:
            mix_reason = "zero_base_or_donor_band_power"
        else:
            if donor_row is not None:
                eligible_strata = {"N1", "N5", "N6", "N7"} if label == 1 else None
                if (donor_row.get("split") != "train" or int(donor_row.get("label", -1)) != 0
                        or donor_row.get("negative_stratum") is None
                        or (eligible_strata is not None and donor_row.get("negative_stratum") not in eligible_strata)
                        or str(donor_row.get("group_key")) == str(row.get("group_key"))):
                    raise ValueError("mix donor must be an eligible different-group train negative")
            donor = np.asarray(donor_power, dtype=np.float32)
            if donor.ndim != 2 or donor.shape[0] != config.n_mels or donor.shape[1] < config.n_win:
                raise ValueError("donor cached power is invalid")
            donor_max = donor.shape[1] - config.n_win
            dcrop = integer("crop_donor", _aug_key(epoch, draw_index, clip_id, "start"), donor_max + 1) if donor_crop_frame is None else int(donor_crop_frame)
            if not 0 <= dcrop <= donor_max:
                raise ValueError("donor crop offset or cached power is invalid")
            if not all(math.isfinite(float(value)) and float(value) > 0 for value in (base_band_power, donor_band_power)):
                raise ValueError("base and donor crop band powers must be finite and positive")
            d = donor[:, dcrop:dcrop + config.n_win]
            dgain = uniform_range("aug_bed_gain", _aug_key(epoch, draw_index, clip_id, "db"), -30., 0.)
            snr = uniform_range("aug_snr", _aug_key(epoch, draw_index, clip_id, "db"), -12. if label == 1 else -5., 20.)
            gained_donor = d * np.float32(10 ** (dgain / 10.))
            target_donor_band = float(donor_band_power) * 10 ** (dgain / 10.)
            base_scale = target_donor_band * 10 ** (snr / 10.) / float(base_band_power)
            x = x * np.float32(base_scale) + gained_donor
            mixed, mix_reason, donor_id = True, "mixed", str((donor_row or {}).get("clip_id", row.get("donor_clip_id", "selected")))
    gain_db = uniform_range("aug_gain", _aug_key(epoch, draw_index, clip_id, "db"), -36., 6.)
    x *= np.float32(10 ** (gain_db / 10.))
    response = uniform("aug_response_flag", _aug_key(epoch, draw_index, clip_id, "flag")) < .5
    hp, lp, tilt, ripple = 0., 0., 0., np.zeros(3, dtype=np.float64)
    if response:
        hp = uniform_range("aug_hp", _aug_key(epoch, draw_index, clip_id, "hz"), 0., 300.)
        if uniform("aug_lp_flag", _aug_key(epoch, draw_index, clip_id, "flag")) < .3:
            lp = uniform_range("aug_lp", _aug_key(epoch, draw_index, clip_id, "hz"), 4000., 7500.)
        tilt = uniform_range("aug_tilt", _aug_key(epoch, draw_index, clip_id, "span_db"), -6., 6.)
        ripple = np.asarray([uniform_range("aug_ripple", _aug_key(epoch, draw_index, clip_id, "db", i), -2., 2.) for i in range(3)])
        tilt_curve = np.linspace(-tilt / 2., tilt / 2., config.n_mels)
        ripple_curve = np.interp(np.arange(config.n_mels), [0., 31.5, 63.], ripple)
        response_curve = _response_curve(config, hp, lp)
        scale = response_curve * np.power(10., (tilt_curve + ripple_curve) / 10.)
        x *= scale.astype(np.float32)[:, None]
    floor_added, floor_db = False, None
    if noise_template is not None and uniform("aug_floor_flag", _aug_key(epoch, draw_index, clip_id, "flag")) < .5:
        template = np.asarray(noise_template, dtype=np.float32)
        if template.shape != (config.n_mels,) or not np.isfinite(template).all() or np.any(template < 0):
            raise ValueError("noise template has invalid shape or values")
        floor_db = uniform_range("aug_floor_rms", _aug_key(epoch, draw_index, clip_id, "dbfs"), -90., -55.)
        x += template[:, None] * np.float32(10 ** (floor_db / 10.))
        floor_added = True
    return {"power": x.astype(np.float32), "record": {"augmentation": "A5_A2_A1_A3_A4", "evaluation": False,
        "crop_base": crop, "mix": mixed, "mix_reason": mix_reason, "donor_clip_id": donor_id,
        "gain_db": gain_db, "response": response, "hp_hz": hp, "lp_hz": lp,
        "tilt_span_db": tilt, "ripple_db": ripple.tolist(), "floor_added": floor_added, "floor_rms_dbfs": floor_db}}


def model_input(power: np.ndarray, normalizer: Normalizer, config, *, row: dict[str, Any],
                training: bool, epoch: int = 0, draw_index: int = 0, **augmentation) -> dict[str, Any]:
    """Run augmentation, frontend transform and train-fitted normalization."""
    from dronedet.features import transform
    augmented = augment_power(power, row=row, epoch=epoch, draw_index=draw_index,
                              config=config, training=training, **augmentation)
    tensor = transform(augmented["power"], config, stats=normalizer.state_dict())
    return {"input": tensor, "power": augmented["power"], "record": augmented["record"]}


def b6_level_gate(train_rows: Iterable[dict[str, Any]], crop_lookup: Iterable[dict[str, Any]],
                  weights: dict[str, float], *, draws: int = 100_000) -> dict[str, Any]:
    """Evaluate fixed epoch-zero weighted draws for A1-only full-window RMS coverage."""
    rows = sorted((dict(r) for r in train_rows), key=lambda r: str(r["clip_id"]))
    if not rows or draws != 100_000:
        raise ValueError("B6 requires train rows and exactly 100,000 draws")
    if set(weights) != {str(r["clip_id"]) for r in rows}:
        raise ValueError("B6 weights must cover exactly the frozen train IDs")
    crop_by_clip: dict[str, list[dict[str, Any]]] = {}
    for item in crop_lookup:
        crop_by_clip.setdefault(str(item["clip_id"]), []).append(dict(item))
    for items in crop_by_clip.values():
        items.sort(key=lambda item: int(item["crop_start_frame"]))
    if set(crop_by_clip) != set(weights):
        raise ValueError("B6 crop lookup must cover exactly the frozen train IDs")
    ids = [str(row["clip_id"]) for row in rows]
    probabilities = np.asarray([float(weights[clip_id]) for clip_id in ids], dtype=np.float64)
    if not np.isfinite(probabilities).all() or np.any(probabilities <= 0):
        raise ValueError("B6 weights must be finite and positive")
    cumulative = np.cumsum(probabilities / probabilities.sum())
    classes = {0: np.zeros(6, dtype=np.int64), 1: np.zeros(6, dtype=np.int64)}
    out_of_range = {0: 0, 1: 0}
    for draw in range(draws):
        # Sampling follows the fitted train weights with a deterministic contract key.
        selected = rows[min(int(np.searchsorted(cumulative, uniform("sampler", [0, draw]), side="right")), len(rows) - 1)]
        clip_id = str(selected["clip_id"])
        crops = crop_by_clip[clip_id]
        crop_index = integer("crop_base", _aug_key(0, draw, clip_id, "start"), len(crops))
        label, rms = int(selected["label"]), float(crops[crop_index]["rms_window_full_dbfs"])
        gain = uniform_range("aug_gain", _aug_key(0, draw, str(selected["clip_id"]), "db"), -36., 6.)
        level = rms + gain
        edges = [-70, -60, -50, -40, -30, -20, -10]
        bin_index = int(np.searchsorted(edges, level, side="right") - 1)
        if 0 <= bin_index < 6:
            classes[label][bin_index] += 1
        else:
            out_of_range[label] += 1
    total_by_class = {label: int(classes[label].sum() + out_of_range[label]) for label in classes}
    shares = {label: (classes[label] / max(total_by_class[label], 1)).tolist() for label in classes}
    passes = all(float(value) >= .05 for label in shares for value in shares[label])
    return {"state": "PASS" if passes else "FAIL", "draws": draws, "bin_edges_dbfs": [-70, -60, -50, -40, -30, -20, -10],
            "per_class_bin_share": shares, "out_of_range_count": out_of_range, "acceptance_min_share": .05}
