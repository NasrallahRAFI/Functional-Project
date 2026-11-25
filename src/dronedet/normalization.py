"""Enforced train-only statistics and librosa PCEN implementation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from scipy import signal

from .config import FrontEndConfig
from .features import to_db

GLOBAL_STATS_POLICY = "train_stats_eligible_per_band_floor_mask_v1"


def pcen_transform(power: np.ndarray, config: FrontEndConfig) -> np.ndarray:
    x = np.asarray(power, dtype=np.float32)
    if x.ndim != 2 or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("PCEN expects finite nonnegative (mel, frames) power")
    magnitude = np.sqrt(x) * np.float32(config.pcen_input_scale)
    time_frames = config.pcen_time_constant * config.sr / config.hop
    b = float((np.sqrt(1.0 + 4.0 * time_frames**2) - 1.0) / (2.0 * time_frames**2))
    # librosa's lfilter state has the input dimensions with a singleton along time.
    zi = ((1.0 - b) * magnitude.mean(axis=-1, keepdims=True)).astype(np.float32)
    result = librosa.pcen(magnitude, sr=config.sr, hop_length=config.hop,
        time_constant=config.pcen_time_constant, gain=config.pcen_gain,
        bias=config.pcen_bias, power=config.pcen_power, eps=config.pcen_eps,
        b=b, zi=zi, axis=-1)
    return np.asarray(result, dtype=np.float32)


class Normalizer:
    """Fit dB or PCEN statistics from explicit train-manifest rows only."""
    def __init__(self, mode: str, mean: np.ndarray | float, std: np.ndarray | float,
                 excluded_count: int = 0, *, fit_record: dict[str, Any] | None = None) -> None:
        self.mode = mode
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.excluded_count = int(excluded_count)
        self.fit_record = dict(fit_record or {})
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.std)) or np.any(self.std <= 0):
            raise ValueError("normalization stats must be finite and std positive")

    @classmethod
    def fit(cls, manifest_rows, config: FrontEndConfig, power_by_file: dict[str, np.ndarray]) -> "Normalizer":
        rows = list(manifest_rows)
        if any(str(row.get("split", "")).lower() != "train" for row in rows):
            raise ValueError("Normalizer.fit accepts train rows only")
        if config.mode == "logstft":
            raise NotImplementedError("logstft requires a separate 513-bin cache")
        values, excluded, names = [], 0, set()
        from .features import stats_exclude
        for row in rows:
            name = str(row.get("file", row.get("clip_id", "")))
            if not name or name in names:
                raise ValueError("manifest rows require unique file or clip_id values")
            names.add(name)
            eligible = row.get("stats_eligible")
            if not isinstance(eligible, (bool, np.bool_)):
                raise ValueError("Normalizer.fit requires stored boolean stats_eligible values")
            flag_names = ("silent", "near_silent", "crop_all_zero")
            if all(key in row for key in flag_names):
                flags = [row[key] for key in flag_names]
                if any(not isinstance(flag, (bool, np.bool_)) for flag in flags):
                    raise ValueError("normalization eligibility flags must be boolean")
                if bool(eligible) == stats_exclude(*flags):
                    raise ValueError(f"{name}: stats_eligible contradicts exclusion flags")
            if not eligible:
                excluded += 1
                continue
            p = np.asarray(power_by_file[name], dtype=np.float32)
            if p.ndim != 2 or p.shape[0] != config.n_mels or p.shape[1] < config.n_win:
                raise ValueError(f"{name}: normalization power has invalid dimensions")
            if not np.isfinite(p).all() or np.any(p < 0):
                raise ValueError(f"{name}: normalization power must be finite and nonnegative")
            # Full-clip dB moments use every frame; PCEN/per-window modes use the demo-sized crop.
            if config.mode in {"pcen", "per_window_mean"}:
                from .features import centre_window
                p = centre_window(p, config)
            if config.mode == "pcen":
                v = pcen_transform(p, config)
            else:
                v = to_db(p, config)
            if config.mode == "per_window_mean":
                v = v - v.mean()
            values.append(v.astype(np.float64))
        if not values: raise ValueError("no eligible training clips for normalization")
        record = {"eligible_clips": len(values), "excluded_clips": excluded,
                  "source_split": "train", "mode": config.mode}
        if config.mode == "global_stats":
            joined = np.concatenate(values, axis=1)
            # Floor-clamped dB cells carry the mel-normalization-dependent
            # amin offset into otherwise per-band-invariant statistics. Keep
            # the train-clip eligibility rule, and additionally omit cells at
            # the configured power_to_db floor from each band's moments.
            floor_db = 10.0 * float(np.log10(config.amin))
            floor_epsilon_db = 1e-6
            mean = np.empty(config.n_mels, dtype=np.float64)
            std = np.empty(config.n_mels, dtype=np.float64)
            counts = np.empty(config.n_mels, dtype=np.int64)
            for band in range(config.n_mels):
                cells = joined[band, joined[band] > floor_db + floor_epsilon_db]
                if cells.size == 0:
                    raise ValueError(f"mel band {band} has no non-floor training cells")
                mean[band] = cells.mean(dtype=np.float64)
                std[band] = cells.std(dtype=np.float64)
                counts[band] = cells.size
            record.update(policy=GLOBAL_STATS_POLICY, floor_threshold_db=floor_db + floor_epsilon_db,
                          frame_count=joined.shape[1], retained_cells_per_band=counts.tolist(),
                          masked_cells_per_band=(joined.shape[1] - counts).tolist())
        else:
            joined = np.concatenate([v.ravel() for v in values])
            mean, std = float(joined.mean()), float(joined.std())
            record.update(policy=f"train_centered_{config.mode}_v1", cells=int(joined.size))
        return cls(config.mode, mean, np.maximum(std, 1e-6), excluded, fit_record=record)

    def state_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "excluded_silent_near_silent": self.excluded_count,
                "fit_record": self.fit_record}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state_dict(), indent=2) + "\n", encoding="utf8")

    def save_float32(self, path: str | Path) -> None:
        """Persist typed float32 statistics for cache/bundle consumers."""
        np.savez_compressed(path, mode=np.asarray(self.mode), mean=self.mean.astype(np.float32),
                            std=self.std.astype(np.float32),
                            excluded_silent_near_silent=np.asarray(self.excluded_count, dtype=np.int64))


def apply_gain(power: np.ndarray, rng: np.random.Generator, config: FrontEndConfig, *, training: bool) -> np.ndarray:
    if not training:
        return np.asarray(power, dtype=np.float32)
    lo, hi = config.gain_range_db
    gain = float(rng.uniform(lo, hi))
    return (np.asarray(power, dtype=np.float32) * np.float32(10 ** (gain / 10))).astype(np.float32)


def apply_fixed_gain(power: np.ndarray, gain_db: float) -> np.ndarray:
    return (np.asarray(power, dtype=np.float32) * np.float32(10 ** (gain_db / 10))).astype(np.float32)


def apply_mel_eq(S_power: np.ndarray, rng: np.random.Generator, *,
                 max_db: float, n_control_points: int) -> np.ndarray:
    """Apply a seeded, piecewise-linear mel equalization curve in power space."""
    power = np.asarray(S_power, dtype=np.float32)
    if power.ndim != 2 or not np.all(np.isfinite(power)) or np.any(power < 0):
        raise ValueError("mel EQ expects finite nonnegative (mel, frames) power")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if not np.isfinite(max_db) or max_db < 0:
        raise ValueError("max_db must be finite and nonnegative")
    if not isinstance(n_control_points, (int, np.integer)) or not 1 <= n_control_points <= power.shape[0]:
        raise ValueError("n_control_points must be between 1 and the number of mel bands")
    control_db = rng.uniform(-max_db, max_db, size=int(n_control_points))
    control_x = np.linspace(0.0, power.shape[0] - 1.0, int(n_control_points))
    curve_db = np.interp(np.arange(power.shape[0], dtype=np.float64), control_x, control_db)
    scale = np.power(10.0, curve_db / 10.0).astype(np.float32)
    return (power * scale[:, None]).astype(np.float32)


def apply_mel_eq_training(S_power: np.ndarray, rng: np.random.Generator, *,
                          training: bool, max_db: float,
                          n_control_points: int) -> np.ndarray:
    """Training-only augmentation guard around the pure mel EQ function."""
    if not training:
        raise ValueError("mel EQ augmentation is available only in training mode")
    return apply_mel_eq(S_power, rng, max_db=max_db,
                        n_control_points=n_control_points)


def pcen_gain_check(train_windows: list[np.ndarray], config: FrontEndConfig) -> dict[str, float | str]:
    """Measure gain robustness on provided train windows; no train data means deferred."""
    if not train_windows: return {"status": "deferred_no_frozen_train_manifest"}
    changes=[]
    for p in train_windows:
        base=pcen_transform(p,config)
        std=max(float(base.std()),1e-12)
        for gain in (-12.0,12.0):
            changed=pcen_transform(apply_fixed_gain(p,gain),config)
            changes.append(float(np.mean(np.abs(changed-base))/std))
    largest=max(changes)
    return {"status": "pass" if largest <= .2 else "fail", "max_change_in_std": largest,
            "acceptance_limit": .2, "window_count": len(train_windows)}


def stress_filter(audio: np.ndarray, sample_rate: int, kind: str) -> np.ndarray:
    specs = {"highpass_150": ("highpass", 150), "highpass_300": ("highpass", 300), "lowpass_5000": ("lowpass", 5000)}
    if kind not in specs: raise ValueError("unknown stress filter")
    typ, cutoff = specs[kind]
    sos = signal.butter(2, cutoff, btype=typ, fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(audio, dtype=np.float32)).astype(np.float32)


def mel_norm_equivalence_check(manifest_rows, raw_root: str | Path, config: FrontEndConfig, *,
                               offset_stride: int = 1, floor_db: float = -120.0,
                               floor_epsilon_db: float = 1e-6,
                               min_nonfloor_cells_per_band: int = 10_000,
                               residual_threshold_std: float = 0.05) -> dict[str, Any]:
    """Diagnose area/peak mel equivalence using train audio only.

    The function keeps both mel-power arms in memory and fits each arm's own
    statistics from the stored train ``stats_eligible`` rows, masking floor
    cells independently in each mel band. G2 equivalence is evaluated only on
    windows with no floor cell in either arm; floor-containing windows are
    counted and reported. Set ``offset_stride=4`` only for the deterministic
    fallback.
    """
    import soundfile as sf
    from .features import file_to_power, to_db

    if offset_stride not in {1, 4}:
        raise ValueError("offset_stride must be 1 (full) or 4 (documented fallback)")
    rows = [dict(row) if isinstance(row, dict) else row._asdict()
            if hasattr(row, "_asdict") else dict(row) for row in manifest_rows]
    if not rows:
        raise ValueError("mel_norm equivalence requires train manifest rows")
    if any("split" in row and str(row["split"]).lower() != "train" for row in rows):
        raise ValueError("mel_norm equivalence accepts train rows only")

    peak_config = __import__("dataclasses").replace(config, mel_norm="peak")
    root = Path(raw_root)
    area_powers: list[np.ndarray] = []
    peak_powers: list[np.ndarray] = []
    eligible: list[bool] = []
    band_edges = librosa.mel_frequencies(n_mels=config.n_mels + 2,
        fmin=config.fmin, fmax=config.fmax, htk=config.mel_scale == "htk")
    band_scale = (2.0 / (band_edges[2:] - band_edges[:-2])).astype(np.float64)

    # D1: verify the library filterbank identity numerically.
    w_area = librosa.filters.mel(sr=config.sr, n_fft=config.n_fft,
        n_mels=config.n_mels, fmin=config.fmin, fmax=config.fmax,
        htk=config.mel_scale == "htk", norm="slaney", dtype=np.float32)
    w_peak = librosa.filters.mel(sr=config.sr, n_fft=config.n_fft,
        n_mels=config.n_mels, fmin=config.fmin, fmax=config.fmax,
        htk=config.mel_scale == "htk", norm=None, dtype=np.float32)
    expected_area = band_scale.astype(np.float32)[:, None] * w_peak
    support = np.array_equal(w_area > 0, w_peak > 0)
    nz = w_peak != 0
    relation_rel = np.abs(w_area[nz] - expected_area[nz]) / np.maximum(
        np.abs(expected_area[nz]), np.float32(1e-30))

    # Compute both real train arms, retaining only in-memory powers.
    for row in rows:
        clip = str(row.get("clip_id", row.get("file", "")))
        if not clip:
            raise ValueError("train manifest rows require clip_id or file")
        area, _ = file_to_power(root / clip, config)
        peak, _ = file_to_power(root / clip, peak_config)
        area_powers.append(area)
        peak_powers.append(peak)
        if "stats_eligible" not in row:
            raise ValueError("mel_norm equivalence requires stored train stats_eligible values")
        eligible.append(bool(row["stats_eligible"]))
    eligible_arr = np.asarray(eligible, dtype=bool)

    def fit_stats(powers: list[np.ndarray], *, exclude_all_floor_frames: bool
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sums = np.zeros(config.n_mels, dtype=np.float64)
        sums2 = np.zeros(config.n_mels, dtype=np.float64)
        counts = np.zeros(config.n_mels, dtype=np.int64)
        for power, use in zip(powers, eligible_arr):
            if not use:
                continue
            values = to_db(power, config).astype(np.float64)
            floor_cells = values <= floor_db + floor_epsilon_db
            mask = ~floor_cells
            if exclude_all_floor_frames:
                # Counterfactual OD1 comparison: after the cell-level fit
                # mask, dropping all-floor frames is redundant by definition.
                keep_frames = ~np.all(floor_cells, axis=0)
                mask &= np.broadcast_to(keep_frames[None, :], values.shape)
            sums += np.where(mask, values, 0.0).sum(axis=1)
            sums2 += np.where(mask, values * values, 0.0).sum(axis=1)
            counts += mask.sum(axis=1)
        if np.any(counts == 0):
            raise ValueError("a mel band has no statistics cells")
        mean = sums / counts
        variance = np.maximum(sums2 / counts - mean * mean, 0.0)
        return mean, np.maximum(np.sqrt(variance), 1e-6), counts

    base = {"area": fit_stats(area_powers, exclude_all_floor_frames=False),
            "peak": fit_stats(peak_powers, exclude_all_floor_frames=False)}
    # Retain a whole-frame counterfactual for continuity with the earlier
    # OD1 report. Under the approved cell-level floor mask, removing an
    # all-floor frame is redundant because all its cells are already omitted.
    no_all_floor = {"area": fit_stats(area_powers, exclude_all_floor_frames=True),
                    "peak": fit_stats(peak_powers, exclude_all_floor_frames=True)}
    arms = {"area": area_powers, "peak": peak_powers}
    floor_stats = {arm: np.zeros(config.n_mels, dtype=np.int64) for arm in arms}
    total_stats = {arm: np.zeros(config.n_mels, dtype=np.int64) for arm in arms}
    for arm, powers in arms.items():
        for power, use in zip(powers, eligible_arr):
            if use:
                values = to_db(power, config)
                floor_stats[arm] += (values <= floor_db + floor_epsilon_db).sum(axis=1)
                total_stats[arm] += values.shape[1]
    nonfloor_stats = {arm: total_stats[arm] - floor_stats[arm] for arm in arms}
    if any(np.any(counts < min_nonfloor_cells_per_band) for counts in nonfloor_stats.values()):
        return {"status": "stop_insufficient_nonfloor_statistics",
                "d1": {"support_identical": support,
                       "max_relative_deviation": float(relation_rel.max(initial=0.0))},
                "d3": {arm: {"min_nonfloor_cells_per_band": int(counts.min()),
                             "counts_per_band": counts.tolist()}
                       for arm, counts in nonfloor_stats.items()},
                "stop_threshold": int(min_nonfloor_cells_per_band)}

    # Compute all eligible frame offsets, or every fourth offset in fallback.
    offsets_by_clip: list[list[int]] = []
    for power in area_powers:
        n_frames = power.shape[1]
        last = n_frames - config.n_win
        offsets = list(range(0, last + 1, offset_stride)) if last >= 0 else []
        if offsets and offsets[-1] != last:
            offsets.append(last)
        offsets_by_clip.append(offsets)

    # The legacy OD1 Delta is measured on centered train windows. With the
    # cell-level floor mask, the whole-frame counterfactual should be a no-op.
    od1_delta_max = {arm: 0.0 for arm in arms}
    od1_delta_cells = {arm: 0 for arm in arms}
    centered_partial_frames = {arm: 0 for arm in arms}
    centered_all_floor_frames = {arm: 0 for arm in arms}
    centered_windows_with_floor = {arm: 0 for arm in arms}
    centered_window_count = 0
    final_stats: dict[str, tuple[np.ndarray, np.ndarray]] = base
    final_stats32: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for arm in arms:
        mean, std, _ = base[arm]
        final_stats32[arm] = (mean.astype(np.float32).astype(np.float64),
                               std.astype(np.float32).astype(np.float64))
    for i, (a_power, p_power) in enumerate(zip(area_powers, peak_powers)):
        if a_power.shape[1] < config.n_win or p_power.shape[1] < config.n_win:
            continue
        start = (a_power.shape[1] - config.n_win) // 2
        centered_window_count += 1
        for arm, power in (("area", a_power), ("peak", p_power)):
            values = to_db(power, config)
            window = values[:, start:start + config.n_win]
            floor_cells = window <= floor_db + floor_epsilon_db
            all_floor = np.all(floor_cells, axis=0)
            partial_floor = np.any(floor_cells, axis=0) & ~all_floor
            centered_partial_frames[arm] += int(partial_floor.sum())
            centered_all_floor_frames[arm] += int(all_floor.sum())
            centered_windows_with_floor[arm] += int(floor_cells.any())
            valid = window > floor_db + 1.0
            mean0, std0, _ = base[arm]
            mean1, std1, _ = no_all_floor[arm]
            diff = np.abs((window - mean0[:, None]) / std0[:, None]
                          - (window - mean1[:, None]) / std1[:, None])
            if np.any(valid):
                od1_delta_max[arm] = max(od1_delta_max[arm],
                                         float(diff[valid].max(initial=0.0)))
                od1_delta_cells[arm] += int(valid.sum())

    # D3/D4 class/flag counts, residuals, precision yardstick, and top clips.
    stats_counts: dict[str, Any] = {}
    window_counts: dict[str, Any] = {}
    unique_window_frame_counts: dict[str, Any] = {}
    histogram_max_std = 5.0
    # A bin width equal to the acceptance threshold yields a conservative
    # upper bound that still decides p99.9 <= 0.05 exactly at the gate.
    histogram_bin_width_std = residual_threshold_std
    histogram_bins = int(histogram_max_std / histogram_bin_width_std)

    def new_residual_metric() -> dict[str, Any]:
        return {"histogram": np.zeros(histogram_bins, dtype=np.int64),
                "count": 0, "sum": 0.0, "max": 0.0, "above_histogram_range": 0}

    def add_residual(metric: dict[str, Any], values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size == 0:
            return
        metric["count"] += int(values.size)
        metric["sum"] += float(values.sum(dtype=np.float64))
        metric["max"] = max(metric["max"], float(values.max(initial=0.0)))
        metric["above_histogram_range"] += int(np.count_nonzero(values >= histogram_max_std))
        clipped = np.minimum(values, np.nextafter(histogram_max_std, 0.0))
        metric["histogram"] += np.histogram(
            clipped, bins=histogram_bins, range=(0.0, histogram_max_std))[0]

    def summarize_residual(metric: dict[str, Any]) -> dict[str, Any]:
        count = metric["count"]
        if not count:
            return {"p99_9_std_upper_bound": None, "mean_std": None,
                    "max_std": None, "valid_cells": 0,
                    "histogram_bin_width_std": histogram_bin_width_std,
                    "above_histogram_range": 0}
        rank = int(np.ceil(0.999 * count))
        cumulative = np.cumsum(metric["histogram"])
        bin_index = min(int(np.searchsorted(cumulative, rank, side="left")),
                        histogram_bins - 1)
        return {"p99_9_std_lower_bound": float(bin_index * histogram_bin_width_std),
                "p99_9_std_upper_bound": float((bin_index + 1) * histogram_bin_width_std),
                "mean_std": metric["sum"] / count, "max_std": metric["max"],
                "valid_cells": count,
                "histogram_bin_width_std": histogram_bin_width_std,
                "above_histogram_range": metric["above_histogram_range"]}

    residual_all_nonfloor = new_residual_metric()
    residual_floor_free_windows = new_residual_metric()
    total_evaluated_windows = 0
    floor_free_window_count = 0
    precision_max = {arm: 0.0 for arm in arms}
    clip_max: list[dict[str, Any]] = []
    for i, (row, a_power, p_power, offsets) in enumerate(
            zip(rows, area_powers, peak_powers, offsets_by_clip)):
        label = str(row.get("label_name", row.get("label", "unknown")))
        flags = {name: bool(row.get(name, row.get("all_zero", False) if name == "silent" else False))
                 for name in ("silent", "near_silent", "clipped")}
        for arm, power in (("area", a_power), ("peak", p_power)):
            if eligible_arr[i]:
                db = to_db(power, config)
                floored = db <= floor_db + floor_epsilon_db
                bucket = stats_counts.setdefault(arm, {})
                for group, key in [("class", label)] + [(name, f"{name}={value}") for name, value in flags.items()]:
                    item = bucket.setdefault(f"{group}:{key}", {"cells": 0, "floor_cells": 0})
                    item["cells"] += int(floored.size if group == "class" else floored.size)
                    item["floor_cells"] += int(floored.sum())
        unique_frames = sorted({frame for start in offsets
                                for frame in range(start, start + config.n_win)})
        if unique_frames:
            frame_idx = np.asarray(unique_frames, dtype=np.int64)
            for arm, power in (("area", a_power), ("peak", p_power)):
                db_frames = to_db(power[:, frame_idx], config)
                floor_mask = db_frames <= floor_db + floor_epsilon_db
                all_floor = np.all(floor_mask, axis=0)
                partial_floor = np.any(floor_mask, axis=0) & ~all_floor
                bucket = unique_window_frame_counts.setdefault(arm, {})
                for group, key in [("class", label)] + [(name, f"{name}={value}") for name, value in flags.items()]:
                    item = bucket.setdefault(f"{group}:{key}", {
                        "unique_frames": 0, "frames_with_floor": 0,
                        "all_floor_frames": 0, "partial_floor_frames": 0})
                    item["unique_frames"] += int(len(unique_frames))
                    item["frames_with_floor"] += int(np.any(floor_mask, axis=0).sum())
                    item["all_floor_frames"] += int(all_floor.sum())
                    item["partial_floor_frames"] += int(partial_floor.sum())
        max_resid = 0.0
        for start in offsets:
            total_evaluated_windows += 1
            a_db = to_db(a_power[:, start:start + config.n_win], config)
            p_db = to_db(p_power[:, start:start + config.n_win], config)
            # Residual applies only where both configurations are >1 dB above floor.
            valid = (a_db > floor_db + 1.0) & (p_db > floor_db + 1.0)
            am, ast, _ = final_stats["area"]
            pm, pst, _ = final_stats["peak"]
            az = (a_db - am[:, None]) / ast[:, None]
            pz = (p_db - pm[:, None]) / pst[:, None]
            residual = np.abs(az - pz)
            if np.any(valid):
                add_residual(residual_all_nonfloor, residual[valid])
                max_resid = max(max_resid, float(residual[valid].max(initial=0.0)))
            floor_free_window = not bool(np.any(a_db <= floor_db + floor_epsilon_db)
                                         or np.any(p_db <= floor_db + floor_epsilon_db))
            if floor_free_window:
                floor_free_window_count += 1
                add_residual(residual_floor_free_windows, residual)
            # D5 compares outputs under float32 and float64 versions of the
            # selected statistics for both arms on the same evaluated cells.
            for arm, values in (("area", a_db), ("peak", p_db)):
                mu64, sd64, _ = final_stats[arm]
                mu32, sd32 = final_stats32[arm]
                z64 = (values - mu64[:, None]) / sd64[:, None]
                z32 = (values - mu32[:, None]) / sd32[:, None]
                precision_max[arm] = max(precision_max[arm],
                    float(np.max(np.abs(z64 - z32), initial=0.0)))
            for arm, db in (("area", a_db), ("peak", p_db)):
                floor_mask = db <= floor_db + floor_epsilon_db
                all_floor = np.all(floor_mask, axis=0)
                partial_floor = np.any(floor_mask, axis=0) & ~all_floor
                bucket = window_counts.setdefault(arm, {})
                for group, key in [("class", label)] + [(name, f"{name}={value}") for name, value in flags.items()]:
                    item = bucket.setdefault(f"{group}:{key}",
                        {"windows": 0, "windows_with_floor": 0, "floor_cells": 0,
                         "cells": 0, "all_floor_frames": 0, "partial_floor_frames": 0})
                    item["windows"] += 1
                    item["windows_with_floor"] += int(floor_mask.any())
                    item["floor_cells"] += int(floor_mask.sum())
                    item["cells"] += int(floor_mask.size)
                    item["all_floor_frames"] += int(all_floor.sum())
                    item["partial_floor_frames"] += int(partial_floor.sum())
        clip_max.append({"clip_id": str(row.get("clip_id", row.get("file", ""))),
                         "max_residual_std": max_resid, "label": label, **flags,
                         "n_samples": int(row.get("n_samples", 0))})

    no_floor_summary = summarize_residual(residual_floor_free_windows)
    all_nonfloor_summary = summarize_residual(residual_all_nonfloor)
    clip_max.sort(key=lambda item: item["max_residual_std"], reverse=True)
    for item in clip_max[:20]:
        audio, _ = sf.read(str(root / item["clip_id"]), dtype="float32", always_2d=True)
        item["exact_zero_samples_channel0"] = int(np.count_nonzero(audio[:, config.channel] == 0.0))

    total_windows = sum(map(len, offsets_by_clip))
    od1_delta = max(od1_delta_max.values(), default=0.0)
    no_floor_p999_upper = no_floor_summary["p99_9_std_upper_bound"]
    status = ("pass" if no_floor_p999_upper is not None
              and no_floor_p999_upper <= residual_threshold_std
              else "stop_no_floor_residual_exceeds_threshold")
    return {
        "status": status,
        "d1": {"support_identical": support,
               "max_relative_deviation": float(relation_rel.max(initial=0.0)),
               "max_absolute_deviation": float(np.abs(w_area - expected_area).max(initial=0.0))},
        "d2": {"train_clips": len(rows), "eligible_clips": int(eligible_arr.sum()),
               "excluded_by_stats_eligible": int((~eligible_arr).sum()),
               "eligibility_source": "stored stats_eligible column",
               "mode": config.mode, "fmin": config.fmin, "fmax": config.fmax,
               "same_arm_own_statistics": True, "non_train_rows_touched": 0},
        "fitted_statistics": {arm: {"mean": base[arm][0].astype(np.float32).tolist(),
                                    "std": base[arm][1].astype(np.float32).tolist(),
                                    "cells_per_band": base[arm][2].tolist()}
                              for arm in arms},
        "d3": {"statistics_floor_counts_by_class_and_flag": {
                   arm: {key: {**value,
                               "floor_cell_fraction": (value["floor_cells"] / value["cells"]
                                                       if value["cells"] else 0.0)}
                         for key, value in counts.items()}
                   for arm, counts in stats_counts.items()},
               "min_nonfloor_cells_per_band": {arm: int(v.min()) for arm, v in nonfloor_stats.items()},
               "nonfloor_cells_per_band": {arm: v.tolist() for arm, v in nonfloor_stats.items()}},
        "d4": {"offset_stride": offset_stride, "fallback_every_fourth_offset": offset_stride == 4,
               "evaluated_windows": int(total_windows),
               "unique_frame_counts_by_class_and_flag": {
                   arm: {key: {**value,
                               "frames_with_floor_fraction": (value["frames_with_floor"] / value["unique_frames"]
                                                               if value["unique_frames"] else 0.0),
                               "all_floor_frame_fraction": (value["all_floor_frames"] / value["unique_frames"]
                                                             if value["unique_frames"] else 0.0),
                               "partial_floor_frame_fraction": (value["partial_floor_frames"] / value["unique_frames"]
                                                                if value["unique_frames"] else 0.0)}
                         for key, value in counts.items()}
                   for arm, counts in unique_window_frame_counts.items()},
               "window_floor_counts_by_class_and_flag": {
                   arm: {key: {**value,
                               "windows_with_floor_fraction": (value["windows_with_floor"] / value["windows"]
                                                                if value["windows"] else 0.0),
                               "floor_cell_fraction": (value["floor_cells"] / value["cells"]
                                                       if value["cells"] else 0.0)}
                         for key, value in counts.items()}
                   for arm, counts in window_counts.items()}},
        "d5": {"float32_vs_float64_max_abs_std": precision_max},
        "od1_whole_frame_exclusion": {
            "rule": "superseded: individual floor cells are already excluded from per-band fit",
            "centered_train_windows": centered_window_count,
            "centered_windows_with_any_floor": centered_windows_with_floor,
            "centered_windows_with_any_floor_fraction": {
                arm: (centered_windows_with_floor[arm] / centered_window_count
                      if centered_window_count else 0.0) for arm in arms},
            "centered_all_floor_frames": centered_all_floor_frames,
            "centered_partial_floor_frames": centered_partial_frames,
            "partial_floor_frames_are_cell_masked": True,
            "delta_max_std_by_arm": od1_delta_max,
            "delta_max_std": od1_delta,
            "delta_valid_cells_by_arm": od1_delta_cells,
            "whole_frame_counterfactual_is_redundant_after_cell_masking": True,
            "excluded_all_floor_statistics_frames_by_arm": {
                arm: int(sum(power.shape[1] - int(np.count_nonzero(
                    ~np.all(to_db(power, config) <= floor_db + floor_epsilon_db, axis=0)))
                    for power, use in zip(powers, eligible_arr) if use))
                for arm, powers in arms.items()},
        },
        "equivalence_residual": {
            "acceptance_subset": "windows with no floor cell in either mel_norm arm",
            "all_evaluated_windows": total_evaluated_windows,
            "floor_free_windows": floor_free_window_count,
            "floor_free_window_fraction": (floor_free_window_count / total_evaluated_windows
                                           if total_evaluated_windows else 0.0),
            "floor_containing_windows": total_evaluated_windows - floor_free_window_count,
            "threshold_std": residual_threshold_std,
            **no_floor_summary,
            "all_windows_nonfloor_cell_residual_secondary": all_nonfloor_summary,
        },
        "top20_clips": clip_max[:20],
        "floor_definition_db": floor_db + floor_epsilon_db,
    }
