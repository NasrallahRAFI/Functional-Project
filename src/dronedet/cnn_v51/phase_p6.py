"""Validation-only stream and event replay evidence for V5.1 preparation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from scipy import signal
from scipy.stats import chi2

from .artifacts import canonical_json_bytes
from .randomness import integer, ranked, uniform


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_replay_contract(output: str | Path, *, python_builder: str | Path,
                          smoother: str | Path, browser_client: str | Path) -> dict[str, Any]:
    """Bind the exact Python/browser replay logic and reset policy by SHA-256."""
    paths = {"LiveWindowBuilder": Path(python_builder), "VoteSmoother": Path(smoother),
             "browser_reset_rules": Path(browser_client)}
    hashes = {name: _sha256(path) for name, path in paths.items()}
    body = {
        "schema": "v5.1_replay_contract_v1", "source_sha256": hashes,
        "score_rule": {"fresh_scored_windows_before_decision": 3, "majority": "2_of_3",
                        "v4_baseline_threshold": None, "threshold_selection": "not_performed"},
        "vote_resets": ["standby", "buffering", "recovered_capture_or_generation_change",
                        "new_file_or_session", "stop", "real_gap"],
        "builder_resets": ["new_file_or_session", "stop", "real_gap"],
        "quiet_window_does_not_reset_builder": True,
        "replay": {"whole_files": True, "flush_finish_tail": True,
                   "timestamp": "standardized_source_sample_window_end",
                   "exposure_samples": 7936, "episode_merge_gap_seconds": 5.0,
                   "exclude_gap_and_recovered": True},
        "classification": {"E1": ["MUSAN noise", "DD BACKGROUND", "Cochl N1"],
                            "E2": ["MUSAN speech"], "E3": ["MUSAN music"],
                            "E4": ["Cochl N6", "Cochl N7"], "E5": ["approved loc"],
                            "E6": ["DD HELICOPTER", "Cochl N5", "MUSAN X2"]},
        "stream_file_cap_hours": 4.0,
    }
    target = Path(output)
    payload = "# V5.1 replay contract\n\n```json\n" + json.dumps(body, indent=2, sort_keys=True) + "\n```\n"
    target.write_text(payload, encoding="utf-8", newline="\n")
    return {"path": str(target), "sha256": _sha256(target), **body}


def environment_for_row(row: dict[str, Any]) -> str | None:
    """Map admitted negative stream metadata to its single environment class."""
    source, stratum = str(row.get("source", "")), str(row.get("negative_stratum", ""))
    sub = str(row.get("substratum", row.get("sub_class", ""))).strip().casefold()
    if source == "musan":
        if stratum == "X2":
            return "E6"
        if sub == "noise":
            return "E1"
        if sub == "speech":
            return "E2"
        if sub == "music":
            return "E3"
    if source == "dd":
        if sub == "background":
            return "E1"
        if sub == "helicopter":
            return "E6"
    if source == "cochl":
        return {"N1": "E1", "N6": "E4", "N7": "E4", "N5": "E6"}.get(stratum)
    if source == "loc":
        return "E5"
    return None


def select_validation_streams(rows: Iterable[dict[str, Any]], *, max_hours: float = 4.0) -> list[dict[str, Any]]:
    """Rank eligible whole validation recordings and apply per-class duration caps."""
    if max_hours <= 0 or not math.isfinite(max_hours):
        raise ValueError("stream duration cap must be positive and finite")
    candidates: dict[str, list[dict[str, Any]]] = {}
    for raw in rows:
        row = dict(raw)
        if (row.get("split") != "val" or int(row.get("label", -1)) != 0
                or row.get("label_weak") is not False):
            continue
        env = environment_for_row(row)
        if env is None:
            continue
        row["environment"] = env
        candidates.setdefault(env, []).append(row)
    selected: list[dict[str, Any]] = []
    cap = max_hours * 3600.0
    for env, members in sorted(candidates.items()):
        by_file: dict[str, dict[str, Any]] = {}
        for row in members:
            file_id = str(row["source_file_id"])
            if file_id in by_file and by_file[file_id]["environment"] != row["environment"]:
                raise ValueError(f"one source file maps to multiple environments: {file_id}")
            by_file.setdefault(file_id, row)
        ranked_members = ranked("file_order", [([str(r["source_revision"]), str(r["relative_path"])], r) for r in by_file.values()])
        used, seen = 0.0, set()
        for row in ranked_members:
            file_id = str(row["source_file_id"])
            if file_id in seen:
                raise ValueError(f"duplicate stream source file in {env}: {file_id}")
            seen.add(file_id)
            duration = float(row.get("native_frames", row.get("n_samples"))) / float(row.get("native_sr", row.get("sample_rate")))
            if duration <= 0 or not math.isfinite(duration):
                raise ValueError("stream file has invalid duration")
            if used + duration > cap:
                continue
            used += duration
            selected.append({**row, "nominal_duration_s": duration, "environment": env})
    return sorted(selected, key=lambda row: (row["environment"], row["source_revision"], row["relative_path"]))


def replay_file(audio: np.ndarray, *, source_rate: int, file_id: str, environment: str,
                config, threshold: float | None, score_window: Callable[[np.ndarray], float],
                chunk_samples: int = 4096) -> list[dict[str, Any]]:
    """Replay one whole validation file through the production V4 window/vote primitives."""
    from demo.run_demo import LiveWindowBuilder, VoteSmoother
    from dronedet.features import clip_flags
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("P6 replay requires a nonempty finite mono source waveform")
    if source_rate < 8000 or source_rate > 192000 or (threshold is not None and not math.isfinite(threshold)) or chunk_samples <= 0:
        raise ValueError("P6 replay rate, threshold or chunk size is invalid")
    builder, smoother = LiveWindowBuilder(config, int(source_rate)), VoteSmoother(3)
    records: list[dict[str, Any]] = []
    consumed = 0

    def collect(new_windows: list[np.ndarray]) -> None:
        nonlocal consumed
        starts = builder.emitted_starts[consumed:consumed + len(new_windows)]
        if len(starts) != len(new_windows):
            raise AssertionError("P6 builder window/sample start counts differ")
        consumed += len(new_windows)
        for window, start in zip(new_windows, starts):
            flags = clip_flags(window, config)
            start_sample, end_sample = int(start), int(start) + int(config.win_samples)
            record = {"file_id": file_id, "environment": environment, "continuity_id": "0",
                      "start_sample_16k": start_sample, "end_sample_16k": end_sample,
                      "start_time_s": start_sample / config.sr, "end_time_s": end_sample / config.sr,
                      "status": "standby", "active": False, "score": None,
                      "silent": bool(flags["silent"]), "near_silent": bool(flags["near_silent"]),
                      "clipped": bool(flags["clipped"]), "gap": False, "recovered": False}
            if flags["silent"] or flags["near_silent"]:
                smoother.reset()
                record["reason"] = "quiet"
                records.append(record)
                continue
            score = float(score_window(window))
            if not math.isfinite(score):
                raise ValueError("P6 baseline scorer returned a non-finite logit")
            record["score"] = score
            if threshold is None:
                record["status"] = "score_only"
                record["active"] = None
                records.append(record)
                continue
            decision = smoother.push(score, float(threshold))
            record["raw_vote"] = bool(score >= threshold)
            if decision is None:
                record["status"] = "warming_up"
                record["votes_received"] = len(smoother.values)
            else:
                record["status"] = "scored"
                record["active"] = bool(decision)
                record["votes"] = list(smoother.values)
            records.append(record)

    for start in range(0, values.size, chunk_samples):
        collect(builder.push(values[start:start + chunk_samples]))
    collect(builder.finish())
    return records


def scored_exposure_seconds(windows: Iterable[dict[str, Any]]) -> float:
    """Union scored intervals; omit gaps and recovered/capture discontinuities."""
    intervals: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for raw in windows:
        row = dict(raw)
        if row.get("status") not in {"scored", "score_only"} or row.get("gap") or row.get("recovered"):
            continue
        key = (str(row["environment"]), str(row["file_id"]), str(row.get("continuity_id", "0")))
        start, end = float(row["start_time_s"]), float(row["end_time_s"])
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError("scored window has an invalid exposure interval")
        intervals.setdefault(key, []).append((start, end))
    total = 0.0
    for spans in intervals.values():
        cursor_start = cursor_end = None
        for start, end in sorted(spans):
            if cursor_start is None:
                cursor_start, cursor_end = start, end
            elif start <= cursor_end:
                cursor_end = max(cursor_end, end)
            else:
                total += cursor_end - cursor_start
                cursor_start, cursor_end = start, end
        if cursor_start is not None:
            total += cursor_end - cursor_start
    return total


def merge_episodes(windows: Iterable[dict[str, Any]], *, merge_gap_seconds: float = 5.0) -> list[dict[str, Any]]:
    """Merge active decisions only inside the same uninterrupted file/session."""
    if merge_gap_seconds < 0 or not math.isfinite(merge_gap_seconds):
        raise ValueError("episode merge gap must be finite and nonnegative")
    active = []
    for raw in windows:
        row = dict(raw)
        if row.get("status") != "scored" or row.get("gap") or row.get("recovered") or not row.get("active"):
            continue
        active.append(row)
    active.sort(key=lambda r: (str(r["environment"]), str(r["file_id"]), str(r.get("continuity_id", "0")),
                               float(r["end_time_s"])))
    episodes: list[dict[str, Any]] = []
    for row in active:
        identity = (str(row["environment"]), str(row["file_id"]), str(row.get("continuity_id", "0")))
        end = float(row["end_time_s"])
        if episodes and episodes[-1]["identity"] == identity and float(row["start_time_s"]) - episodes[-1]["end_time_s"] < merge_gap_seconds:
            episodes[-1]["end_time_s"] = max(episodes[-1]["end_time_s"], end)
            episodes[-1]["window_count"] += 1
        else:
            episodes.append({"identity": identity, "environment": identity[0], "file_id": identity[1],
                             "continuity_id": identity[2], "start_time_s": float(row["start_time_s"]),
                             "end_time_s": end, "window_count": 1})
    for episode in episodes:
        episode.pop("identity")
    return episodes


def event_detection_metrics(event: dict[str, Any], windows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Separate pre-event alarms from a new alarm episode during [7,10] seconds."""
    event_id, file_id = str(event["event_id"]), str(event["file_id"])
    matching = [dict(row) for row in windows if str(row.get("event_id", "")) == event_id
                and str(row.get("file_id", "")) == file_id and row.get("status") == "scored"]
    matching.sort(key=lambda row: float(row["end_time_s"]))
    episodes = merge_episodes(matching)
    pre_episodes = [episode for episode in episodes if float(episode["start_time_s"]) < 7.0]
    event_episodes = [episode for episode in episodes if 7.0 <= float(episode["start_time_s"]) <= 10.0]
    hits = [row for row in matching if bool(row.get("active")) and 7.0 <= float(row["end_time_s"]) <= 10.0]
    latency = float(event_episodes[0]["start_time_s"]) - 7.0 if event_episodes else None
    return {"event_id": event_id, "pre_event_alarm": bool(pre_episodes),
            "pre_event_episode_count": len(pre_episodes),
            "new_active_episode_in_event_interval": bool(event_episodes),
            "first_alarm_latency_s": latency,
            "scored_event_windows": sum(7.0 <= float(row["end_time_s"]) <= 10.0 for row in matching)}


def _band_rms(audio: np.ndarray, sr: int) -> float:
    sos = signal.butter(4, [100.0, 4000.0], btype="bandpass", fs=sr, output="sos")
    filtered = signal.sosfiltfilt(sos, np.asarray(audio, dtype=np.float64))
    return float(np.sqrt(np.mean(filtered * filtered, dtype=np.float64)))


def construct_event_mix(bed: np.ndarray, event: np.ndarray, *, snr_db: float, sample_rate: int = 16000,
                        event_start: int = 112000, join_samples: Iterable[int] = ()) -> dict[str, Any]:
    """Create the fixed 20 s bed/4 s event mix with ramps, joins, and peak record."""
    bed = np.asarray(bed, dtype=np.float32)
    event = np.asarray(event, dtype=np.float32).copy()
    if bed.shape != (320000,) or event.shape != (64000,) or sample_rate != 16000:
        raise ValueError("event mixing requires exact 320000/64000 samples at 16 kHz")
    if not math.isfinite(snr_db) or not 0 <= event_start <= bed.size - event.size:
        raise ValueError("invalid event SNR or placement")
    joins = sorted(set(int(position) for position in join_samples))
    if any(position < 0 or position > event.size for position in joins):
        raise ValueError("event join position lies outside event bounds")
    if any(right - left < 160 for left, right in zip(joins, joins[1:])):
        raise ValueError("event join fade regions must not overlap")
    dips = []
    for join in joins:
        left0, left1 = max(0, join - 80), min(event.size, join)
        right0, right1 = max(0, join), min(event.size, join + 80)
        if left1 > left0:
            event[left0:left1] *= np.linspace(1.0, 0.0, left1 - left0, dtype=np.float32)
        if right1 > right0:
            event[right0:right1] *= np.linspace(0.0, 1.0, right1 - right0, dtype=np.float32)
        dips.append({"join_sample": join, "fade_out": [left0, left1], "fade_in": [right0, right1]})
    ramp = np.arange(4000, dtype=np.float64)
    ramp_in = .5 - .5 * np.cos(np.pi * ramp / 3999.0)
    event[:4000] *= ramp_in.astype(np.float32)
    event[-4000:] *= ramp_in[::-1].astype(np.float32)
    measure = slice(4000, 60000)
    event_rms = _band_rms(event[measure], sample_rate)
    bed_region = bed[event_start + 4000:event_start + 60000]
    bed_rms = _band_rms(bed_region, sample_rate)
    if event_rms <= 1e-6 or bed_rms <= 0:
        return {"state": "SKIP", "reason": "zero_or_negligible_filtered_band_rms", "event_band_rms": event_rms,
                "bed_band_rms": bed_rms, "join_dips": dips}
    event_scale = bed_rms * 10.0 ** (snr_db / 20.0) / event_rms
    mix = bed.copy()
    mix[event_start:event_start + event.size] += event * np.float32(event_scale)
    peak = float(np.max(np.abs(mix)))
    common_scale = min(1.0, .99 / peak) if peak > .99 else 1.0
    if common_scale < 1.0:
        mix *= np.float32(common_scale)
    return {"state": "PASS", "mix": mix, "event_scale": float(event_scale),
            "nominal_snr_db": float(snr_db), "pre_common_peak": peak,
            "common_scale": float(common_scale), "event_start_sample": event_start,
            "event_end_sample": event_start + event.size, "outer_ramp_samples": 4000,
            "join_dips": dips, "bed_band_rms": bed_rms, "event_band_rms": event_rms}


def rate_upper_bound(blocks: Iterable[dict[str, Any]], *, environment: str, split: str,
                     replicates: int = 2000) -> dict[str, Any]:
    """Combine exact Poisson and file-block bootstrap one-sided 95% upper rates."""
    items = [dict(row) for row in blocks]
    if not items or replicates != 2000:
        raise ValueError("rate bound requires blocks and exactly 2,000 bootstrap replicates")
    if any(float(row["exposure_s"]) < 0 or int(row["episodes"]) < 0 for row in items):
        raise ValueError("rate block has negative episodes or exposure")
    observed_exposure = math.fsum(float(row["exposure_s"]) for row in items)
    observed_episodes = sum(int(row["episodes"]) for row in items)
    if observed_exposure <= 0:
        return {"state": "NOT_RUN", "reason": "zero observed exposure", "environment": environment, "split": split}
    poisson = float(chi2.ppf(.95, 2 * (observed_episodes + 1)) / (2 * observed_exposure) * 3600.0)
    rates = []
    invalid = 0
    n = len(items)
    for replicate in range(replicates):
        sampled = [items[integer("bootstrap", [environment, split, replicate, draw], n)] for draw in range(n)]
        exposure = math.fsum(float(row["exposure_s"]) for row in sampled)
        if exposure <= 0:
            invalid += 1
            continue
        episodes = sum(int(row["episodes"]) for row in sampled)
        rates.append(episodes / exposure * 3600.0)
    boot = float(np.quantile(rates, .95, method="linear")) if rates else None
    upper = max(poisson, boot) if boot is not None else None
    return {"state": "PASS" if upper is not None else "NOT_RUN", "environment": environment,
            "split": split, "episodes": observed_episodes, "exposure_s": observed_exposure,
            "poisson_upper_per_hour": poisson, "bootstrap_upper_per_hour": boot,
            "upper_per_hour": upper, "bootstrap_replicates": replicates,
            "bootstrap_invalid_zero_exposure": invalid}
