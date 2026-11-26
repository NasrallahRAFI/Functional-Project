"""Dataset inventory helpers for the frozen DroneAudioDataset preparation recipe."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

CLASS_DIRS = ("unknown", "yes_drone")
_DRONE = re.compile(r"^(?P<prefix>.+)-(?P<type>bebop|membo)_(?P<seg>[0-9]{3})_[.]wav$")
_EXTRA = re.compile(r"^extra_membo_D2_[0-9]{4}[.]wav$")
_ESC = re.compile(r"^(?P<fold>[0-9])-(?P<fsid>[0-9]+)-(?P<take>[A-Z])-(?P<cls>[0-9]+)[.]wav$")
_SIL = re.compile(r"^silence[0-9]{3}[.]wav$")
_STEM = re.compile(r"^(?P<stem>doing_the_dishes|exercise_bike|running_tap|pink_noise|white_noise)[0-9]+[.]wav$")
_FAMILIES = (("B_S2_D1", re.compile(r"^B_S2_D1_([0-9]+)$"), "bebop"),
             ("mixed_bebop", re.compile(r"^mixed_([0-9]+)$"), "bebop"),
             ("Membo_0", re.compile(r"^Membo_0_([0-9]+)$"), "membo"),
             ("Membo_1", re.compile(r"^Membo_1_([0-9]+)$"), "membo"),
             ("Membo_2", re.compile(r"^Membo_2_([0-9]+)$"), "membo"),
             ("mixed_membo", re.compile(r"^mixed_membo_([0-9]+)$"), "membo"))


def locate_dataset_root(raw_root: Path) -> Path:
    raw_root = Path(raw_root)
    for candidate in (raw_root, raw_root / raw_root.name):
        if (candidate / "Binary_Drone_Audio").is_dir():
            return candidate
    raise FileNotFoundError(f"Binary_Drone_Audio not found under {raw_root}")


def list_binary_wavs(root: Path) -> list[str]:
    base = Path(root) / "Binary_Drone_Audio"
    return sorted(p.relative_to(base).as_posix() for d in CLASS_DIRS for p in (base / d).glob("*.wav"))


def clip_stats(path: Path) -> dict:
    info = sf.info(str(path))
    pcm, _ = sf.read(str(path), dtype="int16", always_2d=True)
    ch0 = np.ascontiguousarray(pcm[:, 0])
    x = ch0.astype(np.float64) / 32768.0
    rms = float(np.sqrt(np.mean(x ** 2)))
    return {"sr": info.samplerate, "channels": info.channels, "subtype": info.subtype,
            "n_samples": int(ch0.size), "header_frames": int(info.frames),
            "truncated": bool(ch0.size != info.frames), "pcm_sha1": hashlib.sha1(ch0.tobytes()).hexdigest(),
            "all_zero": bool(not ch0.any()), "zero_frac": float(np.mean(ch0 == 0)),
            "rms_dbfs": 20 * math.log10(rms) if rms > 0 else -240.0, "peak": float(np.abs(x).max()),
            "dc": float(x.mean()), "clipped_frac": float(np.mean(np.abs(x) >= 0.999)),
            "clipped_frac_i16": float(np.mean(np.abs(ch0.astype(np.int32)) >= 32767))}


def build_inventory(root: Path, ids: list[str]) -> pd.DataFrame:
    rows = []
    for cid in ids:
        row = {"clip_id": cid, "folder": cid.split("/")[0], "error": ""}
        try:
            row.update(clip_stats(Path(root) / "Binary_Drone_Audio" / cid))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def parse_clip(clip_id: str) -> dict:
    folder, name = clip_id.split("/")
    row = dict(style="", family="none", drone_type="none", recording="", rec_no=-1, segment=-1, group_raw="")
    if folder == "yes_drone":
        row["style"] = "drone"
        if _EXTRA.match(name):
            row.update(family="extra_membo_D2", drone_type="membo", recording="extra_membo_D2", group_raw="extra_membo_D2")
            return row
        match = _DRONE.match(name)
        if not match:
            raise ValueError(f"unrecognised drone name: {clip_id}")
        prefix, kind = match.group("prefix"), match.group("type")
        for family, regex, required_type in _FAMILIES:
            fam_match = regex.match(prefix)
            if fam_match:
                if kind != required_type:
                    raise ValueError(f"family {family} with type {kind}: {clip_id}")
                recording = f"{prefix}-{kind}"
                row.update(family=family, drone_type=kind, recording=recording, group_raw=recording,
                           rec_no=int(fam_match.group(1)), segment=int(match.group("seg")))
                return row
        raise ValueError(f"unknown drone family: {clip_id}")
    if _SIL.match(name):
        row.update(style="silence", group_raw=name[:-4])
    elif _STEM.match(name):
        row.update(style="other", group_raw=_STEM.match(name).group("stem"))
    elif name == "1-7973-A-7_1002.wav":
        row.update(style="other", group_raw=name[:-4])
    elif _ESC.match(name):
        match = _ESC.match(name)
        row.update(style="esc", group_raw=f"{match.group('fold')}-{match.group('fsid')}", segment=int(match.group("cls")[-1]))
    else:
        raise ValueError(f"unrecognised unknown-class name: {clip_id}")
    return row


def l_rule_table(inv: pd.DataFrame, sr: int = 16000, n_fft: int = 1024, hop: int = 256,
                 candidates=(0.5, 1.0, 1.5, 2.0)) -> pd.DataFrame:
    rows = []
    for length in candidates:
        n_win = int((length * sr - n_fft) // hop) + 1
        window = (n_win - 1) * hop + n_fft
        for label, group in inv.groupby("folder"):
            samples = group.n_samples
            rows.append({"L": length, "win_samples": window, "cls": label, "n": len(group),
                         "full": int((samples >= window).sum()),
                         "pad": int(((samples >= window / 2) & (samples < window)).sum()),
                         "reject": int((samples < window / 2).sum())})
    result = pd.DataFrame(rows)
    result["full_pct"] = 100 * result.full / result.n
    return result


def measure_resampler_edge(in_sr: int = 48000, out_sr: int = 16000, quality: str = "HQ",
                           duration: float = 6.0, points: int = 12000) -> dict:
    """Measure the canonical swept response at both common capture rates.

    ``in_sr`` remains a compatibility argument. An additional rate is measured
    when requested; neither of the two registered capture rates is omitted.
    """
    from .resample import fmax_below_passband, measure_swept_passband

    measurements = {}
    for source_rate in sorted({44100, 48000, in_sr}):
        edge, frequency, response = measure_swept_passband(
            source_rate, out_sr, quality=quality, duration_s=duration, points=points)
        measurements[str(source_rate)] = {
            "source_rate": source_rate, "target_rate": out_sr, "edge_hz": edge,
            "frequencies_hz": frequency.tolist(),
            "relative_level_db": [float(value) if np.isfinite(value) else None for value in response],
        }
    edge = min(record["edge_hz"] for record in measurements.values())
    return {"f1db": edge, "fmax": fmax_below_passband(edge),
            "method": "log_swept_chirp_instantaneous_frequency_v1",
            "quality": quality, "target_rate": out_sr, "duration_s": duration,
            "frequency_points": points, "reference_band_hz": [900, 1100],
            "reference": "median response near 1 kHz", "measurements": measurements}


def record_resampler_measurement(config_path: str | Path, record_path: str | Path, *,
                                  measurement: dict | None = None) -> dict:
    """Write measured evidence; a frozen configuration is always read-only.

    A frozen mismatch is returned explicitly for the caller's acceptance gate.
    Existing measurements/configuration are not silently reconciled after freeze.
    """
    import os
    import tempfile
    import yaml
    from .config import load_config

    config_path, record_path = Path(config_path), Path(record_path)
    if record_path.resolve() == config_path.resolve():
        raise ValueError("measurement record must not replace configuration")
    original = config_path.read_bytes()
    config = load_config(config_path)
    result = dict(measurement if measurement is not None else measure_resampler_edge(
        out_sr=config.sr, quality=config.resample_quality))
    if not np.isfinite(result["f1db"]) or not np.isfinite(result["fmax"]):
        raise ValueError("measured passband values must be finite")
    from .resample import fmax_below_passband
    if result["fmax"] != fmax_below_passband(result["f1db"]):
        raise ValueError("measured fmax does not follow the registered passband rule")
    frozen = (config_path.parent / "data/processed/splits/FROZEN.txt").is_file()
    matches = (config.passband_status == "measured"
               and config.measured_passband_hz is not None
               and abs(config.measured_passband_hz - result["f1db"]) <= 1e-8
               and config.fmax == result["fmax"])
    result.update(generated_utc=datetime.now(timezone.utc).isoformat(),
                  library_versions={name: version(name) for name in ("numpy", "scipy", "soxr")},
                  frozen_configuration=frozen, configuration_matches=bool(matches),
                  configuration_updated=not frozen,
                  status="MISMATCH_FROZEN_CONFIG" if frozen and not matches else "PASS")
    if not frozen:
        raw = yaml.safe_load(original.decode("utf-8"))
        raw["frontend"].update(fmax=result["fmax"], measured_passband_hz=result["f1db"],
                               passband_status="measured")
        # Check the candidate before replacing the original configuration.
        from .config import config_from_mapping
        config_from_mapping(raw)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=config_path.parent,
                                         delete=False, suffix=".tmp") as stream:
            temporary = Path(stream.name)
            yaml.safe_dump(raw, stream, sort_keys=False, allow_unicode=True)
        try:
            os.replace(temporary, config_path)
        finally:
            temporary.unlink(missing_ok=True)
    elif config_path.read_bytes() != original:
        raise RuntimeError("frozen configuration changed during measurement")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=record_path.parent,
                                     delete=False, suffix=".tmp") as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    try:
        os.replace(temporary, record_path)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def environment_log(path: Path, *, dataset_sha256=None, download_date=None, repository_commit=None,
                    frontend_params_hash=None, zip_path: Path | None = None,
                    source_repository=None, repository_commit_zip_sha256=None,
                    repository_commit_basis=None) -> dict:
    packages = {}
    for name in ("numpy", "pandas", "librosa", "soundfile", "soxr", "scikit-learn", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    payload = {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
               "logged_utc": datetime.now(timezone.utc).isoformat(), "dataset_sha256": dataset_sha256,
               "zip_sha256": None, "zip_bytes": None, "zip_matches_exploration": None,
               "download_date": download_date, "repository_commit": repository_commit,
               "source_repository": source_repository,
               "repository_commit_basis": repository_commit_basis,
               "frontend_params_hash": frontend_params_hash}
    if zip_path is not None and Path(zip_path).is_file():
        hasher = hashlib.sha256()
        with Path(zip_path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
        payload["zip_sha256"] = hasher.hexdigest()
        payload["zip_bytes"] = Path(zip_path).stat().st_size
        payload["zip_matches_exploration"] = (
            payload["zip_sha256"] == "90d09a04ce08bd08f0e76d92d0aebd394346317460dad4bdc8344c2ea9683f37"
            and payload["zip_bytes"] == 580430228
        )
    if repository_commit_zip_sha256 is not None and payload["zip_sha256"] != repository_commit_zip_sha256:
        payload["repository_commit"] = None
        payload["repository_commit_basis"] = "not assigned: local archive SHA-256 did not match the verified source archive"
    elif repository_commit_zip_sha256 is not None and payload["zip_sha256"] == repository_commit_zip_sha256:
        payload["repository_commit_basis"] = repository_commit_basis
    elif repository_commit is not None and zip_path is None:
        payload["repository_commit"] = None
        payload["repository_commit_basis"] = "not assigned: source archive was unavailable for verification"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


class V4PreparationStop(RuntimeError):
    """A failed fixed-plan gate; never retry with a different rule or seed."""


def _v4_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _v4_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def v4_descriptor(audio: np.ndarray, sample_rate: int = 16000) -> dict:
    """Whole-clip periodogram descriptors; weak-positive estimator approved by owner."""
    from scipy.signal import periodogram

    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("descriptors require a finite nonempty mono waveform")
    frequencies, density = periodogram(x, fs=sample_rate, window="boxcar",
                                      detrend=False, return_onesided=True, scaling="density")
    bin_hz = sample_rate / x.size
    power = density * bin_hz
    total = float(power.sum())
    band = float(power[(frequencies >= 100) & (frequencies <= 4000)].sum())
    rms = float(np.sqrt(np.mean(x * x)))
    rolloff = float(frequencies[min(int(np.searchsorted(np.cumsum(power), .99 * total)), len(frequencies) - 1)]) if total else 0.0
    return {"rms_dbfs": float(20 * np.log10(rms)) if rms else -240.0,
            "rolloff99_hz": rolloff,
            "band_100_4000_dbfs": float(10 * np.log10(band)) if band else -240.0,
            "peak": float(np.max(np.abs(x))), "dc": float(x.mean()),
            "zero_frac": float(np.mean(x == 0)),
            "clipped_frac": float(np.mean(np.abs(x) >= .999)),
            "pcm_sha256": hashlib.sha256(np.asarray(audio, dtype="<f4").tobytes()).hexdigest()}


def v4_source_only_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Descriptive pooled pre-split source-majority baseline; no model or test scoring."""
    for source, group in frame.groupby("source"):
        counts = group.label.value_counts()
        if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
            raise V4PreparationStop(f"source-majority tie is undefined by the plan: {source}")
    majority = frame.groupby("source").label.agg(lambda x: int(x.value_counts().idxmax()))
    predictions = frame.source.map(majority).astype(int)
    recalls = {str(label): float((predictions[frame.label == label] == label).mean()) for label in (0, 1)}
    balanced = sum(recalls.values()) / 2
    table = frame.groupby(["source", "label"]).size().rename("clips").reset_index()
    table["source_majority_prediction"] = table.source.map(majority)
    return table, {"balanced_accuracy": balanced, "class_recalls": recalls,
                   "majority": {str(k): int(v) for k, v in majority.items()},
                   "population": "pooled surviving clips before split; owner-approved",
                   "ties": "STOP; no tie-break rule is invented",
                   "is_model_evaluation": False}


def run_v4_preparation(config_path: str | Path) -> dict:
    """Execute the fixed V4 A–F plan, preserving raw sources and stopping on gates."""
    from concurrent.futures import ThreadPoolExecutor
    import os
    import subprocess
    import yaml
    from .config import config_from_mapping
    from . import features, splits

    config_path = Path(config_path).resolve()
    checkout = config_path.parent
    base = checkout.parent.parent
    doc = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fe = config_from_mapping(doc["frontend"])
    root = base / "storage/corpora/corpus-002"
    clips_root = base / "storage/derived/corpus-002/dd-clips"
    split_root = base / "storage/splits/split-002"
    report = base / "reports/study-v4/DATA_PREPARATION_CNN_V4_0_REPORT.md"
    audits = root / "audits"
    gates: list[dict] = []
    source_hashes: dict[str, str] = {}
    state: dict = {"status": "RUNNING", "phase": "preflight", "plan": "CNN V4.0",
                   "gates": gates, "test_audio_opened_after_freeze": False,
                   "salford_processed": False, "models_loaded": False,
                   "normalization_statistics_fitted": False, "feature_caches_written": False}

    def table(name, frame):
        audits.mkdir(parents=True, exist_ok=True)
        frame.to_csv(audits / name, index=False, lineterminator="\n", float_format="%.12g")

    def gate(name, passed, actual):
        gates.append({"gate": name, "passed": bool(passed), "actual": str(actual)})
        _v4_json(audits / "gates.json", {"gates": gates})
        if not passed:
            raise V4PreparationStop(f"{name}: {actual}")

    def progress(phase):
        state["phase"] = phase
        _v4_json(audits / "status.json", state)
        print(phase, flush=True)

    def write_report():
        report.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# CNN V4.0 data preparation report", "", f"Status: **{state['status']}**. Last phase: {state['phase']}.",
                 "", "Sources: Al-Emadi binary audio (ae), Drone-detection-dataset audio (dd). Salford is excluded and was not inventoried or processed.",
                 "", "## Execution and fixed protocol", "",
                 "Seed 42; channel 0; 16 kHz; soxr HQ; dd segments are 16,000 samples, non-overlapping float32 WAVs. Al-Emadi is referenced without copying or resampling. No level normalization or dither.",
                 "", "The owner approved the pooled pre-split source-majority audit and the absolute ≤3 dB weak-positive comparison. Band power uses the one-sided SciPy periodogram, boxcar window, no detrending, inclusive 100–4,000 Hz bins and amplitude-1 reference. Zero-energy descriptor values use the existing −240 dB sentinel.",
                 "", "Original Al-Emadi test-use history remains applicable; the new split does not establish previously unseen recordings. No model was loaded or scored; no mel cache, fitted normalization statistics, threshold or runtime package was produced.",
                 "", "## Gates", "", "| Gate | Passed | Observed |", "|---|---|---|"]
        lines += [f"| {g['gate']} | {g['passed']} | {g['actual'].replace('|', '/')} |" for g in gates]
        for key in ("inventory_counts", "ae_cleaning", "dd_segmentation", "duplicates", "source_only", "split_counts", "smoke", "manifest_sha256", "warnings"):
            if key in state:
                lines += ["", f"## {key.replace('_', ' ')}", "", "```json", json.dumps(state[key], indent=2), "```"]
        if "stop_reason" in state:
            lines += ["", "## STOP finding", "", state["stop_reason"], "", "No rule, seed, fold count or label was changed to bypass this finding."]
        lines += ["", "## Evidence and limitations", "",
                  "Detailed CSV/JSON/PNG evidence is in storage/corpora/corpus-002/audits; source license records and environment provenance are beside it. A requirement audit is recorded separately after completion verification.",
                  "", "Al-Emadi was recorded indoors and covers Bebop/Membo. dd has only 30 original recordings per class. Quiet or distant drone segments retain source labels; weak_positive is a descriptor flag, not verified audibility. Calibration and segment-label uncertainty are reasons Salford is excluded.",
                  "", "Rejected clips too short for a canonical crop have crop_all_zero and stats_eligible recorded as not applicable (empty CSV values), rather than fabricated window measurements. No such clip enters a retained split."]
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if (split_root / "FROZEN.json").exists() or (audits / "status.json").exists():
        raise FileExistsError("V4 output exists; preserve it and inspect status before any new execution")
    try:
        for path in (root, clips_root, split_root, report.parent):
            if not path.resolve().is_relative_to(base.resolve()):
                raise ValueError("output escapes project")
        root.mkdir(parents=True, exist_ok=True)
        progress("A1 environment and implementation preflight")
        gate("SEED", doc.get("seed") == 42, doc.get("seed"))
        gate("FRONTEND", (fe.sr, fe.channel, fe.n_fft, fe.hop, fe.n_mels, fe.fmax, fe.win_samples, fe.n_win) == (16000, 0, 1024, 256, 64, 7500, 7936, 28), "16 kHz / 1024 / 256 / 64 / 7500 / 7936 / 28")
        required = {"clip_flags": features.clip_flags, "crop_flags": features.power_quality_flags,
                    "eligibility": features.stats_exclude, "dc_flags": splits.flag_columns,
                    "groups": splits.assign_group, "cleaning": splits.apply_cleaning}
        gate("EXISTING_FLAGS", all(callable(x) for x in required.values()), list(required))
        sources = {
            "ae": base / "storage/sources/al-emadi/al-emadi-r001/original",
            "dd": base / "storage/sources/drone-detection-thesis/thesis-r001/original"}
        environment = {"python": platform.python_version(), "executable": sys.executable,
                       "execution_environment": "existing pinned interpreter reused without modifying its environment; all code and output paths are V4-owned",
                       "date_utc": datetime.now(timezone.utc).isoformat(),
                       "packages": {name: version(name) for name in ("librosa", "soundfile", "soxr", "scikit-learn", "numpy", "scipy", "pandas")},
                       "sources": {k: str(v) for k, v in sources.items()},
                       "config_sha256": _v4_sha(config_path),
                       "plan_sha256": _v4_sha(checkout / "docs/v4/DATA_PREPARATION_CNN_V4_0.md"),
                       "git_commit": subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"]).decode().strip(),
                       "code_files": {str(p.resolve().relative_to(checkout)): _v4_sha(p) for p in [Path(__file__), checkout / "src/dronedet/features.py", checkout / "src/dronedet/splits.py", checkout / "src/dronedet/resample.py"]}}
        zip_path = Path(os.environ.get("USERPROFILE", "")) / "Downloads/DroneAudioDataset-master.zip"
        environment["al_emadi_zip"] = {"exists": zip_path.is_file(), "sha256": _v4_sha(zip_path) if zip_path.is_file() else None}
        snapshot = audits / "implementation_source"
        for relative in environment["code_files"]:
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((checkout / relative).read_bytes())
        (snapshot / "config.yaml").write_bytes(config_path.read_bytes())
        environment["implementation_source_snapshot"] = str(snapshot)
        environment["git_commit_is_complete_working_source_identity"] = False
        environment["working_source_identity"] = "per-file hashes and exact implementation_source snapshot; no assumption that HEAD includes uncommitted code"
        _v4_json(root / "environment_log_v4.json", environment)

        progress("A2 Al-Emadi inventory")
        ae_ids = list_binary_wavs(sources["ae"])
        def inventory_one(cid):
            path = sources["ae"] / "Binary_Drone_Audio" / cid
            return {"clip_id": cid, "folder": cid.split("/")[0], "error": "", **clip_stats(path),
                    "source_file_sha256": _v4_sha(path), "file": str(path)}
        with ThreadPoolExecutor(max_workers=4) as pool:
            ae = pd.DataFrame(list(pool.map(inventory_one, ae_ids)))
        table("A2_ae_inventory.csv", ae)
        counts = ae.folder.value_counts().to_dict()
        parsed = pd.DataFrame([parse_clip(cid) for cid in ae.clip_id])
        state["inventory_counts"] = {"ae": counts, "ae_drone_types": parsed.drone_type.value_counts().to_dict()}
        gate("A2_COUNTS", counts == {"unknown": 10372, "yes_drone": 1332} and len(ae) == 11704, counts)
        gate("A2_SUBFAMILIES", parsed.drone_type.eq("bebop").sum() == 666 and parsed.drone_type.eq("membo").sum() == 666 and parsed["style"].value_counts().to_dict() == {"esc": 10000, "drone": 1332, "silence": 301, "other": 71}, parsed["style"].value_counts().to_dict())
        gate("A2_FORMAT", (ae.sr.eq(16000) & ae.channels.eq(1) & ae.subtype.eq("PCM_16") & ~ae.truncated).all(), ae.groupby(["sr", "channels", "subtype"]).size().to_dict())
        source_hashes.update({row.file: row.source_file_sha256 for row in ae.itertuples()})

        progress("A3 dd inventory")
        dd_records = []
        for path in sorted((sources["dd"] / "Data/Audio").glob("*.wav")):
            info = sf.info(str(path))
            dd_records.append({"file": str(path), "recording": path.stem, "sub_class": path.stem.split("_")[0],
                               "native_sr": info.samplerate, "channels": info.channels, "subtype": info.subtype,
                               "native_frames": info.frames, "duration_s": info.duration, "sha256": _v4_sha(path)})
        dd_inventory = pd.DataFrame(dd_records)
        table("A3_dd_inventory.csv", dd_inventory)
        dd_counts = dd_inventory.sub_class.value_counts().to_dict()
        state["inventory_counts"]["dd"] = dd_counts
        gate("A3_COUNTS", dd_counts == {"BACKGROUND": 30, "DRONE": 30, "HELICOPTER": 30}, dd_counts)
        source_hashes.update({row.file: row.sha256 for row in dd_inventory.itertuples()})

        progress("A4 source license records")
        ae_license = (sources["ae"] / "LICENSE.md").read_text(encoding="utf-8")
        ae_readme = (sources["ae"] / "README.md").read_text(encoding="utf-8")
        dd_license = (sources["dd"] / "LICENSE").read_text(encoding="utf-8")
        dd_readme = (sources["dd"] / "README.md").read_text(encoding="utf-8")
        context_lines = (checkout / "docs/history/original_context/PROJECT_CONTEXT_V8.md").read_text(encoding="utf-8").splitlines()
        d031 = next(line for line in context_lines if line.startswith("- **D031"))
        license_text = ["# Included-source license record — V4.0", "", "Only ae and dd are included. These are source-document records, not new legal permissions.", "", "## Al-Emadi", "", d031, "", "### Local LICENSE.md", "", ae_license,
                        "", "README was read; its citation and separate third-party terms remain applicable. No new third-party license verification is claimed.", "", "## Drone-detection-dataset", "", "Local LICENSE is CC0 1.0 Universal. README requests citation of Svanström (2020), the ICPR sensor paper, or Svanström, Alonso-Fernandez and Englund (2021), A Dataset for Multi-Sensor Drone Detection. CC0 provides a public-domain dedication and fallback license; its limitations/disclaimers remain as written.", "", "### Local LICENSE", "", dd_license,
                        "", "## Source document hashes", "", json.dumps({"ae_LICENSE": _v4_sha(sources["ae"] / "LICENSE.md"), "ae_README": _v4_sha(sources["ae"] / "README.md"), "dd_LICENSE": _v4_sha(sources["dd"] / "LICENSE"), "dd_README": _v4_sha(sources["dd"] / "README.md")}, indent=2)]
        (root / "source_licences_v4.md").write_text("\n".join(license_text) + "\n", encoding="utf-8")
        gate("A4_LICENSES_READ", bool(ae_license and ae_readme and dd_license and dd_readme and d031), "README/LICENSE read for ae and dd; D031 copied")

        progress("B1 Al-Emadi grouping and cleaning")
        ae = pd.concat([ae.reset_index(drop=True), parsed], axis=1)
        ae = splits.flag_columns(ae, doc["cleaning"])
        gate("B1_GROUP_COUNTS", ae.loc[ae.label.eq(1), "group"].nunique() == 203 and ae.loc[ae["style"].eq("esc"), "group"].nunique() == 1528, {"positive": int(ae.loc[ae.label.eq(1), "group"].nunique()), "esc": int(ae.loc[ae["style"].eq("esc"), "group"].nunique())})
        ae, representatives = splits.apply_cleaning(ae, fe.win_samples)
        short = ae.loc[ae.exclusion_reason.eq("too_short"), "clip_id"].tolist()
        gate("B1_SHORT", set(short) == {"unknown/silence300.wav", "unknown/doing_the_dishes019.wav"}, short)
        gate("B1_ZEROS", int(ae.all_zero.sum()) == 1089 and ae.loc[ae.all_zero, "pcm_sha1"].nunique() == 10 and len(representatives) == 9, {"clips": int(ae.all_zero.sum()), "contents": int(ae.loc[ae.all_zero, "pcm_sha1"].nunique()), "representatives": len(representatives)})
        gate("B1_KEPT", int(ae.kept.sum()) == 10623 and int((ae.kept & ae.label.eq(1)).sum()) == 1332 and int((ae.kept & ae.label.eq(0)).sum()) == 9291, ae.loc[ae.kept, "label"].value_counts().to_dict())
        representatives_by_pcm = dict(zip(ae.loc[ae.clip_id.isin(representatives), "pcm_sha1"], ae.loc[ae.clip_id.isin(representatives), "clip_id"]))
        ae["duplicate_representative"] = ae.pcm_sha1.map(representatives_by_pcm).fillna("")
        ae["source"] = "ae"
        ae["source_clip_id"] = ae.clip_id
        ae["source_group"] = ae.group
        ae["group"] = "ae:" + ae.group
        ae["clip_id"] = "ae:" + ae.clip_id
        ae["source_revision"] = "al-emadi-r001"
        ae["sub_class"] = np.where(ae.label.eq(1), ae.drone_type, ae["style"])
        ae["native_sr"] = ae.sr
        ae["start_sample"] = 0
        ae["start_sample_domain"] = "standardized-16kHz"
        ae["native_start_sample"] = 0
        ae["weak_positive"] = False
        state["ae_cleaning"] = {"input": len(ae), "retained": int(ae.kept.sum()), "rejections": ae.loc[~ae.kept, "exclusion_reason"].value_counts().to_dict(), "zero_representatives": representatives, "positive_groups": 203}
        table("B1_ae_cleaning.csv", ae)

        return _v4_finish_preparation(doc, fe, checkout, base, sources, ae, dd_inventory,
                                      source_hashes, state, table, gate, progress, write_report,
                                      root, clips_root, split_root, audits)
    except Exception as exc:
        state.update(status="STOP", stop_reason=f"{type(exc).__name__}: {exc}")
        _v4_json(audits / "status.json", state)
        write_report()
        raise


def _v4_finish_preparation(doc, fe, checkout, base, sources, ae, dd_inventory,
                           source_hashes, state, table, gate, progress, write_report,
                           root, clips_root, split_root, audits):
    from concurrent.futures import ThreadPoolExecutor
    from . import features, splits

    def quality(x):
        flags = features.clip_flags(x, fe)
        fixed, length_status = features.fix_length(x, fe)
        if fixed.size:
            power = features.mel_power(fixed, fe)
            crop = features.power_quality_flags(power, fe, silent=flags["silent"], near_silent=flags["near_silent"])
            flags.update(crop)
        else:
            flags.update(crop_all_zero=None, stats_eligible=None, stats_exclude=None, win_start_frame=None)
        return {**v4_descriptor(x, fe.sr), **flags, "crop_flag_applicable": bool(fixed.size), "length_status": length_status}

    # Warm the library's lazy frontend with synthetic data before parallel reads.
    quality(np.zeros(16000, dtype=np.float32))
    progress("B1 Al-Emadi whole-clip descriptors and existing flags")
    def measure_ae(row):
        x = features.load_wave(row.file, fe)
        return quality(x)
    with ThreadPoolExecutor(max_workers=4) as pool:
        measurements = []
        for index, measured in enumerate(pool.map(measure_ae, ae.itertuples()), 1):
            measurements.append(measured)
            if index % 1000 == 0:
                print(f"ae descriptors {index}/{len(ae)}", flush=True)
    for key in measurements[0]:
        ae[key] = [row[key] for row in measurements]
    ae["dc_offset"] = ae.dc_flag
    ae["duplicate_representative"] = ae.duplicate_representative.map(lambda value: "ae:" + value if value else "")

    progress("B2/C dd resampling, segmentation, float32 readback and flags")
    clips_root.mkdir(parents=True, exist_ok=True)
    dd_rows, segmentation = [], []
    for item in dd_inventory.itertuples():
        original = features.load_wave(Path(item.file), fe)
        count, remainder = divmod(len(original), 16000)
        segmentation.append({"recording": item.recording, "sub_class": item.sub_class,
                             "standardized_samples": len(original), "segments": count,
                             "discarded_remainder_samples": remainder, "source_sha256": item.sha256})
        for segment in range(count):
            start = segment * 16000
            x = np.ascontiguousarray(original[start:start + 16000], dtype=np.float32)
            destination = clips_root / f"dd__{item.recording}__{segment * 1000:07d}.wav"
            if destination.exists():
                raise FileExistsError(destination)
            sf.write(str(destination), x, 16000, subtype="FLOAT")
            readback, sr = sf.read(str(destination), dtype="float32", always_2d=True)
            info = sf.info(str(destination))
            ok = (sr == 16000 and readback.shape == (16000, 1) and info.subtype == "FLOAT"
                  and np.isfinite(readback).all() and np.array_equal(readback[:, 0], x))
            if not ok:
                gate("C4_READBACK", False, destination.name)
            if float(np.max(np.abs(x))) > 1:
                gate("C_TARGET_RANGE", False, {"file": destination.name, "peak": float(np.max(np.abs(x)))})
            label = int(item.sub_class == "DRONE")
            measured = quality(x)
            row = {"clip_id": destination.stem, "file": str(destination),
                   "label": label, "label_name": "drone" if label else "unknown",
                   "folder": "yes_drone" if label else "unknown", "source": "dd",
                   "source_revision": "thesis-r001", "source_file": item.file,
                   "source_clip_id": item.recording, "source_file_sha256": item.sha256,
                   "sub_class": item.sub_class, "drone_type": "not_provided" if label else "none",
                   "family": "dd_" + item.sub_class.lower(), "style": "dd", "recording": item.recording,
                   "group_raw": "dd:" + item.recording, "source_group": item.recording,
                   "rec_no": -1, "segment": segment, "n_samples": len(x), "sr": 16000,
                   "native_sr": item.native_sr, "start_sample": start,
                   "start_sample_domain": "standardized-16kHz", "native_start_sample": segment * item.native_sr,
                   "duration_s": 1.0, "all_zero": bool(np.all(x == 0)),
                   "pcm_sha1": hashlib.sha1(x.astype("<f4").tobytes()).hexdigest(),
                   "pcm_identity_encoding": "float32-le", "kept": True, "exclusion_reason": "",
                   "pinned": False, "weak_positive": False, "duplicate_representative": "", **measured}
            dd_rows.append(row)
    dd = pd.DataFrame(dd_rows)
    dd = splits.flag_columns(dd, doc["cleaning"])
    dd["dc_offset"] = dd.dc_flag
    background_median = float(dd.loc[dd.sub_class.eq("BACKGROUND"), "band_100_4000_dbfs"].median())
    dd["weak_positive"] = dd.label.eq(1) & dd.band_100_4000_dbfs.sub(background_median).abs().le(3)
    ae["pcm_identity_encoding"] = "int16-le-original"
    ae["source_file"] = ae.file
    table("C_dd_segmentation.csv", pd.DataFrame(segmentation))
    table("C_dd_written_clips.csv", dd)
    gate("C4_ALL_READBACK", len(dd) == sum(x["segments"] for x in segmentation), {"clips": len(dd), "all_float32_readbacks_exact": True})
    state["dd_segmentation"] = {"recordings": len(dd_inventory), "written_segments": len(dd),
                                "class_counts": dd.sub_class.value_counts().to_dict(),
                                "discarded_samples": sum(x["discarded_remainder_samples"] for x in segmentation),
                                "background_median_band_dbfs": background_median,
                                "weak_positive": int(dd.weak_positive.sum())}

    progress("D duplicates and silence selection")
    full = pd.concat([ae, dd], ignore_index=True, sort=False)
    full["split"] = "excluded"
    duplicate_rows = []
    for digest, indices in full.groupby("pcm_sha256").groups.items():
        group = full.loc[indices].sort_values("clip_id")
        if len(group) < 2:
            continue
        for row in group.itertuples():
            duplicate_rows.append({"sha256": digest, "clip_id": row.clip_id, "source": row.source,
                                   "label": row.label, "n_samples": row.n_samples,
                                   "kept_after_B": bool(row.kept), "copies": len(group),
                                   "cross_source": group.source.nunique() > 1})
        if group.label.nunique() > 1:
            table("D_exact_duplicates.csv", pd.DataFrame(duplicate_rows))
            gate("D_CROSS_CLASS", False, group.clip_id.tolist())
        retained = group[group.kept]
        if len(retained) > 1:
            # B1.3 already selected AE zero representatives; preserve that explicit
            # exception before applying the general first-id rule to other groups.
            special = retained[retained.source.eq("ae") & retained.silent]
            representative = special.iloc[0] if len(special) else retained.iloc[0]
            remove = retained.index.difference([representative.name])
            full.loc[remove, "kept"] = False
            full.loc[remove, "exclusion_reason"] = "exact_float32_duplicate"
            full.loc[remove, "duplicate_representative"] = representative.clip_id
    remove_silence = full.kept & full.silent & ~full.source.eq("ae")
    full.loc[remove_silence, ["kept", "exclusion_reason"]] = [False, "silent_non_ae"]
    duplicates = pd.DataFrame(duplicate_rows, columns=["sha256", "clip_id", "source", "label", "n_samples", "kept_after_B", "copies", "cross_source"])
    table("D_exact_duplicates.csv", duplicates)
    table("D_exclusions.csv", full.loc[~full.kept])
    gate("D_CROSS_CLASS", True, "no float32 identity spans labels")
    kept = full[full.kept].copy()
    gate("D_ZERO_REPRESENTATIVES", int((kept.source.eq("ae") & kept.silent).sum()) == 9 and not kept.loc[~kept.source.eq("ae"), "silent"].any(), "nine AE representatives; no other retained silence")
    gate("D_NO_DUPLICATES", not kept.pcm_sha256.duplicated().any(), "unique retained float32 SHA-256")
    state["duplicates"] = {"groups": int(duplicates.sha256.nunique()), "cross_source_groups": int(duplicates.loc[duplicates.cross_source, "sha256"].nunique()), "exclusions": full.loc[~full.kept, "exclusion_reason"].value_counts().to_dict(), "retained": len(kept)}

    progress("E pre-split source and descriptor audits")
    summaries = []
    for (source, label), group in kept.groupby(["source", "label"]):
        row = {"source": source, "label": int(label), "clips": len(group),
               "seconds": float(group.duration_s.sum()), "groups": int(group.group.nunique())}
        for column in ("rms_dbfs", "rolloff99_hz"):
            row.update({column + "_q25": float(group[column].quantile(.25)), column + "_median": float(group[column].median()), column + "_q75": float(group[column].quantile(.75))})
        for column in ("clipped", "dc_offset", "silent", "near_silent", "weak_positive"):
            row[column + "_fraction"] = float(group[column].astype(bool).mean())
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    table("E_source_class_summary.csv", summary)
    source_table, baseline = v4_source_only_audit(kept)
    table("E_source_only_baseline.csv", source_table)
    _v4_json(audits / "E_source_only_baseline.json", baseline)
    state["source_only"] = baseline
    gate("G-SRC", baseline["balanced_accuracy"] <= .75, baseline["balanced_accuracy"])
    warnings = []
    for source, rows in summary.groupby("source"):
        by = rows.set_index("label")
        level = abs(float(by.loc[1, "rms_dbfs_median"] - by.loc[0, "rms_dbfs_median"]))
        rolloff = abs(float(by.loc[1, "rolloff99_hz_median"] - by.loc[0, "rolloff99_hz_median"]))
        if level > 6 or rolloff > 1000:
            warnings.append({"source": source, "median_level_difference_db": level,
                             "median_rolloff_difference_hz": rolloff,
                             "level_warning": level > 6, "rolloff_warning": rolloff > 1000})
    state["warnings"] = warnings
    table("E_shortcut_warnings.csv", pd.DataFrame(warnings, columns=["source", "median_level_difference_db", "median_rolloff_difference_hz", "level_warning", "rolloff_warning"]))

    # Verify byte integrity before a new test split is assigned. No raw source
    # is opened for integrity checks after the new manifest freeze.
    progress("E included-source integrity verification before split")
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda item: _v4_sha(Path(item[0])) == item[1], source_hashes.items()))
    gate("RAW_READ_ONLY", all(values), {"verified_included_source_files": len(values), "unchanged": sum(values)})
    _v4_json(audits / "raw_read_only.json", {"files": len(values), "all_unchanged": all(values), "completed_before_test_assignment": True})

    progress("F fixed group-aware split and gates")
    kept["stratum"] = kept.label.astype(str) + "|" + kept.source
    kept["split"] = "train"
    free = kept.loc[~kept.pinned].copy()
    assignments = splits.two_stage_split(free, int(doc["seed"]), label_column="stratum")
    kept.loc[assignments.index, "split"] = assignments
    gate("G-GRP", kept.groupby("group").split.nunique().eq(1).all(), "one split per source-qualified group")
    gate("G-DUP", kept.groupby("pcm_sha256").split.nunique().eq(1).all(), "one split per float32 identity")
    fractions = kept.groupby(["split", "label"]).size().unstack(fill_value=0).div(kept.groupby("label").size(), axis=1)
    table("F_class_fractions.csv", fractions.reset_index())
    gate("G-FRAC", fractions.loc[["val", "test"]].ge(.12).all().all() and fractions.loc[["val", "test"]].le(.17).all().all(), fractions.loc[["val", "test"]].to_dict())
    positive_groups = kept.loc[kept.label.eq(1)].groupby("split").group.nunique()
    gate("G-POS", all(int(positive_groups.get(split, 0)) >= 25 for split in ("val", "test")), positive_groups.to_dict())
    stratum_groups = kept.groupby(["stratum", "split"]).group.nunique().unstack(fill_value=0)
    eligible = kept.groupby("stratum").group.nunique().ge(6)
    table("F_stratum_groups.csv", stratum_groups.reset_index())
    gate("G-STR", stratum_groups.loc[eligible, ["val", "test"]].ge(1).all().all(), stratum_groups.to_dict())
    gate("F_PINNED", kept.loc[kept.pinned, "split"].eq("train").all(), int(kept.pinned.sum()))
    full["stratum"] = full.label.astype(str) + "|" + full.source
    full.loc[kept.index, "split"] = kept.split
    counts = kept.groupby(["split", "source", "label"]).agg(clips=("clip_id", "size"), groups=("group", "nunique"), seconds=("duration_s", "sum")).reset_index()
    table("F_split_counts.csv", counts)
    state["split_counts"] = counts.to_dict("records")

    # PNG audits use only already-computed descriptors and manifest counts.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def barplot(name, series, title, ylabel):
        fig, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
        series.plot.bar(ax=axis, color="#315879")
        axis.set(title=title, ylabel=ylabel, xlabel="")
        axis.tick_params(axis="x", labelrotation=25)
        fig.savefig(audits / name, dpi=150)
        plt.close(fig)
    barplot("A_raw_recording_counts.png", pd.Series({"ae unknown": 10372, "ae drone": 1332, **{"dd " + str(k): int(v) for k, v in dd_inventory.sub_class.value_counts().items()}}), "Raw source recording/clip counts", "WAV files")
    barplot("B_ae_cleaning.png", ae.exclusion_reason.replace("", "retained").value_counts(), "Al-Emadi cleaning", "Input clips")
    barplot("C_dd_segments.png", dd.groupby("sub_class").size(), "dd one-second segments written", "Segments")
    barplot("D_exclusions.png", full.loc[~full.kept, "exclusion_reason"].value_counts(), "Excluded clips by recorded reason", "Clips")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for axis, column, title in zip(axes, ["rms_dbfs", "rolloff99_hz"], ["Whole-clip RMS (dBFS)", "99% power rolloff (Hz)"]):
        ordered = list(kept.groupby(["source", "label"]))
        axis.boxplot([g[column].to_numpy() for _, g in ordered], tick_labels=[f"{s} / class {l}" for (s, l), _ in ordered], showfliers=False)
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=25)
    fig.savefig(audits / "E_source_descriptors.png", dpi=150)
    plt.close(fig)
    barplot("F_split_counts.png", counts.set_index(["split", "source", "label"]).clips, "Frozen split composition", "Clips")

    progress("F manifest publication; test audio locked after this point")
    extra_columns = [column for column in full if column not in splits.MANIFEST_COLUMNS]
    hashes = {"full": splits.write_manifest(full, root / "manifest_v4_full.csv", extra_columns=extra_columns)}
    for split in ("train", "val", "test"):
        hashes[split] = splits.write_manifest(full.loc[full.kept & full.split.eq(split)], split_root / f"{split}.csv", extra_columns=extra_columns)
    _v4_json(split_root / "FROZEN.json", {"study": "study-v4", "corpus": "corpus-002", "split": "split-002", "seed": int(doc["seed"]), "manifest_sha256": hashes, "gates_passed_before_freeze": True, "test_access": "locked after publication", "prior_source_test_use": "Al-Emadi source has previously evaluated groups; this is a new split, not a new unseen source"})
    state["manifest_sha256"] = hashes

    progress("F training-only canonical frontend smoke check")
    train = kept[kept.split.eq("train")]
    smoke_rows = []
    for source in ("ae", "dd"):
        selected = train[train.source.eq(source)].sample(n=20, random_state=int(doc["seed"])).sort_values("clip_id")
        for row in selected.itertuples():
            x = features.load_wave(Path(row.file), fe)
            tensor, flags = features.wave_to_input(x, fe, {"mean": np.zeros(64, np.float32), "std": np.ones(64, np.float32)})
            smoke_rows.append({"clip_id": row.clip_id, "source": source, "split": "train", "shape": str(tuple(tensor.shape)), "finite": bool(np.isfinite(tensor).all()), "correct_shape": tensor.shape == (1, 64, 28)})
    smoke = pd.DataFrame(smoke_rows)
    table("F_training_frontend_smoke.csv", smoke)
    gate("F_SMOKE", len(smoke) == 40 and smoke.finite.all() and smoke.correct_shape.all(), "20 training clips per included source, shape (1,64,28), finite")
    state["smoke"] = {"clips": 40, "split": "train", "shape": [1, 64, 28], "finite": True,
                       "normalization": "identity placeholders only to exercise canonical tensor interface; no data-fitted statistics computed or persisted"}
    state.update(status="PASS", phase="complete", manifest_rows=len(full), retained_rows=len(kept))
    _v4_json(audits / "status.json", state)
    write_report()
    print(json.dumps({"status": "PASS", "retained": len(kept), "hashes": hashes}), flush=True)
    return state


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Execute the fixed V4 data preparation plan in its isolated workspace")
    parser.add_argument("--v4-config", type=Path, required=True)
    arguments = parser.parse_args()
    run_v4_preparation(arguments.v4_config)
