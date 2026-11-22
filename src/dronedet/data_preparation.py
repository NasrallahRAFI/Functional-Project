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
import soxr

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


def measure_resampler_edge(in_sr: int = 48000, out_sr: int = 16000, quality: str = "HQ", duration: float = 3.0) -> dict:
    time = np.arange(int(in_sr * duration)) / in_sr
    def level(freq: float) -> float:
        wave = (0.5 * np.sin(2 * np.pi * freq * time)).astype(np.float32)
        output = soxr.resample(wave, in_sr, out_sr, quality=quality)
        middle = output[len(output) // 4: 3 * len(output) // 4]
        return float(20 * np.log10(np.sqrt(np.mean(middle.astype(np.float64) ** 2)) + 1e-20))
    reference = level(1000.0)
    frequencies = np.arange(5000.0, 8001.0, 25.0)
    response = np.array([level(float(freq)) - reference for freq in frequencies])
    below = np.where(response < -1.0)[0]
    f1db = float(frequencies[below[0]]) if below.size else float(frequencies[-1])
    fmax = min(8000.0, 250.0 * (math.ceil(f1db / 250.0) - 1))
    return {"f1db": f1db, "fmax": fmax, "freqs": frequencies.tolist(), "resp_db": response.tolist(),
            "method": "stepped sine response, 25 Hz increments, HQ, 48 kHz to 16 kHz"}


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
