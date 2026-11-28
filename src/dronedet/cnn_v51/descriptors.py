"""V5.1 named waveform descriptors and train-only decile bindings."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable

import librosa
import numpy as np
from scipy.signal import periodogram

from dronedet.config import FrontEndConfig
from dronedet.features import mel_power, to_db

from .artifacts import canonical_json_bytes


def v51_descriptor(audio: np.ndarray, config: FrontEndConfig) -> dict[str, Any]:
    """Compute the V5.1 descriptor contract for one standardized clip."""
    wave = np.asarray(audio, dtype=np.float32)
    if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
        raise ValueError("V5.1 descriptors require finite nonempty standardized mono audio")
    if wave.size < config.n_fft:
        raise ValueError("V5.1 descriptor clip is shorter than n_fft")
    nonzero = wave[wave != 0].astype(np.float64)
    exact_zero = nonzero.size == 0
    rms_dbfs = -240.0 if exact_zero else 20.0 * math.log10(math.sqrt(float(np.mean(nonzero * nonzero))))

    spectrum = librosa.stft(
        y=wave,
        n_fft=config.n_fft,
        hop_length=config.hop,
        win_length=config.win_length,
        window=config.window,
        center=config.center,
        dtype=np.complex64,
    )
    stft_power = np.asarray(np.abs(spectrum).astype(np.float64) ** 2, dtype=np.float64)
    mean_power = np.mean(stft_power, axis=1, dtype=np.float64)
    total_power = float(np.sum(mean_power, dtype=np.float64))
    if exact_zero or total_power == 0.0:
        rolloff_hz = 0.0
    else:
        rolloff_index = int(np.searchsorted(np.cumsum(mean_power, dtype=np.float64), 0.99 * total_power, side="left"))
        rolloff_index = min(rolloff_index, config.n_fft // 2)
        rolloff_hz = float(rolloff_index * config.sr / config.n_fft)

    mel = mel_power(wave, config)
    mel_db = to_db(mel, config).astype(np.float64)
    frame_mean_db = np.mean(mel_db, axis=0, dtype=np.float64)
    noise_floor_db = float(np.quantile(frame_mean_db, 0.10, method="linear"))
    band_index = np.arange(config.n_mels, dtype=np.float64)
    centered_index = band_index - float(np.mean(band_index))
    centered_db = np.mean(mel_db, axis=1, dtype=np.float64)
    denominator = float(np.dot(centered_index, centered_index))
    tilt = float(np.dot(centered_index, centered_db - np.mean(centered_db)) / denominator)

    frequencies, density = periodogram(
        wave.astype(np.float64),
        fs=config.sr,
        window="boxcar",
        detrend=False,
        return_onesided=True,
        scaling="density",
    )
    bin_hz = config.sr / wave.size
    bin_power = density * bin_hz
    band_power = float(np.sum(bin_power[(frequencies >= 100.0) & (frequencies <= 4000.0)], dtype=np.float64))
    band_dbfs = 10.0 * math.log10(band_power) if band_power > 0.0 else -240.0

    del spectrum, stft_power, mean_power, mel, mel_db, frame_mean_db, frequencies, density, bin_power
    return {
        "v51_rms_dbfs_nonzero": float(rms_dbfs),
        "v51_rolloff99_hz": rolloff_hz,
        "v51_noise_floor_db": noise_floor_db,
        "v51_tilt_db_per_band": tilt,
        "band_100_4000_dbfs": float(band_dbfs),
        "all_zero_descriptor": bool(exact_zero),
        "rms_missing": False,
    }


def fit_decile_edges(
    train_rows: Iterable[dict[str, Any]],
    *,
    train_manifest_sha256: str,
) -> dict[str, Any]:
    """Fit linear-quantile RMS/rolloff deciles on unweighted train rows only."""
    rows = [dict(row) for row in train_rows]
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("decile edges require a nonempty train-only descriptor population")
    if len(train_manifest_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in train_manifest_sha256):
        raise ValueError("train manifest hash must be lowercase SHA-256")
    probabilities = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    edges: dict[str, list[float]] = {}
    for descriptor, key in (
        ("v51_rms_dbfs_nonzero", "rms_edges"),
        ("v51_rolloff99_hz", "rolloff_edges"),
    ):
        values = np.asarray([float(row[descriptor]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite train descriptor values for {descriptor}")
        edges[key] = [float(value) for value in np.quantile(values, probabilities, method="linear")]
    payload = {
        "train_manifest_sha256": train_manifest_sha256,
        "quantile_method": "linear",
        "bins": 10,
        **edges,
    }
    payload["edges_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def apply_decile_edges(
    rows: Iterable[dict[str, Any]],
    edge_record: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Assign bins with searchsorted-right, preserving duplicate-edge empties."""
    if edge_record.get("bins") != 10 or edge_record.get("quantile_method") != "linear":
        raise ValueError("unsupported descriptor-decile contract")
    payload = {key: value for key, value in edge_record.items() if key != "edges_sha256"}
    if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != edge_record.get("edges_sha256"):
        raise ValueError("descriptor-decile edge hash mismatch")
    rms_edges = np.asarray(edge_record["rms_edges"], dtype=np.float64)
    rolloff_edges = np.asarray(edge_record["rolloff_edges"], dtype=np.float64)
    if rms_edges.shape != (11,) or rolloff_edges.shape != (11,) or not np.isfinite(rms_edges).all() or not np.isfinite(rolloff_edges).all():
        raise ValueError("each decile edge vector must contain eleven finite values")
    if np.any(np.diff(rms_edges) < 0) or np.any(np.diff(rolloff_edges) < 0):
        raise ValueError("decile edges must be monotonically nondecreasing")

    output = []
    for original in rows:
        row = dict(original)
        for descriptor, edges, target in (
            ("v51_rms_dbfs_nonzero", rms_edges, "rms_decile"),
            ("v51_rolloff99_hz", rolloff_edges, "rolloff_decile"),
        ):
            value = float(row[descriptor])
            if not math.isfinite(value):
                raise ValueError(f"non-finite descriptor in {row.get('clip_id')}: {descriptor}")
            index = int(np.searchsorted(edges[1:-1], value, side="right"))
            row[target] = min(max(index, 0), 9)
        output.append(row)
    return tuple(output)
