"""Transient Z0 mechanical flags with no retained RMS/spectral descriptors."""

from __future__ import annotations

import numpy as np

from dronedet.config import FrontEndConfig
from dronedet.features import (
    clip_flags,
    fix_length,
    load_wave,
    mel_power,
    power_quality_flags,
)


def mechanical_flags(standardized_wave: np.ndarray, config: FrontEndConfig) -> dict[str, bool | int | None]:
    """Compute only the closed mechanical booleans permitted by the plan.

    Temporary RMS, zero-fraction and centre-mel-power values are discarded
    before returning. No retained descriptor, plot, or cache is produced.
    """
    wave = np.asarray(standardized_wave, dtype=np.float32)
    if wave.ndim != 1 or wave.size == 0 or not np.isfinite(wave).all():
        raise ValueError("mechanical pass requires a non-empty finite standardized mono waveform")

    v4_flags = clip_flags(wave, config)
    exact_zero = bool(np.all(wave == 0))
    if exact_zero:
        zero_fraction = 1.0
        dc_mean = 0.0
        clipped_fraction = 0.0
    else:
        # These scalar intermediates exist only for fixed boolean decisions.
        zero_fraction = float(np.mean(wave == 0))
        dc_mean = float(np.mean(wave, dtype=np.float64))
        clipped_fraction = float(np.mean(np.abs(wave) >= 0.999))
    partial_zero = bool(not exact_zero and zero_fraction > 0.5)
    clipped_flag = bool(clipped_fraction > 0.01)
    dc_flag = bool(abs(dc_mean) > 0.01)

    fixed, length_status = fix_length(wave, config)
    if not fixed.size:
        return {
            "exact_zero": exact_zero,
            "silent": bool(v4_flags["silent"]),
            "near_silent": bool(v4_flags["near_silent"]),
            "partial_zero": partial_zero,
            "clipped_flag": clipped_flag,
            "dc_flag": dc_flag,
            "crop_all_zero": None,
            "stats_eligible": None,
            "too_short": True,
            "standardized_samples": int(wave.size),
        }

    power = mel_power(fixed, config)
    power_flags = power_quality_flags(
        power,
        config,
        silent=bool(v4_flags["silent"]),
        near_silent=bool(v4_flags["near_silent"]),
    )
    crop_all_zero = bool(power_flags["crop_all_zero"])
    stats_eligible = bool(not (v4_flags["silent"] or v4_flags["near_silent"] or crop_all_zero))
    del power, fixed, zero_fraction, dc_mean, clipped_fraction, length_status
    return {
        "exact_zero": exact_zero,
        "silent": bool(v4_flags["silent"]),
        "near_silent": bool(v4_flags["near_silent"]),
        "partial_zero": partial_zero,
        "clipped_flag": clipped_flag,
        "dc_flag": dc_flag,
        "crop_all_zero": crop_all_zero,
        "stats_eligible": stats_eligible,
        "too_short": False,
        "standardized_samples": int(wave.size),
    }


def mechanical_flags_for_path(path: str, config: FrontEndConfig) -> dict[str, bool | int | None]:
    """Read one source waveform, standardize it in memory, and discard it."""
    standardized = load_wave(path, config)
    try:
        return mechanical_flags(standardized, config)
    finally:
        del standardized
