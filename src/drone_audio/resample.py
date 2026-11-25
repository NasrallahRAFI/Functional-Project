"""Legacy resampling shim retaining the older sizing-argument signature."""

from __future__ import annotations

import warnings

import numpy as np

warnings.warn(
    "drone_audio.resample is deprecated; use dronedet.resample",
    DeprecationWarning,
    stacklevel=2,
)
from dronedet.resample import (
    fmax_below_passband,
    measure_minus_1db_point,
    measure_swept_passband,
    resample_stream,
)
from dronedet.resample import (
    resample as _resample,
)


def resample(
    audio: np.ndarray,
    sample_rate_in: int,
    sample_rate_out: int,
    half_width: int | None = None,
    block_size: int | None = None,
    *,
    quality: str = "HQ",
) -> np.ndarray:
    if half_width is not None:
        warnings.warn(
            "half_width is ignored by the replacement resampler",
            UserWarning,
            stacklevel=2,
        )
    if block_size is not None:
        warnings.warn(
            "block_size is ignored by the replacement resampler",
            UserWarning,
            stacklevel=2,
        )
    return _resample(audio, sample_rate_in, sample_rate_out, quality=quality)


__all__ = [
    "fmax_below_passband",
    "measure_minus_1db_point",
    "measure_swept_passband",
    "resample",
    "resample_stream",
]
