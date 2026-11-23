"""Legacy resampling shim retaining the older sizing-argument signature."""
from __future__ import annotations
import numpy as np
import warnings

warnings.warn("drone_audio.resample is deprecated; use dronedet.features", DeprecationWarning, stacklevel=2)
from dronedet.resample import (fmax_below_passband, measure_minus_1db_point,
                               measure_swept_passband, resample as _resample,
                               resample_stream)

def resample(audio: np.ndarray, sample_rate_in: int, sample_rate_out: int,
             half_width: int | None = None, block_size: int | None = None, *,
             quality: str = "HQ") -> np.ndarray:
    if half_width is not None:
        warnings.warn("half_width is ignored by the replacement resampler", UserWarning, stacklevel=2)
    if block_size is not None:
        warnings.warn("block_size is ignored by the replacement resampler", UserWarning, stacklevel=2)
    return _resample(audio, sample_rate_in, sample_rate_out, quality=quality)

__all__ = ["resample", "resample_stream", "measure_minus_1db_point", "measure_swept_passband", "fmax_below_passband"]
