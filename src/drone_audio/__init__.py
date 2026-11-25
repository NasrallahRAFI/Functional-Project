"""Deprecated compatibility namespace for historical ``drone_audio`` imports."""

import warnings

warnings.warn(
    "drone_audio is deprecated; use dronedet.features for the current frontend "
    "and dronedet.legacy_frontend only when preserving historical behavior",
    DeprecationWarning,
    stacklevel=2,
)
from dronedet.legacy_frontend import (
    FrontEnd,
    FrontEndConfig,
    frontend_hash,
    load_frontend_config,
)

from .resample import (
    fmax_below_passband,
    measure_minus_1db_point,
    resample,
    resample_stream,
)

__all__ = [
    "FrontEnd",
    "FrontEndConfig",
    "fmax_below_passband",
    "frontend_hash",
    "load_frontend_config",
    "measure_minus_1db_point",
    "resample",
    "resample_stream",
]
