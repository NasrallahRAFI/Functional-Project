"""Compatibility imports; new front-end code lives in :mod:`dronedet.features`."""
import warnings
warnings.warn("drone_audio is deprecated; use dronedet.features", DeprecationWarning, stacklevel=2)
from dronedet.legacy_frontend import FrontEnd, FrontEndConfig, frontend_hash, load_frontend_config
from .resample import fmax_below_passband, measure_minus_1db_point, resample, resample_stream

__all__ = ["FrontEnd", "FrontEndConfig", "frontend_hash", "load_frontend_config", "fmax_below_passband", "measure_minus_1db_point", "resample", "resample_stream"]
