"""Legacy normalization import shim for pre-contract training helpers."""

import warnings

warnings.warn(
    "drone_audio.normalization is deprecated; use dronedet.features.Normalizer "
    "for global statistics and dronedet.normalization for PCEN/gain operations",
    DeprecationWarning,
    stacklevel=2,
)
from dronedet.legacy_normalization import *
