"""Legacy feature import shim; new code should use :mod:`dronedet.features`."""

import warnings

warnings.warn(
    "drone_audio.features is deprecated; use dronedet.features for the current API; "
    "legacy mel_tabular_features remains available only to preserve old descriptors",
    DeprecationWarning,
    stacklevel=2,
)
from dronedet.legacy_features import *
