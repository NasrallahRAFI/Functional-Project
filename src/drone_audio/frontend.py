"""Legacy front-end import shim; new code should use dronedet.features."""

import warnings

warnings.warn(
    "drone_audio.frontend is deprecated; use dronedet.features for the current frontend "
    "or dronedet.legacy_frontend only for historical outputs",
    DeprecationWarning,
    stacklevel=2,
)
from dronedet.legacy_frontend import *
