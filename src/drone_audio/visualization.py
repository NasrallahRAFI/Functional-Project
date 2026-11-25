"""Legacy plotting import shim."""

import warnings

warnings.warn(
    "drone_audio.visualization is deprecated; use dronedet.visualization",
    DeprecationWarning,
    stacklevel=2,
)
from dronedet.visualization import *
