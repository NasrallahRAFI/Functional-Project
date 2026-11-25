"""Legacy DSP import shim."""

import warnings

warnings.warn(
    "drone_audio.dsp is deprecated; use dronedet.dsp for reference functions; "
    "production feature extraction uses dronedet.features",
    DeprecationWarning,
    stacklevel=2,
)
from dronedet.dsp import *
