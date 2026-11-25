"""Legacy audio I/O import shim."""

import warnings

warnings.warn(
    "drone_audio.audio_io is deprecated; use dronedet.audio_io",
    DeprecationWarning,
    stacklevel=2,
)
from dronedet.audio_io import *
