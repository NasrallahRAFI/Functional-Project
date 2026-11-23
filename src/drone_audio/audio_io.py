"""Legacy audio I/O import shim."""
import warnings
warnings.warn("drone_audio.audio_io is deprecated; use dronedet.features", DeprecationWarning, stacklevel=2)
from dronedet.audio_io import *  # noqa: F401,F403
