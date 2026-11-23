"""Legacy DSP import shim."""
import warnings
warnings.warn("drone_audio.dsp is deprecated; use dronedet.features", DeprecationWarning, stacklevel=2)
from dronedet.dsp import *  # noqa: F401,F403
