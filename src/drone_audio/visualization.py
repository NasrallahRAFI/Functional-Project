"""Legacy plotting import shim."""
import warnings
warnings.warn("drone_audio.visualization is deprecated; use dronedet.features", DeprecationWarning, stacklevel=2)
from dronedet.visualization import *  # noqa: F401,F403
