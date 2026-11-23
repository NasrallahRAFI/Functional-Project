"""Legacy feature import shim; new code should use dronedet.features."""
import warnings
warnings.warn("drone_audio.features is deprecated; use dronedet.features", DeprecationWarning, stacklevel=2)
from dronedet.legacy_features import *  # noqa: F401,F403
