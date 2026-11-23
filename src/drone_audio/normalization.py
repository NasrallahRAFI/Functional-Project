"""Legacy normalizer import shim; new code should use dronedet.features.Normalizer."""
import warnings
warnings.warn("drone_audio.normalization is deprecated; use dronedet.features", DeprecationWarning, stacklevel=2)
from dronedet.legacy_normalization import *  # noqa: F401,F403
