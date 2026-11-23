"""Legacy front-end import shim; new code should use dronedet.features."""
import warnings
warnings.warn("drone_audio.frontend is deprecated; use dronedet.features", DeprecationWarning, stacklevel=2)
from dronedet.legacy_frontend import *  # noqa: F401,F403
