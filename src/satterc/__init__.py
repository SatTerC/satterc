"""Satellite to Terrestrial Carbon."""

import warnings

from ._version import __version__
from .config import IOSpec, ParsedConfig, ResampleSpec, load_config
from .dag.driver import build_driver
from .io import get_final_vars, get_outputs, load_inputs, save_outputs

__all__ = [
    "IOSpec",
    "ParsedConfig",
    "ResampleSpec",
    "__version__",
    "build_driver",
    "get_final_vars",
    "get_outputs",
    "load_config",
    "load_inputs",
    "save_outputs",
]

# Suppress known pyrealm warnings that are harmless but noisy:
# 1. np.sqrt(where=...) without out= — pyrealm backfills NaN values immediately after,
#    so the uninitialized memory is never used. Fixed in pyrealm upstream pending.
warnings.filterwarnings(
    "ignore",
    message=".*'where' used without 'out'.*",
    category=UserWarning,
)
# 2. Pyrealm 2.0.0 phi0 default change — informational only, we explicitly set
#    method_kphio so the default does not affect our results.
warnings.filterwarnings(
    "ignore",
    message=".*Pyrealm 2\\.0\\.0 uses a new default.*",
    category=UserWarning,
)
# 3. ExperimentalFeatureWarning — we knowingly use QuantumYieldSandoval (method_kphio
#    = "sandoval") and accept the experimental API risk.
warnings.filterwarnings(
    "ignore",
    category=Warning,
    module="pyrealm",
)
