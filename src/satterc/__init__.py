"""Satellite to Terrestrial Carbon.

satterc supplies the carbon-cycle model modules — SPLASH, the P-model, SGAM and
RothC — for pipelines built with [conduit](https://github.com/NERC-CEH/conduit).
conduit owns the framework: config parsing, the DAG, contract validation, I/O and
execution. Run a pipeline with `conduit.run("config.toml")`, or check one without
computing with `conduit.dry_run`.

satterc registers its four model modules with conduit, so a config addresses one
by section name alone::

    [rothc]
    n_years_spinup = 1

Import the models from `satterc.models` to call their functions directly.
"""

import warnings

from ._version import __version__

__all__ = ["__version__"]

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
