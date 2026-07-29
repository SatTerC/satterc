"""Time-axis extraction for the model wrappers.

The `Freq` contracts themselves live in `satterc.frequencies`, because the
scaffolding subpackages need them too; they are re-exported here so a model
module has one import for everything time-related.
"""

import pandas as pd
import xarray as xr
from conduit.io import sole_time_dim

from ..frequencies import DAILY, MONTHLY, WEEKLY

__all__ = ["DAILY", "MONTHLY", "WEEKLY", "time_index"]


def time_index(da: xr.DataArray, what: str) -> pd.DatetimeIndex:
    """Return ``da``'s time coordinate as a `pandas.DatetimeIndex`.

    The models that wrap a sequential algorithm (SPLASH, SGAM, RothC) need the
    calendar as well as the values, and conduit no longer supplies it as a
    separate ``dates_*`` node. The time dimension is detected from the data
    rather than assumed to be called ``time``, via `conduit.io.sole_time_dim`.

    Parameters
    ----------
    da
        A time-bearing array; only its time coordinate is read.
    what
        Human-readable name of ``da``, used in the error raised when it carries
        no time dimension, or more than one.

    Returns
    -------
    pandas.DatetimeIndex
        The time coordinate.
    """
    dim = sole_time_dim(da, what)
    return pd.DatetimeIndex(da.coords[dim].values)
