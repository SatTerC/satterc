"""Frequency conventions and time-axis extraction for the model wrappers.

conduit deliberately infers nothing from a node's name: an input's frequency is
validated only where a consumer *declares* it. These are satterc's declarations
— the pandas offsets behind its ``_daily`` / ``_weekly`` / ``_monthly`` node-name
suffixes — collected here so the four model modules agree on them.

The offsets are unanchored on purpose. ``Freq("7D")`` constrains the *spacing*
only, so a weekly series is accepted whichever weekday it starts on; pinning the
phase (``"W-SUN"``) would reject a perfectly good pipeline whose resample happens
to land on a Wednesday.
"""

import pandas as pd
import xarray as xr
from conduit.io import sole_time_dim
from xarray_annotated.temporal import Freq

#: Daily: one sample per day.
DAILY = Freq("D")

#: Weekly: seven-day spacing, any weekday.
WEEKLY = Freq("7D")

#: Monthly: one sample per calendar month, month-end convention.
MONTHLY = Freq("1ME")


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
