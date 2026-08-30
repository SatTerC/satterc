"""The temporal resolutions a satterc pipeline speaks in.

conduit infers nothing from a node's name: an input's frequency is validated
only where a consumer declares it. These are satterc's declarations — the pandas
offsets behind its ``_daily`` / ``_weekly`` / ``_monthly`` node-name suffixes.
`satterc.models` declares them as `Freq` contracts, `satterc.scaffold.config_gen`
writes them into the ``[[resample]]`` entries of a generated config, and
`satterc.scaffold.data_gen` resamples synthetic data onto them.

The offsets are unanchored, so ``Freq("7D")`` constrains the spacing only: a
weekly series is accepted whichever weekday it starts on.
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

#: The offset behind each input-section label, keyed by the label conduit uses.
BY_LABEL: dict[str, Freq] = {
    "daily": DAILY,
    "weekly": WEEKLY,
    "monthly": MONTHLY,
}


def offset(label: str) -> str:
    """Return the pandas offset alias for an input-section ``label``.

    `Freq` carries the offset for contract declaration; this is the same string
    in the form `conduit.transforms.resample` and a ``[[resample]]`` entry want.
    """
    try:
        return str(BY_LABEL[label].freq)
    except KeyError:
        raise ValueError(
            f"Unknown frequency label {label!r}; expected one of "
            f"{', '.join(map(repr, BY_LABEL))}."
        ) from None


def resample_offset(from_label: str, to_label: str) -> str:
    """Return the offset for resampling ``from_label`` onto ``to_label``.

    Resampling only coarsens, so this raises `ValueError` unless ``to_label`` is
    coarser than ``from_label``.
    """
    order = list(BY_LABEL)
    for label in (from_label, to_label):
        if label not in order:
            raise ValueError(
                f"Unknown frequency label {label!r}; expected one of "
                f"{', '.join(map(repr, order))}."
            )
    if order.index(to_label) <= order.index(from_label):
        raise ValueError(
            f"Cannot resample {from_label!r} to {to_label!r}: resampling coarsens, "
            f"so the target must be coarser than the source."
        )
    return offset(to_label)


def time_index(da: xr.DataArray, what: str) -> pd.DatetimeIndex:
    """Return ``da``'s time coordinate as a `pandas.DatetimeIndex`.

    The models that wrap a sequential algorithm (SPLASH, SGAM, RothC) need the
    calendar as well as the values. The time dimension is detected from the data
    rather than assumed to be called ``time``.

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
