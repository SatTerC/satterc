"""The time axis: the resolutions a pipeline speaks in, and how to read one.

conduit deliberately infers nothing from a node's name: an input's frequency is
validated only where a consumer *declares* it. These are satterc's declarations —
the pandas offsets behind its ``_daily`` / ``_weekly`` / ``_monthly`` node-name
suffixes.

They live at the package root because three subpackages have to agree on them,
not just the models. `satterc.models` declares them as `Freq` contracts,
`satterc.scaffold.config_gen` writes them into the ``[[resample]]`` entries of a
generated config, and `satterc.scaffold.data_gen` resamples synthetic data onto
them. Those three were previously three separate spellings of ``"7D"`` and
``"1ME"``, kept in step by a comment; changing the weekly convention meant
finding all three, and missing one surfaced as a contract mismatch at runtime.

`time_index` sits here rather than with the models because it is the other half
of the same subject — the models declare a `Freq` contract, and this is how they
get the axis that contract describes.

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

    Coarsening only: the target's own offset is what the resample lands on, so
    this exists to reject the nonsensical direction rather than to compute
    anything. Ordered by `BY_LABEL`, which is coarsest-last.
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
