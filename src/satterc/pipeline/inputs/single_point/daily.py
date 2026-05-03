from os import PathLike
from typing import cast

import pandas as pd
import xarray as xr
from hamilton.function_modifiers import check_output_custom, extract_fields, ResolveAt

from .._utils import stack_if_spatial, DatetimeIndexValidator
from ..._hamilton_fixes import FixedResolve, NoOpDecorator
from ._utils import load_timeseries


def loaded_daily_inputs(daily_inputs_path: str | PathLike) -> xr.Dataset:
    """Load single-point daily inputs from a CSV or Parquet file.

    Parameters
    ----------
    daily_inputs_path : str | PathLike
        Path to a .csv or .parquet file with a datetime index and one column
        per variable. A 'pixel' dimension of size 1 is added automatically.

    Returns
    -------
    xr.Dataset
        Dataset with dims (time, pixel).
    """
    return load_timeseries(daily_inputs_path)


def stacked_daily_inputs(loaded_daily_inputs: xr.Dataset) -> xr.Dataset:
    """Pass through: single-point data already has a 'pixel' dimension."""
    return stack_if_spatial(loaded_daily_inputs)


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda daily_inputs_vars: (
        extract_fields({f"{var}_daily": xr.DataArray for var in daily_inputs_vars})
        if daily_inputs_vars
        else NoOpDecorator()
    ),
)
def split_daily_inputs(
    stacked_daily_inputs: xr.Dataset,
    daily_inputs_vars: list[str],
) -> dict[str, xr.DataArray]:
    """Unpack the stacked daily inputs dataset into individual DataArrays."""
    return {
        f"{var}_daily": stacked_daily_inputs[var]
        for var in stacked_daily_inputs.data_vars
    }


@check_output_custom(DatetimeIndexValidator("D"))
def dates_daily(loaded_daily_inputs: xr.Dataset) -> pd.DatetimeIndex:
    """Extract daily datetime index from the loaded dataset."""
    return cast(pd.DatetimeIndex, loaded_daily_inputs.get_index("time"))
