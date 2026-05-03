from os import PathLike
from typing import cast

import pandas as pd
import xarray as xr
from hamilton.function_modifiers import check_output_custom, extract_fields, ResolveAt

from .._utils import stack_if_spatial, DatetimeIndexValidator
from ..._hamilton_fixes import FixedResolve, NoOpDecorator
from ._utils import load_timeseries


def loaded_weekly_inputs(weekly_inputs_path: str | PathLike) -> xr.Dataset:
    """Load single-point weekly inputs from a CSV or Parquet file.

    Parameters
    ----------
    weekly_inputs_path : str | PathLike
        Path to a .csv or .parquet file with a datetime index and one column
        per variable. A 'pixel' dimension of size 1 is added automatically.

    Returns
    -------
    xr.Dataset
        Dataset with dims (time, pixel).
    """
    return load_timeseries(weekly_inputs_path)


def stacked_weekly_inputs(loaded_weekly_inputs: xr.Dataset) -> xr.Dataset:
    """Pass through: single-point data already has a 'pixel' dimension."""
    return stack_if_spatial(loaded_weekly_inputs)


@check_output_custom(DatetimeIndexValidator("W"))
def dates_weekly(stacked_weekly_inputs: xr.Dataset) -> pd.DatetimeIndex:
    """Extract weekly datetime index from the stacked dataset."""
    return cast(pd.DatetimeIndex, stacked_weekly_inputs.get_index("time"))


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda weekly_inputs_vars: (
        extract_fields({f"{var}_weekly": xr.DataArray for var in weekly_inputs_vars})
        if weekly_inputs_vars
        else NoOpDecorator()
    ),
)
def split_weekly_inputs(
    stacked_weekly_inputs: xr.Dataset,
    weekly_inputs_vars: list[str],
) -> dict[str, xr.DataArray]:
    """Unpack the stacked weekly inputs dataset into individual DataArrays."""
    return {
        f"{var}_weekly": stacked_weekly_inputs[var]
        for var in stacked_weekly_inputs.data_vars
    }
