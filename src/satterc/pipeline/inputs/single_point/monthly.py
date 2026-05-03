from os import PathLike
from typing import cast

import pandas as pd
import xarray as xr
from hamilton.function_modifiers import check_output_custom, extract_fields, ResolveAt

from .._utils import stack_if_spatial, DatetimeIndexValidator
from ..._hamilton_fixes import FixedResolve, NoOpDecorator
from ._utils import load_timeseries


def loaded_monthly_inputs(monthly_inputs_path: str | PathLike) -> xr.Dataset:
    """Load single-point monthly inputs from a CSV or Parquet file.

    Parameters
    ----------
    monthly_inputs_path : str | PathLike
        Path to a .csv or .parquet file with a datetime index and one column
        per variable. A 'pixel' dimension of size 1 is added automatically.

    Returns
    -------
    xr.Dataset
        Dataset with dims (time, pixel).
    """
    return load_timeseries(monthly_inputs_path)


def stacked_monthly_inputs(loaded_monthly_inputs: xr.Dataset) -> xr.Dataset:
    """Pass through: single-point data already has a 'pixel' dimension."""
    return stack_if_spatial(loaded_monthly_inputs)


@check_output_custom(DatetimeIndexValidator("ME"))
def dates_monthly(stacked_monthly_inputs: xr.Dataset) -> pd.DatetimeIndex:
    """Extract monthly datetime index from the stacked dataset."""
    return cast(pd.DatetimeIndex, stacked_monthly_inputs.get_index("time"))


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda monthly_inputs_vars: (
        extract_fields({f"{var}_monthly": xr.DataArray for var in monthly_inputs_vars})
        if monthly_inputs_vars
        else NoOpDecorator()
    ),
)
def split_monthly_inputs(
    stacked_monthly_inputs: xr.Dataset,
    monthly_inputs_vars: list[str],
) -> dict[str, xr.DataArray]:
    """Unpack the stacked monthly inputs dataset into individual DataArrays."""
    return {
        f"{var}_monthly": stacked_monthly_inputs[var]
        for var in stacked_monthly_inputs.data_vars
    }
