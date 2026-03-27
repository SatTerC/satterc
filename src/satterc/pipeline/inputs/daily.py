from os import PathLike
from typing import cast

import pandas as pd
import xarray as xr
from hamilton.function_modifiers import check_output_custom, extract_fields, ResolveAt

from ._utils import load_dataset, stack_spatial_dims, DatetimeIndexValidator
from .._hamilton_fixes import FixedResolve


def loaded_daily_inputs(daily_inputs_path: str | PathLike) -> xr.Dataset:
    """Load daily input dataset from file.

    Parameters
    ----------
    daily_inputs_path : Path
        Path to the NetCDF or Zarr dataset.

    Returns
    -------
    xr.Dataset
        The loaded dataset.
    """
    return load_dataset(daily_inputs_path)


def stacked_daily_inputs(loaded_daily_inputs: xr.Dataset) -> xr.Dataset:
    """Stack spatial dimensions of daily inputs dataset.

    Parameters
    ----------
    loaded_daily_inputs : xr.Dataset
        The loaded daily inputs dataset.

    Returns
    -------
    xr.Dataset
        Dataset with spatial dimensions stacked into 'pixel' dimension.
    """
    return stack_spatial_dims(loaded_daily_inputs)


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda daily_inputs_vars: extract_fields(
        {f"{var}_daily": xr.DataArray for var in daily_inputs_vars}
    ),
)
def split_daily_inputs(
    stacked_daily_inputs: xr.Dataset,
    daily_inputs_vars: list[str],
) -> dict[str, xr.DataArray]:
    """Unpacks the stacked daily inputs dataset into individual arrays of input variables.

    Parameters
    ----------
    stacked_daily_inputs : xr.Dataset
        The loaded dataset with coordinate reference system information.
    daily_inputs_vars : list[str]
        List of variable names to extract (resolved from config).

    Returns
    -------
    dict[str, xr.DataArray]
            The data arrays.
    """
    return {
        f"{var}_daily": stacked_daily_inputs[var]
        for var in stacked_daily_inputs.data_vars
    }


@check_output_custom(DatetimeIndexValidator("D"))
def dates_daily(loaded_daily_inputs: xr.Dataset) -> pd.DatetimeIndex:
    """Extract daily datetime index from dataset.

    Parameters
    ----------
    loaded_daily_inputs : xr.Dataset
        The loaded daily inputs dataset.

    Returns
    -------
    pd.DatetimeIndex
        DatetimeIndex with daily frequency.
    """
    return cast(pd.DatetimeIndex, loaded_daily_inputs.get_index("time"))
