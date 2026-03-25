from os import PathLike
from typing import List, cast

from hamilton.function_modifiers import check_output_custom
import pandas as pd
import xarray as xr

from ._utils import load_dataset, stack_spatial_dims, DatetimeIndexValidator
from .._hamilton_utils import (
    make_extract_fields_resolver,
    make_parameterize_sources_resolver,
)


def daily_inputs(daily_inputs_path: str | PathLike) -> xr.Dataset:
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


def daily_inputs_stacked(daily_inputs: xr.Dataset) -> xr.Dataset:
    """Stack spatial dimensions of daily inputs dataset.

    Parameters
    ----------
    daily_inputs : xr.Dataset
        The loaded daily inputs dataset.

    Returns
    -------
    xr.Dataset
        Dataset with spatial dimensions stacked into 'pixel' dimension.
    """
    return stack_spatial_dims(daily_inputs)


@make_extract_fields_resolver("daily", "_daily")
def unpack_daily_inputs(
    daily_inputs_stacked: xr.Dataset,
    daily: List[str],
) -> dict[str, xr.DataArray]:
    """Unpacks the stacked daily inputs dataset into individual arrays of input variables.

    Parameters
    ----------
    daily_inputs_stacked : xr.Dataset
        The loaded dataset with coordinate reference system information.
    daily : List[str]
        List of variable names to extract (resolved from config).

    Returns
    -------
    dict[str, xr.DataArray]
            The data arrays.
    """
    return {
        f"{var}_daily": daily_inputs_stacked[var]
        for var in daily_inputs_stacked.data_vars
    }


@check_output_custom(DatetimeIndexValidator("D"))
def dates_daily(daily_inputs: xr.Dataset) -> pd.DatetimeIndex:
    """Extract daily datetime index from dataset.

    Parameters
    ----------
    daily_inputs : xr.Dataset
        The loaded daily inputs dataset.

    Returns
    -------
    pd.DatetimeIndex
        DatetimeIndex with daily frequency.
    """
    return cast(pd.DatetimeIndex, daily_inputs.get_index("time"))
