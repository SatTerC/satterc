from os import PathLike
from typing import cast

from hamilton.function_modifiers import check_output_custom
import pandas as pd
import xarray as xr

from ._utils import load_dataset, stack_spatial_dims, DatetimeIndexValidator


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
