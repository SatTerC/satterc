from os import PathLike
from typing import cast

from hamilton.function_modifiers import check_output_custom
import pandas as pd
import xarray as xr

from ._utils import load_dataset, stack_spatial_dims, DatetimeIndexValidator


def weekly_inputs(weekly_inputs_path: str | PathLike) -> xr.Dataset:
    """Load weekly input dataset from file.

    Parameters
    ----------
    weekly_inputs_path : Path
        Path to the NetCDF or Zarr dataset.

    Returns
    -------
    xr.Dataset
        The loaded dataset.
    """
    return load_dataset(weekly_inputs_path)


def weekly_inputs_stacked(weekly_inputs: xr.Dataset) -> xr.Dataset:
    """Stack spatial dimensions of weekly inputs dataset.

    Parameters
    ----------
    weekly_inputs : xr.Dataset
        The loaded weekly inputs dataset.

    Returns
    -------
    xr.Dataset
        Dataset with spatial dimensions stacked into 'pixel' dimension.
    """
    return stack_spatial_dims(weekly_inputs)


@check_output_custom(DatetimeIndexValidator("W-SUN"))
def dates_weekly(weekly_inputs: xr.Dataset) -> pd.DatetimeIndex:
    """Extract weekly datetime index from dataset.

    Parameters
    ----------
    weekly_inputs : xr.Dataset
        The loaded weekly inputs dataset.

    Returns
    -------
    pd.DatetimeIndex
        DatetimeIndex with weekly frequency.
    """
    return cast(pd.DatetimeIndex, weekly_inputs.get_index("time"))
