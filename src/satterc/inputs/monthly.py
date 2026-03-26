from os import PathLike
from typing import cast

from hamilton.function_modifiers import check_output_custom
import pandas as pd
import xarray as xr

from ._utils import load_dataset, stack_spatial_dims, DatetimeIndexValidator


def monthly_inputs(monthly_inputs_path: str | PathLike) -> xr.Dataset:
    """Load monthly input dataset from file.

    Parameters
    ----------
    monthly_inputs_path : Path
        Path to the NetCDF or Zarr dataset.

    Returns
    -------
    xr.Dataset
        The loaded dataset.
    """
    return load_dataset(monthly_inputs_path)


def monthly_inputs_stacked(monthly_inputs: xr.Dataset) -> xr.Dataset:
    """Stack spatial dimensions of monthly inputs dataset.

    Parameters
    ----------
    monthly_inputs : xr.Dataset
        The loaded monthly inputs dataset.

    Returns
    -------
    xr.Dataset
        Dataset with spatial dimensions stacked into 'pixel' dimension.
    """
    return stack_spatial_dims(monthly_inputs)


@check_output_custom(DatetimeIndexValidator("ME"))
def dates_monthly(monthly_inputs: xr.Dataset) -> pd.DatetimeIndex:
    """Extract monthly datetime index from dataset.

    Parameters
    ----------
    monthly_inputs : xr.Dataset
        The loaded monthly inputs dataset.

    Returns
    -------
    pd.DatetimeIndex
        DatetimeIndex with monthly frequency.
    """
    return cast(pd.DatetimeIndex, monthly_inputs.get_index("time"))
