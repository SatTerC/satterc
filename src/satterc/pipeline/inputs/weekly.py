from os import PathLike
from typing import cast

import pandas as pd
import xarray as xr
from hamilton.function_modifiers import check_output_custom, extract_fields, ResolveAt

from ._utils import load_dataset, stack_spatial_dims, DatetimeIndexValidator
from .._hamilton_fixes import FixedResolve


def loaded_weekly_inputs(weekly_inputs_path: str | PathLike) -> xr.Dataset:
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


def stacked_weekly_inputs(loaded_weekly_inputs: xr.Dataset) -> xr.Dataset:
    """Stack spatial dimensions of weekly inputs dataset.

    Parameters
    ----------
    loaded_weekly_inputs : xr.Dataset
        The loaded weekly inputs dataset.

    Returns
    -------
    xr.Dataset
        Dataset with spatial dimensions stacked into 'pixel' dimension.
    """
    return stack_spatial_dims(loaded_weekly_inputs)


@check_output_custom(DatetimeIndexValidator("W"))
def dates_weekly(stacked_weekly_inputs: xr.Dataset) -> pd.DatetimeIndex:
    """Extract weekly datetime index from dataset.

    Parameters
    ----------
    stacked_weekly_inputs : xr.Dataset
        The loaded weekly inputs dataset.

    Returns
    -------
    pd.DatetimeIndex
        DatetimeIndex with weekly frequency.
    """
    return cast(pd.DatetimeIndex, stacked_weekly_inputs.get_index("time"))


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda weekly_inputs_vars: extract_fields(
        {f"{var}_weekly": xr.DataArray for var in weekly_inputs_vars}
    ),
)
def split_weekly_inputs(
    stacked_weekly_inputs: xr.Dataset,
    weekly_inputs_vars: list[str],
) -> dict[str, xr.DataArray]:
    """Unpacks the raw dataset into individual arrays of input variables.

    Spatial coordinates are stacked into a single "pixel" dimension.

    Parameters
    ----------
    stacked_weekly_inputs : xr.Dataset
        The loaded dataset with coordinate reference system information.
    weekly_inputs_vars : list[str]
        List of variable names to extract (resolved from config).

    Returns
    -------
    dict[str, xr.DataArray]
            The data arrays.
    """
    return {
        f"{var}_weekly": stacked_weekly_inputs[var]
        for var in stacked_weekly_inputs.data_vars
    }
