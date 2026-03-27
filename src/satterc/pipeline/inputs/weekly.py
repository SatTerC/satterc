from os import PathLike
from typing import cast

import pandas as pd
import xarray as xr
from hamilton.function_modifiers import check_output_custom, extract_fields, ResolveAt

from ._utils import load_dataset, stack_spatial_dims, DatetimeIndexValidator
from .._hamilton_fixes import FixedResolve


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
def dates_weekly(weekly_inputs_stacked: xr.Dataset) -> pd.DatetimeIndex:
    """Extract weekly datetime index from dataset.

    Parameters
    ----------
    weekly_inputs_stacked : xr.Dataset
        The loaded weekly inputs dataset.

    Returns
    -------
    pd.DatetimeIndex
        DatetimeIndex with weekly frequency.
    """
    return cast(pd.DatetimeIndex, weekly_inputs_stacked.get_index("time"))


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda weekly_inputs_vars: extract_fields(
        {f"{var}_weekly": xr.DataArray for var in weekly_inputs_vars}
    ),
)
def unpack_weekly_inputs(
    weekly_inputs_stacked: xr.Dataset,
    weekly_inputs_vars: list[str],
) -> dict[str, xr.DataArray]:
    """Unpacks the raw dataset into individual arrays of input variables.

    Spatial coordinates are stacked into a single "pixel" dimension.

    Parameters
    ----------
    weekly_inputs_stacked : xr.Dataset
        The loaded dataset with coordinate reference system information.
    weekly_inputs_vars : List[str]
        List of variable names to extract (resolved from config).

    Returns
    -------
    dict[str, xr.DataArray]
            The data arrays.
    """
    return {
        f"{var}_weekly": weekly_inputs_stacked[var]
        for var in weekly_inputs_stacked.data_vars
    }
