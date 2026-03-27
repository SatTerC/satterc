from os import PathLike
from typing import cast

import pandas as pd
import xarray as xr
from hamilton.function_modifiers import check_output_custom, extract_fields, ResolveAt

from ._utils import load_dataset, stack_spatial_dims, DatetimeIndexValidator
from .._hamilton_fixes import FixedResolve


def loaded_monthly_inputs(monthly_inputs_path: str | PathLike) -> xr.Dataset:
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


def stacked_monthly_inputs(loaded_monthly_inputs: xr.Dataset) -> xr.Dataset:
    """Stack spatial dimensions of monthly inputs dataset.

    Parameters
    ----------
    loaded_monthly_inputs : xr.Dataset
        The loaded monthly inputs dataset.

    Returns
    -------
    xr.Dataset
        Dataset with spatial dimensions stacked into 'pixel' dimension.
    """
    return stack_spatial_dims(loaded_monthly_inputs)


@check_output_custom(DatetimeIndexValidator("ME"))
def dates_monthly(stacked_monthly_inputs: xr.Dataset) -> pd.DatetimeIndex:
    """Extract monthly datetime index from dataset.

    Parameters
    ----------
    stacked_monthly_inputs : xr.Dataset
        The loaded monthly inputs dataset.

    Returns
    -------
    pd.DatetimeIndex
        DatetimeIndex with monthly frequency.
    """
    return cast(pd.DatetimeIndex, stacked_monthly_inputs.get_index("time"))


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda monthly_inputs_vars: extract_fields(
        {f"{var}_monthly": xr.DataArray for var in monthly_inputs_vars}
    ),
)
def split_monthly_inputs(
    stacked_monthly_inputs: xr.Dataset,
    monthly_inputs_vars: list[str],
) -> dict[str, xr.DataArray]:
    """Unpacks the raw dataset into individual arrays of input variables.

    Spatial coordinates are stacked into a single "pixel" dimension.

    Parameters
    ----------
    stacked_monthly_inputs : xr.Dataset
        The loaded dataset with coordinate reference system information.
    monthly_inputs_vars : list[str]
        List of variable names to extract (resolved from config).

    Returns
    -------
    dict[str, xr.DataArray]
            The data arrays.
    """
    return {
        f"{var}_monthly": stacked_monthly_inputs[var]
        for var in stacked_monthly_inputs.data_vars
    }
