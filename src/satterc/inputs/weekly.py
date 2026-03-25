from os import PathLike
from typing import List, cast

from hamilton.function_modifiers import (
    check_output_custom,
    resolve,
    ResolveAt,
)
import pandas as pd
import xarray as xr

from ._utils import load_dataset, stack_spatial_dims, DatetimeIndexValidator
from .._hamilton_utils import LazyExtractFields, NoOpDecorator


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


@resolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda weekly: LazyExtractFields(
        {f"{var}_weekly": xr.DataArray for var in weekly}
    ),
)
def unpack_weekly_inputs(
    weekly_inputs_stacked: xr.Dataset,
    weekly: List[str],
) -> dict[str, xr.DataArray]:
    """Unpacks the raw dataset into individual arrays of input variables.

    Spatial coordinates are stacked into a single "pixel" dimension.

    Parameters
    ----------
    weekly_inputs_stacked : xr.Dataset
        The loaded dataset with coordinate reference system information.
    weekly : List[str]
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


@resolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda weekly_from_daily: (
        _make_parameterize_sources(
            {
                f"{var}_weekly": {"var_daily": f"{var}_daily"}
                for var in weekly_from_daily
            }
        )
        if weekly_from_daily
        else NoOpDecorator()
    ),
)
def aggregate_daily_to_weekly(var_daily: xr.DataArray) -> xr.DataArray:
    """Resamples daily xarray data to weekly mean."""
    return var_daily.resample(time="1W").mean()


def _make_parameterize_sources(parameterization: dict):
    """Create a parameterize_sources decorator with dynamic parameterization."""
    from hamilton.function_modifiers import parameterize_sources

    return parameterize_sources(**parameterization)
