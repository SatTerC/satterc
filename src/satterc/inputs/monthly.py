from os import PathLike
from typing import List, cast

from hamilton.function_modifiers import (
    check_output_custom,
    parameterize_sources,
    resolve,
    ResolveAt,
)
import pandas as pd
import xarray as xr

from ._utils import load_dataset, stack_spatial_dims, DatetimeIndexValidator
from .._hamilton_utils import LazyExtractFields, NoOpDecorator


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


@resolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda monthly_from_daily: (
        parameterize_sources(
            **{
                f"{var}_monthly": {"var_daily": f"{var}_daily"}
                for var in monthly_from_daily
            }
        )
        if monthly_from_daily
        else NoOpDecorator()
    ),
)
def aggregate_daily_to_monthly(var_daily: xr.DataArray) -> xr.DataArray:
    """Resample daily data to monthly mean.

    Parameters
    ----------
    var_daily : xr.DataArray
        Daily input data.

    Returns
    -------
    xr.DataArray
        Monthly averaged data.
    """
    return var_daily.resample(time="1ME").mean()


@resolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda monthly_from_weekly: (
        parameterize_sources(
            **{
                f"{var}_monthly": {"var_weekly": f"{var}_weekly"}
                for var in monthly_from_weekly
            }
        )
        if monthly_from_weekly
        else NoOpDecorator()
    ),
)
def aggregate_weekly_to_monthly(var_weekly: xr.DataArray) -> xr.DataArray:
    """Resample weekly data to monthly mean.

    Parameters
    ----------
    var_weekly : xr.DataArray
        Weekly input data.

    Returns
    -------
    xr.DataArray
        Monthly averaged data.
    """
    return var_weekly.resample(time="1ME").mean()
