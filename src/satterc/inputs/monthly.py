from pathlib import Path
from typing import cast

from hamilton.function_modifiers import (
    check_output_custom,
    extract_fields,
    parameterize_sources,
)
import pandas as pd
import xarray as xr

from ._utils import load_dataset, stack_spatial_dims, DatetimeIndexValidator

MONTHLY_INPUTS = ["plant_cover", "dpm_rpm_ratio", "farmyard_manure_input"]

MONTHLY_FROM_DAILY = [
    # Input data
    "temperature_celcius",
    "precipitation_mm",
    # Splash output
    "actual_evapotranspiration",
]

MONTHLY_FROM_WEEKLY = [
    # Sgam output
    "litter_to_soil",
]


def monthly_inputs(monthly_inputs_path: Path) -> xr.Dataset:
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


@extract_fields([f"{var}_monthly" for var in MONTHLY_INPUTS])
def unpack_monthly_inputs(
    monthly_inputs_stacked: xr.Dataset,
) -> dict[str, xr.DataArray]:
    """Unpacks the raw dataset into individual arrays of input variables.

    Spatial coordinates are stacked into a single "pixel" dimension.

    Parameters
    ----------
    monthly_inputs_stacked : xr.Dataset
        The loaded dataset with coordinate reference system information.

    Returns
    -------
    dict[str, xr.DataArray]
            The data arrays.
    """
    return {
        f"{var}_monthly": monthly_inputs_stacked[var]
        for var in monthly_inputs_stacked.data_vars
    }


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


@parameterize_sources(
    **{f"{var}_monthly": {"var_daily": f"{var}_daily"} for var in MONTHLY_FROM_DAILY}
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


@parameterize_sources(
    **{f"{var}_monthly": {"var_weekly": f"{var}_weekly"} for var in MONTHLY_FROM_WEEKLY}
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
