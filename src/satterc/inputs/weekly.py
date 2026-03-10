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

WEEKLY_INPUTS = [
    "co2_ppm",
    "fapar",
    "ppfd_umol_m2_s1",
    "pressure_pa",
    "vpd_pa",
]


WEEKLY_FROM_DAILY = [
    "temperature_celcius",
    "precipitation_mm",
    "soil_moisture",
    # Derived variable
    "aridity_index",
]


def weekly_inputs(weekly_inputs_path: Path) -> xr.Dataset:
    return load_dataset(weekly_inputs_path)


def weekly_inputs_stacked(weekly_inputs: xr.Dataset) -> xr.Dataset:
    return stack_spatial_dims(weekly_inputs)


@extract_fields([f"{var}_weekly" for var in WEEKLY_INPUTS])
def unpack_weekly_inputs(weekly_inputs_stacked: xr.Dataset) -> dict[str, xr.DataArray]:
    """Unpacks the raw dataset into individual arrays of input variables.

    Spatial coordinates are stacked into a single "pixel" dimension.

    Parameters
    ----------
    weekly_inputs_stacked : xr.Dataset
        The loaded dataset with coordinate reference system information.

    Returns
    -------
    dict[str, xr.DataArray]
            The data arrays.
    """
    return {
        f"{var}_weekly": weekly_inputs_stacked[var]
        for var in weekly_inputs_stacked.data_vars
    }


@check_output_custom(DatetimeIndexValidator("W"))
def dates_weekly(weekly_inputs: xr.Dataset) -> pd.DatetimeIndex:
    return cast(pd.DatetimeIndex, weekly_inputs.get_index("time"))


@parameterize_sources(
    **{f"{var}_weekly": {"var_daily": f"{var}_daily"} for var in WEEKLY_FROM_DAILY}
)
def aggregate_daily_to_weekly(var_daily: xr.DataArray) -> xr.DataArray:
    """Resamples daily xarray data to weekly mean."""
    return var_daily.resample(time="1W").mean()
