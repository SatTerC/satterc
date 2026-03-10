from pathlib import Path
from typing import cast

from hamilton.function_modifiers import check_output_custom, extract_fields
import pandas as pd
import xarray as xr

from ._utils import load_dataset, stack_spatial_dims, DatetimeIndexValidator

# TODO: figure out how to make this more flexible.
# Not sure if user-provided list is actually a good idea.
DAILY_INPUTS = [
    "precipitation_mm",
    "sunshine_fraction",
    "temperature_celcius",
    "lai",
]


def daily_inputs(daily_inputs_path: Path) -> xr.Dataset:
    return load_dataset(daily_inputs_path)


def daily_inputs_stacked(daily_inputs: xr.Dataset) -> xr.Dataset:
    return stack_spatial_dims(daily_inputs)


@extract_fields([f"{var}_daily" for var in DAILY_INPUTS])
def unpack_daily_inputs(daily_inputs_stacked: xr.Dataset) -> dict[str, xr.DataArray]:
    """Unpacks the stacked daily inputs dataset into individual arrays of input variables.

    Parameters
    ----------
    daily_inputs_stacked : xr.Dataset
        The loaded dataset with coordinate reference system information.

    Returns
    -------
    dict[str, xr.DataArray]
            The data arrays.
    """
    return {
        f"{var}_daily": daily_inputs_stacked[var]
        for var in daily_inputs_stacked.data_vars
    }


@check_output_custom(DatetimeIndexValidator("D"))
def dates_daily(daily_inputs: xr.Dataset) -> pd.DatetimeIndex:
    return cast(pd.DatetimeIndex, daily_inputs.get_index("time"))
