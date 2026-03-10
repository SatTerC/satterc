from pathlib import Path

from hamilton.function_modifiers import extract_fields, parameterize_sources
import xarray as xr

from ._utils import _load_dataset, _stack_spatial_dims

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
    return _load_dataset(monthly_inputs_path)


def monthly_inputs_stacked(monthly_inputs: xr.Dataset) -> xr.Dataset:
    return _stack_spatial_dims(monthly_inputs)


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


@parameterize_sources(
    **{f"{var}_monthly": {"var_daily": f"{var}_daily"} for var in MONTHLY_FROM_DAILY}
)
def aggregate_daily_to_monthly(var_daily: xr.DataArray) -> xr.DataArray:
    return var_daily.resample(time="1ME").mean()


@parameterize_sources(
    **{f"{var}_monthly": {"var_weekly": f"{var}_weekly"} for var in MONTHLY_FROM_WEEKLY}
)
def aggregate_weekly_to_monthly(var_weekly: xr.DataArray) -> xr.DataArray:
    return var_weekly.resample(time="1ME").mean()
