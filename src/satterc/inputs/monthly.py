from hamilton.function_modifiers import extract_fields, parameterize_sources
import xarray as xr


MONTHLY_INPUT_VARIABLES = ["plant_cover", "dpm_rpm_ratio", "farmyard_manure_input"]


@extract_fields([f"{var}_monthly" for var in MONTHLY_INPUT_VARIABLES])
def monthly_inputs(monthly_inputs_dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """Unpacks the raw dataset into individual arrays of input variables.

    Spatial coordinates are stacked into a single "pixel" dimension.

    Parameters
    ----------
    monthly_inputs : xr.Dataset
        The loaded dataset with coordinate reference system information.

    Returns
    -------
    dict[str, xr.DataArray]
            The data arrays.
    """
    return {
        f"{var}_monthly": monthly_inputs_dataset[var]
        for var in monthly_inputs_dataset.data_vars
    }


MONTHLY_FROM_DAILY = [
    # Input data
    "temperature_celcius",
    "precipitation_mm",
    # Splash output
    "actual_evapotranspiration",
]


@parameterize_sources(
    **{f"{v}_monthly": {"daily_da": f"{v}_daily"} for v in MONTHLY_FROM_DAILY}
)
def aggregate_daily_to_monthly(daily_da: xr.DataArray) -> xr.DataArray:
    return daily_da.resample(time="1ME").mean()


MONTHLY_FROM_WEEKLY = [
    # Sgam output
    "litter_to_soil",
]


@parameterize_sources(
    **{f"{v}_monthly": {"weekly_da": f"{v}_weekly"} for v in MONTHLY_FROM_WEEKLY}
)
def aggregate_weekly_to_monthly(weekly_da: xr.DataArray) -> xr.DataArray:
    return weekly_da.resample(time="1ME").mean()
