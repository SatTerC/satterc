from hamilton.function_modifiers import extract_fields, parameterize_sources
import xarray as xr


WEEKLY_INPUT_VARIABLES = [
    "co2_ppm",
    "fapar",
    "ppfd_umol_m2_s1",
    "pressure_pa",
    "vpd_pa",
]


@extract_fields([f"{var}_weekly" for var in WEEKLY_INPUT_VARIABLES])
def weekly_inputs(weekly_inputs_dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """Unpacks the raw dataset into individual arrays of input variables.

    Spatial coordinates are stacked into a single "pixel" dimension.

    Parameters
    ----------
    weekly_inputs : xr.Dataset
        The loaded dataset with coordinate reference system information.

    Returns
    -------
    dict[str, xr.DataArray]
            The data arrays.
    """
    return {
        f"{var}_weekly": weekly_inputs_dataset[var]
        for var in weekly_inputs_dataset.data_vars
    }


WEEKLY_FROM_DAILY = [
    "temperature_celcius",
    "precipitation_mm",
    "soil_moisture",
    # Derived variable
    "aridity_index",
]


@parameterize_sources(
    **{f"{v}_weekly": {"daily_da": f"{v}_daily"} for v in WEEKLY_FROM_DAILY}
)
def aggregate_daily_to_weekly(daily_da: xr.DataArray) -> xr.DataArray:
    """Resamples daily xarray data to weekly mean."""
    return daily_da.resample(time="1W").mean()
