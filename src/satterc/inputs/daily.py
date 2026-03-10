from hamilton.function_modifiers import extract_fields
import xarray as xr


# TODO: figure out how to make this more flexible.
# Not sure if user-provided list is actually a good idea.
DAILY_INPUT_VARIABLES = [
    "precipitation_mm",
    "sunshine_fraction",
    "temperature_celcius",
]


@extract_fields([f"{var}_daily" for var in DAILY_INPUT_VARIABLES])
def daily_inputs(daily_inputs_dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """Unpacks the raw dataset into individual arrays of input variables.

    Spatial coordinates are stacked into a single "pixel" dimension.

    Parameters
    ----------
    daily_inputs : xr.Dataset
        The loaded dataset with coordinate reference system information.

    Returns
    -------
    dict[str, xr.DataArray]
            The data arrays.
    """
    return {
        f"{var}_daily": daily_inputs_dataset[var]
        for var in daily_inputs_dataset.data_vars
    }
