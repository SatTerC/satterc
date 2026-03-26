import xarray as xr

from ._utils import _save_dataset


def daily_outputs_stacked(
    actual_evapotranspiration_daily: xr.DataArray,
    soil_moisture_daily: xr.DataArray,
    runoff_daily: xr.DataArray,
) -> xr.Dataset:
    """Merge daily output data arrays into a single dataset.

    Parameters
    ----------
    actual_evapotranspiration_daily : xr.DataArray
        Daily actual evapotranspiration.
    soil_moisture_daily : xr.DataArray
        Daily soil moisture.
    runoff_daily : xr.DataArray
        Daily runoff.

    Returns
    -------
    xr.Dataset
        Merged dataset with stacked spatial dimensions.
    """
    return xr.merge(
        [actual_evapotranspiration_daily, soil_moisture_daily, runoff_daily]
    )


def daily_outputs(daily_outputs_stacked: xr.Dataset) -> xr.Dataset:
    """Unstack spatial dimensions from daily outputs.

    Parameters
    ----------
    daily_outputs_stacked : xr.Dataset
        Daily outputs with stacked spatial dimensions.

    Returns
    -------
    xr.Dataset
        Daily outputs with original spatial dimensions restored.
    """
    return daily_outputs_stacked.unstack("pixel")


def saved_daily_outputs(daily_outputs: xr.Dataset, daily_outputs_path: str) -> None:
    """Save daily outputs to file.

    Parameters
    ----------
    daily_outputs : xr.Dataset
        Daily outputs dataset.
    daily_outputs_path : str
        Path to save the dataset.
    """
    _save_dataset(daily_outputs, daily_outputs_path)
