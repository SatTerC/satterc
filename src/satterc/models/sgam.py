import numpy as np
import pandas as pd
from numpy.typing import NDArray
import xarray as xr
from sgam import Disturbances, SgamComponent

from ..utils import xarray_io


@xarray_io()
def _disturbances_daily(
    temperature_celcius_daily: NDArray,
    gpp_daily: NDArray,
    lai_daily: NDArray,
) -> NDArray:
    return Disturbances(growing_season_limit=10.0, disturbance_threshold=0.3)(
        temperature_celcius_daily, gpp_daily, lai_daily, aggregate=False
    )


def disturbances_daily(
    temperature_celcius_daily: xr.DataArray,
    gpp_daily: xr.DataArray,
    lai_daily: xr.DataArray,
) -> xr.DataArray:
    return _disturbances_daily(temperature_celcius_daily, gpp_daily, lai_daily)


def disturbances_weekly(disturbances_daily: xr.DataArray) -> xr.DataArray:
    return disturbances_daily.resample(time="W").max()  # or time="7D"


@xarray_io()
def _sgam(
    plant_type: NDArray[np.str_],
    temperature_celcius_weekly: NDArray[np.float64],
    gpp_weekly: NDArray[np.float64],
    soil_moisture_weekly: NDArray[np.float64],
    vpd_pa_weekly: NDArray[np.float64],
    lue_weekly: NDArray[np.float64],
    iwue_weekly: NDArray[np.float64],
    dates_weekly: pd.DatetimeIndex,
    disturbances_weekly: NDArray[np.float64],
    leaf_pool_init: float,
    stem_pool_init: float,
    root_pool_init: float,
) -> dict[str, NDArray]:
    # Week index, from 1-52
    week_of_year = dates_weekly.isocalendar().week

    # TODO: manual loop might benefit from an apply_ufunc or something.
    results_all_pixels = []

    for i in range(len(plant_type)):
        results_i = SgamComponent(plant_type[i])(
            gpp=gpp_weekly[:, i],
            temperature=temperature_celcius_weekly[:, i],
            soil_moisture=soil_moisture_weekly[:, i],
            vpd=vpd_pa_weekly[:, i],
            lue=lue_weekly[:, i],
            iwue=iwue_weekly[:, i],
            week_of_year=week_of_year,
            disturbances=disturbances_weekly[:, i],
            leaf_pool_init=leaf_pool_init,
            stem_pool_init=stem_pool_init,
            root_pool_init=root_pool_init,
        )
        results_all_pixels.append(results_i)

    # Stack list of dicts of 1D arrays (time) into dict of 2D arrays (time, pixel)
    keys = results_all_pixels[0].keys()
    results_stacked = {
        key: np.vstack([d[key] for d in results_all_pixels]) for key in keys
    }

    return results_stacked


def sgam(
    plant_type: xr.DataArray,
    temperature_celcius_weekly: xr.DataArray,
    gpp_weekly: xr.DataArray,
    soil_moisture_weekly: xr.DataArray,
    vpd_pa_weekly: xr.DataArray,
    lue_weekly: xr.DataArray,
    iwue_weekly: xr.DataArray,
    disturbances_weekly: xr.DataArray,
    leaf_pool_init: float,
    stem_pool_init: float,
    root_pool_init: float,
) -> dict[str, xr.DataArray]:
    dates_weekly = temperature_celcius_weekly.get_index("time")
    return _sgam(
        plant_type=plant_type.values,
        temperature_celcius_weekly=temperature_celcius_weekly,
        gpp_weekly=gpp_weekly,
        soil_moisture_weekly=soil_moisture_weekly,
        vpd_pa_weekly=vpd_pa_weekly,
        lue_weekly=lue_weekly,
        iwue_weekly=iwue_weekly,
        dates_weekly=dates_weekly,
        disturbances_weekly=disturbances_weekly,
        leaf_pool_init=leaf_pool_init,
        stem_pool_init=stem_pool_init,
        root_pool_init=root_pool_init,
    )
