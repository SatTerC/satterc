"""
Satterc-compatable interface to PyRealm's 'Splash' model.

This module provides the Splash class, which wraps the SPLASH model
to calculate soil moisture, actual evapotranspiration (AET), and runoff based
on climate inputs.
"""

from hamilton.function_modifiers import unpack_fields
import numpy as np
from numpy.typing import NDArray
from pandas import DatetimeIndex
from xarray import DataArray
import pyrealm.splash.splash
import pyrealm.core.calendar

from ._utils import xarray_io


@xarray_io()
def _splash(
    sunshine_fraction_daily: NDArray[np.float64],
    temperature_celcius_daily: NDArray[np.float64],
    precipitation_mm_daily: NDArray[np.float64],
    elevation: NDArray[np.float64],
    latitude: NDArray[np.float64],
    max_soil_moisture: NDArray[np.float64],
    soil_moisture_init_max_iter: int,
    soil_moisture_init_max_diff: float,
    dates_daily: DatetimeIndex,
) -> tuple[NDArray, NDArray, NDArray]:
    calendar = pyrealm.core.calendar.Calendar(dates_daily.values)

    model = pyrealm.splash.splash.SplashModel(
        lat=latitude[0],
        elv=elevation[0],
        sf=sunshine_fraction_daily,
        tc=temperature_celcius_daily,
        pn=precipitation_mm_daily,
        dates=calendar,
        kWm=max_soil_moisture[0],
    )

    init_moisture = model.estimate_initial_soil_moisture(
        max_iter=soil_moisture_init_max_iter,
        max_diff=soil_moisture_init_max_diff,
        verbose=False,
    )
    aet, moisture, runoff = model.calculate_soil_moisture(init_moisture)

    return aet, moisture, runoff


def splash_parameters(
    soil_moisture_init_max_iter: int = 10,
    soil_moisture_init_max_diff: float = 1.0,
) -> tuple[int, float]:
    """
    Parameters for the splash model.

    Parameters
    ----------
    soil_moisture_init_max_iter
        Maximum number of one year iterations used to estimate initial soil moisture.
    soil_moisture_init_max_diff
        Maximum acceptable difference between year start and year end soil moisture.
    """
    return (soil_moisture_init_max_iter, soil_moisture_init_max_diff)


@unpack_fields(
    "actual_evapotranspiration_daily",
    "soil_moisture_daily",
    "runoff_daily",
)
def splash(
    sunshine_fraction_daily: DataArray,
    temperature_celcius_daily: DataArray,
    precipitation_mm_daily: DataArray,
    elevation: DataArray,
    latitude: DataArray,
    max_soil_moisture: DataArray,
    splash_parameters: tuple[int, float],
) -> tuple[DataArray, DataArray, DataArray]:
    """Run the SPLASH water balance model.

    This function is intended to act as a node in a Hamilton DAG.

    Parameters
    ----------
    sunshine_fraction_daily
        Fraction of daylight hours that are sunny (dimensionless, 0-1).
    temperature_celcius_daily
        Air temperature (degrees Celsius).
    precipitation_mm_daily
        Precipitation (mm).
    latitude
        Latitude of the site (degrees).
    elevation
        Elevation of the site (meters).

    Returns
    -------
    tuple
        Tuple containing:
        - actual_evapotranspiration_daily: actual evapotranspiration (mm per day)
        - soil_moisture_daily: soil moisture content (mm)
        - runoff_daily: runoff (mm per day)
    """
    soil_moisture_init_max_iter, soil_moisture_init_max_diff = splash_parameters
    dates_daily = sunshine_fraction_daily.get_index("time")

    return _splash(
        sunshine_fraction_daily=sunshine_fraction_daily,
        temperature_celcius_daily=temperature_celcius_daily,
        precipitation_mm_daily=precipitation_mm_daily,
        elevation=elevation,
        latitude=latitude,
        max_soil_moisture=max_soil_moisture,
        soil_moisture_init_max_iter=soil_moisture_init_max_iter,
        soil_moisture_init_max_diff=soil_moisture_init_max_diff,
        dates_daily=dates_daily,
    )
