"""
Satterc-compatable interface to PyRealm's 'Splash' model.

This module provides the Splash class, which wraps the SPLASH model
to calculate soil moisture, actual evapotranspiration (AET), and runoff based
on climate inputs.
"""

from hamilton.function_modifiers import unpack_fields
from xarray import DataArray
import pyrealm.splash.splash
import pyrealm.core.calendar


def splash_parameters(
    max_soil_moisture: float,
) -> tuple[float]:
    """
    Parameters for the splash model.

    Parameters
    ----------
    max_soil_moisture
        Maximum soil moisture capacity (mm).
    """
    return (max_soil_moisture,)


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
    splash_parameters: tuple[float],
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
    (max_soil_moisture,) = splash_parameters

    # NOTE: dates and latitude should probably be promoted to a node
    dates_daily = sunshine_fraction_daily.time.values
    latitude = sunshine_fraction_daily.coords["lat"].values

    calendar = pyrealm.core.calendar.Calendar(dates_daily)

    # TODO: check this actually runs with arrays extracted from DataArray
    model = pyrealm.splash.splash.SplashModel(
        lat=latitude,
        elv=elevation.data,
        dates=calendar,
        sf=sunshine_fraction_daily.data,
        tc=temperature_celcius_daily.data,
        pn=precipitation_mm_daily.data,
    )

    init_moisture = model.estimate_initial_soil_moisture(verbose=False)
    aet, moisture, runoff = model.calculate_soil_moisture(init_moisture)

    return aet.flatten(), moisture.flatten(), runoff.flatten()
