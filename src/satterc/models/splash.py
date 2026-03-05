"""
Satterc-compatable interface to PyRealm's 'Splash' model.

This module provides the Splash class, which wraps the SPLASH model
to calculate soil moisture, actual evapotranspiration (AET), and runoff based
on climate inputs.
"""

from hamilton.function_modifiers import unpack_fields
import numpy as np
from numpy.typing import NDArray
import pyrealm.splash.splash
import pyrealm.core.calendar


def splash_parameters(
    latitude: float,
    elevation: float,
    max_soil_moisture: float,
) -> tuple[float, float, float]:
    """
    Parameters for the splash model.

    Parameters
    ----------
    latitude
        Latitude of the site (degrees).
    elevation
        Elevation of the site (meters).
    max_soil_moisture
        Maximum soil moisture capacity (mm).
    """
    return latitude, elevation, max_soil_moisture


@unpack_fields(
    "actual_evapotranspiration_daily",
    "soil_moisture_daily",
    "runoff_daily",
)
def splash(
    sunshine_fraction_daily: NDArray[np.float64],
    temperature_celcius_daily: NDArray[np.float64],
    precipitation_mm_daily: NDArray[np.float64],
    dates_daily: NDArray[np.datetime64],
    splash_parameters: tuple[float, float, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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
    dates_daily
        Dates corresponding to the input data (datetime64).

    Returns
    -------
    tuple
        Tuple containing:
        - actual_evapotranspiration_daily: actual evapotranspiration (mm per day)
        - soil_moisture_daily: soil moisture content (mm)
        - runoff_daily: runoff (mm per day)
    """
    latitude, elevation, max_soil_moisture = splash_parameters

    calendar = pyrealm.core.calendar.Calendar(dates_daily)
    n = len(calendar)

    # NOTE: need to add dummy spatial dimension due to bug in Splash.
    # See https://github.com/ImperialCollegeLondon/pyrealm/issues/626
    model = pyrealm.splash.splash.SplashModel(
        lat=np.full((n, 1), latitude),
        elv=np.full((n, 1), elevation),
        dates=calendar,
        sf=sunshine_fraction_daily.reshape(n, 1),
        tc=temperature_celcius_daily.reshape(n, 1),
        pn=precipitation_mm_daily.reshape(n, 1),
    )

    init_moisture = model.estimate_initial_soil_moisture(verbose=False)
    aet, moisture, runoff = model.calculate_soil_moisture(init_moisture)

    return aet.flatten(), moisture.flatten(), runoff.flatten()
