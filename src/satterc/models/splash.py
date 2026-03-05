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


@unpack_fields("actual_evapotranspiration", "soil_moisture", "runoff")
def splash(
    sunshine_fraction: NDArray[np.float64],
    temperature_celcius: NDArray[np.float64],
    precipitation_mm: NDArray[np.float64],
    dates: NDArray[np.datetime64],
    latitude: float,
    elevation: float,
    max_soil_moisture: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Run the SPLASH water balance model.

    This function is intended to act as a node in a Hamilton DAG.

    Parameters
    ----------
    sunshine_fraction
        Fraction of daylight hours that are sunny (dimensionless, 0-1).
    temperature_celcius
        Air temperature (degrees Celsius).
    precipitation_mm
        Precipitation (mm).
    dates
        Dates corresponding to the input data (datetime64).
    latitude
        Latitude of the site (degrees).
    elevation
        Elevation of the site (meters).
    max_soil_moisture
        Maximum soil moisture capacity (mm).
 
    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        Tuple containing actual evapotranspiration (mm per day), soil_moisture content (mm), \
        and runoff (mm per day).
    """
    calendar = pyrealm.core.calendar.Calendar(dates)
    n = len(calendar)

    # NOTE: need to add dummy spatial dimension due to bug in Splash.
    # See https://github.com/ImperialCollegeLondon/pyrealm/issues/626
    model = pyrealm.splash.splash.SplashModel(
        lat=np.full((n, 1), latitude),
        elv=np.full((n, 1), elevation),
        dates=calendar,
        sf=sunshine_fraction.reshape(n, 1),
        tc=temperature_celcius.reshape(n, 1),
        pn=precipitation_mm.reshape(n, 1),
    )

    init_moisture = model.estimate_initial_soil_moisture(verbose=False)
    aet, moisture, runoff = model.calculate_soil_moisture(init_moisture)

    return aet.flatten(), moisture.flatten(), runoff.flatten()
