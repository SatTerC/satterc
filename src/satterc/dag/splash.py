"""
Satterc-compatable interface to PyRealm's 'Splash' model.

This module provides the Splash class, which wraps the SPLASH model
to calculate soil moisture, actual evapotranspiration (AET), and runoff based
on climate inputs.
"""

from typing import Annotated, TypedDict, cast

import numpy as np
import pyrealm.core.calendar
import pyrealm.splash.splash
import xarray as xr
from hamilton.function_modifiers import extract_fields
from numpy.typing import NDArray
from pandas import DatetimeIndex
from xarray import DataArray

from ._utils import declare_units


class SplashOut(TypedDict):
    """Outputs of the :func:`splash` node, with their declared units."""

    actual_evapotranspiration_daily: Annotated[DataArray, "mm d-1"]
    soil_moisture_daily: Annotated[DataArray, "mm"]
    runoff_daily: Annotated[DataArray, "mm d-1"]


def _splash_block(
    sunshine_fraction: NDArray[np.float64],
    temperature_celcius: NDArray[np.float64],
    precipitation_mm: NDArray[np.float64],
    elevation: NDArray[np.float64],
    latitude: NDArray[np.float64],
    max_soil_moisture: NDArray[np.float64],
    *,
    dates: pyrealm.core.calendar.Calendar,
    max_iter: int,
    max_diff: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Run SPLASH on a whole pixel-block (vectorised over pixels by pyrealm).

    ``apply_ufunc`` places the ``time`` core dim last, so the climate inputs arrive as
    ``(pixel, time)`` and the per-pixel statics as ``(pixel,)``. SPLASH expects time
    along the first axis, so the climate arrays are moved to ``(time, pixel)`` and the
    statics reshaped to the layout pyrealm requires: ``lat``/``elv`` as ``(1, pixel)``
    (these pass through ``check_input_shapes``, which demands equal ndim with the 2D
    climate arrays, then ``broadcast_time``); ``kWm`` as ``(pixel,)`` (it is neither
    time-broadcast nor shape-checked, and a 2D value corrupts the per-day spatial
    shape). Outputs are returned as ``(pixel, time)`` (core dim last) for
    ``apply_ufunc``. This whole-block kernel replaces a former per-pixel-0 indexing
    bug, where every pixel silently received pixel 0's latitude/elevation/capacity.
    """
    sf = np.moveaxis(np.asarray(sunshine_fraction, dtype=float), -1, 0)
    tc = np.moveaxis(np.asarray(temperature_celcius, dtype=float), -1, 0)
    pn = np.moveaxis(np.asarray(precipitation_mm, dtype=float), -1, 0)

    model = pyrealm.splash.splash.SplashModel(
        lat=np.atleast_2d(np.asarray(latitude, dtype=float)),
        elv=np.atleast_2d(np.asarray(elevation, dtype=float)),
        sf=sf,
        tc=tc,
        pn=pn,
        dates=dates,
        kWm=np.asarray(max_soil_moisture, dtype=float).ravel(),
    )

    init_moisture = model.estimate_initial_soil_moisture(
        max_iter=max_iter,
        max_diff=max_diff,
        verbose=False,
    )
    aet, moisture, runoff = model.calculate_soil_moisture(init_moisture)

    return (
        np.moveaxis(aet, 0, -1),
        np.moveaxis(moisture, 0, -1),
        np.moveaxis(runoff, 0, -1),
    )


def _splash(
    sunshine_fraction_daily: DataArray,
    temperature_celcius_daily: DataArray,
    precipitation_mm_daily: DataArray,
    elevation: DataArray,
    latitude: DataArray,
    max_soil_moisture: DataArray,
    dates_daily: DatetimeIndex,
    *,
    soil_moisture_init_max_iter: int = 10,
    soil_moisture_init_max_diff: float = 1.0,
) -> SplashOut:
    """Apply SPLASH over the ``(time, pixel)`` block via :func:`xarray.apply_ufunc`.

    SPLASH is sequential along ``time`` (soil moisture carries state day to day) but
    embarrassingly parallel over ``pixel``, so ``time`` is the input/output core dim and
    ``pixel`` the broadcast (mapped) dim. ``vectorize`` is left ``False`` so the whole
    pixel-block reaches pyrealm in one call and is vectorised internally; with
    ``dask="parallelized"`` a future dask backend can still chunk over ``pixel``.
    """
    calendar = pyrealm.core.calendar.Calendar(dates_daily.values)

    aet, moisture, runoff = xr.apply_ufunc(
        _splash_block,
        sunshine_fraction_daily,
        temperature_celcius_daily,
        precipitation_mm_daily,
        elevation,
        latitude,
        max_soil_moisture,
        input_core_dims=[["time"]] * 3 + [[]] * 3,
        output_core_dims=[["time"]] * 3,
        kwargs={
            "dates": calendar,
            "max_iter": soil_moisture_init_max_iter,
            "max_diff": soil_moisture_init_max_diff,
        },
        dask="parallelized",
        output_dtypes=[float] * 3,
    )

    # apply_ufunc drops the `time` coord (a core dim) and orders outputs as
    # (pixel, time); reattach the coord and restore the canonical (time, pixel).
    time_coord = sunshine_fraction_daily.coords["time"]
    return cast(
        SplashOut,
        {
            "actual_evapotranspiration_daily": aet.assign_coords(
                time=time_coord
            ).transpose("time", "pixel"),
            "soil_moisture_daily": moisture.assign_coords(time=time_coord).transpose(
                "time", "pixel"
            ),
            "runoff_daily": runoff.assign_coords(time=time_coord).transpose(
                "time", "pixel"
            ),
        },
    )


@extract_fields()
@declare_units
def splash(
    dates_daily: DatetimeIndex,
    sunshine_fraction_daily: Annotated[DataArray, "1"],
    temperature_celcius_daily: Annotated[DataArray, "degC"],
    precipitation_mm_daily: Annotated[DataArray, "mm"],
    elevation: Annotated[DataArray, "m"],
    latitude: DataArray,
    max_soil_moisture: Annotated[DataArray, "mm"],
    *,
    soil_moisture_init_max_iter: int = 10,
    soil_moisture_init_max_diff: float = 1.0,
) -> SplashOut:
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
    soil_moisture_init_max_iter
        Maximum number of one year iterations used to estimate initial soil moisture.
    soil_moisture_init_max_diff
        Maximum acceptable difference between year start and year end soil moisture.

    Returns
    -------
    tuple
        Tuple containing:
        - actual_evapotranspiration_daily: actual evapotranspiration (mm per day)
        - soil_moisture_daily: soil moisture content (mm)
        - runoff_daily: runoff (mm per day)
    """
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
