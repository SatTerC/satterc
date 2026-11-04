"""
Satterc-compatible interface to PyRealm's 'SPLASH' model.

This module provides the `splash` node, which wraps PyRealm's SPLASH model to
calculate actual evapotranspiration (AET), soil moisture, and runoff from climate
inputs, and returns the physical fluxes pyrealm computes along the way.
"""

from operator import attrgetter
from typing import Annotated, TypedDict, cast

import numpy as np
import pyrealm.core.calendar
import pyrealm.splash.splash
import xarray as xr
from hamilton.function_modifiers import extract_fields
from numpy.typing import NDArray
from pandas import DatetimeIndex
from xarray import DataArray
from xarray_annotated.temporal import declare_freq
from xarray_annotated.units import declare_units

from ..temporal import DAILY, time_index


class SplashOut(TypedDict):
    """Outputs of the `splash` node, at daily resolution."""

    actual_evapotranspiration_daily: Annotated[DataArray, "mm d-1", DAILY]
    """Actual evapotranspiration: the daily water loss to the atmosphere
    (millimetres per day)."""
    soil_moisture_daily: Annotated[DataArray, "mm", DAILY]
    """Soil moisture content at the end of the day (millimetres)."""
    runoff_daily: Annotated[DataArray, "mm", DAILY]
    """Runoff: the soil-moisture overflow amount above capacity for the day
    (millimetres, an amount rather than a rate)."""
    potential_evapotranspiration_daily: Annotated[DataArray, "mm d-1", DAILY]
    """Potential evapotranspiration: the water that *would* be lost to the
    atmosphere given the available energy, if soil moisture were not limiting
    (millimetres per day).

    SPLASH computes this on the way to actual evapotranspiration, by the
    Priestley-Taylor relation ``PET = (1 + k_w) * EET``, where equilibrium
    evapotranspiration follows from daytime net radiation and ``k_w = 0.26``. It
    is the demand term in an aridity index, which `satterc.models.pmodel` needs
    for ``method_kphio = "sandoval"``."""
    equilibrium_evapotranspiration_daily: Annotated[DataArray, "mm d-1", DAILY]
    """Equilibrium evapotranspiration: the energy-limited loss from daytime net
    radiation, before the Priestley-Taylor scaling that gives potential
    evapotranspiration (millimetres per day)."""
    condensation_daily: Annotated[DataArray, "mm d-1", DAILY]
    """Condensation, from nighttime net radiation (millimetres per day). With
    precipitation, this is the supply side of the daily water balance."""
    extraterrestrial_radiation_daily: Annotated[DataArray, "J m-2 d-1", DAILY]
    """Solar radiation arriving at the top of the atmosphere over the day."""
    transmissivity_daily: Annotated[DataArray, "1", DAILY]
    """Bulk atmospheric transmissivity, from sunshine fraction and elevation."""
    estimated_ppfd_daily: Annotated[DataArray, "mol m-2 d-1", DAILY]
    """Photosynthetic photon flux density estimated from the radiation model.

    The units follow the calculation, which converts a daily radiation total.
    pyrealm's docstrings say micromoles per square metre per second, but its
    values are moles per square metre per day. This is a daily total, unlike the
    instantaneous rate `satterc.models.pmodel` takes as ``ppfd_weekly``."""
    net_longwave_radiation_daily: Annotated[DataArray, "W m-2", DAILY]
    """Net longwave radiation, as a rate that is constant over the day."""
    daytime_net_radiation_daily: Annotated[DataArray, "J m-2 d-1", DAILY]
    """Net radiation over the hours when it is positive."""
    nighttime_net_radiation_daily: Annotated[DataArray, "J m-2 d-1", DAILY]
    """Net radiation over the hours when it is negative, so this is negative."""


#: `SplashOut` field -> where pyrealm's `SplashModel` holds it. These are computed on
#: construction, independently of the soil moisture iteration, so are simply read off.
_FLUX_ATTRIBUTES = {
    "potential_evapotranspiration_daily": "evap.pet_d",
    "equilibrium_evapotranspiration_daily": "evap.eet_d",
    "condensation_daily": "evap.cond",
    "extraterrestrial_radiation_daily": "solar.daily_solar_radiation",
    "transmissivity_daily": "solar.transmissivity",
    "estimated_ppfd_daily": "solar.daily_ppfd",
    "net_longwave_radiation_daily": "solar.net_longwave_radiation",
    "daytime_net_radiation_daily": "solar.daytime_net_radiation",
    "nighttime_net_radiation_daily": "solar.nighttime_net_radiation",
}

#: `SplashOut` fields, in the order `_splash_block` returns them.
_OUTPUT_FIELDS = (
    "actual_evapotranspiration_daily",
    "soil_moisture_daily",
    "runoff_daily",
    *_FLUX_ATTRIBUTES,
)


def _splash_block(
    sunshine_fraction: NDArray[np.float64],
    temperature: NDArray[np.float64],
    precipitation: NDArray[np.float64],
    elevation: NDArray[np.float64],
    latitude: NDArray[np.float64],
    max_soil_moisture: NDArray[np.float64],
    *,
    dates: pyrealm.core.calendar.Calendar,
    max_iter: int,
    max_diff: float,
) -> tuple[NDArray[np.float64], ...]:
    """Run SPLASH on a whole pixel-block (vectorised over pixels by pyrealm).

    ``apply_ufunc`` places the ``time`` core dim last, so the climate inputs arrive as
    ``(pixel, time)`` and the per-pixel statics as ``(pixel,)``. SPLASH expects time
    along the first axis, so the climate arrays are moved to ``(time, pixel)`` and the
    statics reshaped to the layout pyrealm requires: ``lat``/``elv`` as ``(1, pixel)``
    (these pass through ``check_input_shapes``, which demands equal ndim with the 2D
    climate arrays, then ``broadcast_time``); ``kWm`` as ``(pixel,)`` (it is neither
    time-broadcast nor shape-checked, and a 2D value corrupts the per-day spatial
    shape). Outputs are returned as ``(pixel, time)`` (core dim last) for
    ``apply_ufunc``, in the order of `_OUTPUT_FIELDS`. This whole-block kernel
    replaces a former per-pixel-0 indexing bug, where every pixel silently received
    pixel 0's latitude/elevation/capacity.
    """
    sf = np.moveaxis(np.asarray(sunshine_fraction, dtype=float), -1, 0)
    tc = np.moveaxis(np.asarray(temperature, dtype=float), -1, 0)
    pn = np.moveaxis(np.asarray(precipitation, dtype=float), -1, 0)

    model = pyrealm.splash.splash.SplashModel(
        lat=np.atleast_2d(np.asarray(latitude, dtype=float)),
        elv=np.atleast_2d(np.asarray(elevation, dtype=float)),
        sf=sf,
        tc=tc,
        pn=pn,
        dates=dates,
        kWm=np.asarray(max_soil_moisture, dtype=float).ravel(),
    )
    # pyrealm 2.0.0 does not pass kWm on to its evaporative fluxes, which would leave
    # the supply rate in AET using a capacity of 150 mm whatever is configured.
    model.evap.kWm = model.kWm

    init_moisture = model.estimate_initial_soil_moisture(
        max_iter=max_iter,
        max_diff=max_diff,
        verbose=False,
    )
    aet, moisture, runoff = model.calculate_soil_moisture(init_moisture)

    fluxes = (attrgetter(name)(model) for name in _FLUX_ATTRIBUTES.values())
    return tuple(np.moveaxis(out, 0, -1) for out in (aet, moisture, runoff, *fluxes))


def _splash(
    sunshine_fraction_daily: DataArray,
    temperature_daily: DataArray,
    precipitation_daily: DataArray,
    elevation: DataArray,
    latitude: DataArray,
    max_soil_moisture: DataArray,
    dates_daily: DatetimeIndex,
    *,
    soil_moisture_init_max_iter: int = 10,
    soil_moisture_init_max_diff: float = 1.0,
) -> SplashOut:
    """Apply SPLASH over the ``(time, pixel)`` block via `xarray.apply_ufunc`.

    SPLASH is sequential along ``time`` (soil moisture carries state day to day) but
    embarrassingly parallel over ``pixel``, so ``time`` is the input/output core dim and
    ``pixel`` the broadcast (mapped) dim. ``vectorize`` is left ``False`` so the whole
    pixel-block reaches pyrealm in one call and is vectorised internally; with
    ``dask="parallelized"`` a future dask backend can still chunk over ``pixel``.
    """
    calendar = pyrealm.core.calendar.Calendar(dates_daily.values)

    n_outputs = len(_OUTPUT_FIELDS)
    outputs = xr.apply_ufunc(
        _splash_block,
        sunshine_fraction_daily,
        temperature_daily,
        precipitation_daily,
        elevation,
        latitude,
        max_soil_moisture,
        input_core_dims=[["time"]] * 3 + [[]] * 3,
        output_core_dims=[["time"]] * n_outputs,
        kwargs={
            "dates": calendar,
            "max_iter": soil_moisture_init_max_iter,
            "max_diff": soil_moisture_init_max_diff,
        },
        dask="parallelized",
        output_dtypes=[float] * n_outputs,
    )

    # apply_ufunc drops the `time` coord (a core dim) and orders outputs as
    # (pixel, time); reattach the coord and restore the canonical (time, pixel).
    time_coord = sunshine_fraction_daily.coords["time"]
    return cast(
        SplashOut,
        {
            name: out.assign_coords(time=time_coord).transpose("time", "pixel")
            for name, out in zip(_OUTPUT_FIELDS, outputs, strict=True)
        },
    )


class SplashConfig(TypedDict):
    """Settings for the `splash` node, as returned by `splash_config`.

    The descriptions and defaults live on `splash_config`, which is what the
    pipeline config populates.
    """

    soil_moisture_init_max_iter: int
    soil_moisture_init_max_diff: float


def splash_config(
    *,
    soil_moisture_init_max_iter: int = 10,
    soil_moisture_init_max_diff: float = 1.0,
) -> SplashConfig:
    """Collect the SPLASH settings from the pipeline config.

    Grouping the settings into one node keeps them out of `splash`'s data
    inputs. It does not namespace them: conduit merges every module section's
    params into one flat driver config, so these are still configured as
    top-level keys of ``[splash]``.

    Parameters
    ----------
    soil_moisture_init_max_iter
        Maximum number of one-year iterations used to estimate initial soil
        moisture.
    soil_moisture_init_max_diff
        Maximum acceptable difference between year-start and year-end soil
        moisture (millimetres).

    Returns
    -------
    SplashConfig
        The settings, ready to unpack into the model call.
    """
    return SplashConfig(
        soil_moisture_init_max_iter=soil_moisture_init_max_iter,
        soil_moisture_init_max_diff=soil_moisture_init_max_diff,
    )


@extract_fields()
@declare_units
@declare_freq
def splash(
    sunshine_fraction_daily: Annotated[DataArray, "1", DAILY],
    temperature_daily: Annotated[DataArray, "degC", DAILY],
    precipitation_daily: Annotated[DataArray, "mm d-1", DAILY],
    elevation: Annotated[DataArray, "m"],
    # Deliberately undeclared. pyrealm wants degrees, but conduit computes
    # `latitude` from the input CRS (`conduit.gridded.io.compute_lat_lon`) and
    # builds the array without a `units` attribute, so under strict validation
    # (`on_missing="error"`, as tests/test_contracts.py runs) declaring a unit
    # here fails every gridded pipeline. Stamping that attribute is generic
    # geospatial behaviour and belongs in conduit. Once it lands, declare
    # "degrees" here and add ("latitude", "lat") back to SPLASH_INPUT_UNITS in
    # tests/test_pyrealm_units.py.
    latitude: DataArray,
    max_soil_moisture: Annotated[DataArray, "mm"],
    splash_config: SplashConfig,
) -> SplashOut:
    """Run the SPLASH water balance model.

    The daily calendar SPLASH needs is read off ``temperature_daily``'s time
    coordinate.

    Parameters
    ----------
    sunshine_fraction_daily
        Fraction of daylight hours that are sunny (dimensionless, 0-1).
    temperature_daily
        Daily mean air temperature (degrees Celsius).
    precipitation_daily
        Precipitation (millimetres per day).
    elevation
        Elevation of the site (metres).
    latitude
        Latitude of the site (degrees).
    max_soil_moisture
        Maximum soil moisture capacity (millimetres).
    splash_config
        Model settings; see `splash_config`.

    Returns
    -------
    SplashOut
        Dictionary of daily outputs:

        - actual_evapotranspiration_daily: actual evapotranspiration
          (millimetres per day)
        - soil_moisture_daily: soil moisture content (millimetres)
        - runoff_daily: runoff overflow amount (millimetres)
        - potential_evapotranspiration_daily: energy-limited water demand
          (millimetres per day)
        - equilibrium_evapotranspiration_daily, condensation_daily: the other
          water fluxes (millimetres per day)
        - extraterrestrial_radiation_daily, transmissivity_daily,
          estimated_ppfd_daily, net_longwave_radiation_daily,
          daytime_net_radiation_daily, nighttime_net_radiation_daily: the
          radiation fluxes

        See `SplashOut` for per-output detail.
    """
    return _splash(
        sunshine_fraction_daily=sunshine_fraction_daily,
        temperature_daily=temperature_daily,
        precipitation_daily=precipitation_daily,
        elevation=elevation,
        latitude=latitude,
        max_soil_moisture=max_soil_moisture,
        dates_daily=time_index(temperature_daily, "temperature_daily"),
        **splash_config,
    )
