"""
Satterc-compatible interface to PyRealm's 'SPLASH' model.

This module provides the `splash` node, which wraps PyRealm's SPLASH model to
calculate actual evapotranspiration (AET), soil moisture, and runoff from climate
inputs, and returns the physical fluxes pyrealm computes along the way.
"""

import warnings
from operator import attrgetter
from typing import Annotated, TypedDict, cast

import numpy as np
import pyrealm.core.calendar
import pyrealm.splash.splash
import xarray as xr
from hamilton.function_modifiers import config, extract_fields
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

#: How many pixels an error or warning lists before truncating.
_MAX_PIXELS_NAMED = 10


def _describe_pixels(labels: NDArray) -> str:
    """Name up to `_MAX_PIXELS_NAMED` pixels in an error or warning."""
    named = ", ".join(str(label) for label in labels[:_MAX_PIXELS_NAMED])
    more = ", ..." if labels.size > _MAX_PIXELS_NAMED else ""
    return f"{labels.size} pixel(s) ({named}{more})"


def _spin_up(
    model: pyrealm.splash.splash.SplashModel, max_iter: int, max_diff: float
) -> NDArray[np.float64]:
    """Estimate initial soil moisture, allowing for every pixel being NaN.

    pyrealm's convergence test ignores NaN pixels until there are no others, and then
    reports that it failed to converge with a difference of NaN. That only matters to a
    block where pyrealm's calculation gives NaN for every pixel, which should get the
    same outcome as it would in a block with other pixels.
    """
    try:
        return model.estimate_initial_soil_moisture(
            max_iter=max_iter, max_diff=max_diff, verbose=False
        )
    except RuntimeError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            one_year = model.estimate_initial_soil_moisture(
                max_iter=1, return_convergence=True
            )
        if np.isnan(one_year).all():
            return one_year[-1]
        raise


def _run_pyrealm(
    sf: NDArray[np.float64],
    tc: NDArray[np.float64],
    pn: NDArray[np.float64],
    elevation: NDArray[np.float64],
    latitude: NDArray[np.float64],
    capacity: NDArray[np.float64],
    initial_soil_moisture: NDArray[np.float64] | None,
    *,
    dates: pyrealm.core.calendar.Calendar,
    max_iter: int,
    max_diff: float,
) -> tuple[NDArray[np.float64], ...]:
    """Run SPLASH on ``(time, pixel)`` arrays, in the order of `_OUTPUT_FIELDS`.

    ``lat``/``elv`` go in as ``(1, pixel)``: they pass through ``check_input_shapes``,
    which demands equal ndim with the climate arrays, then ``broadcast_time``. ``kWm``
    goes in as ``(pixel,)``: it is neither time-broadcast nor shape-checked, and a 2D
    value corrupts the per-day spatial shape.
    """
    model = pyrealm.splash.splash.SplashModel(
        lat=latitude[np.newaxis],
        elv=elevation[np.newaxis],
        sf=sf,
        tc=tc,
        pn=pn,
        dates=dates,
        kWm=capacity,
    )
    # pyrealm 2.0.0 does not pass kWm on to its evaporative fluxes, which would leave
    # the supply rate in AET using a capacity of 150 mm whatever is configured.
    model.evap.kWm = model.kWm

    if initial_soil_moisture is None:
        initial_soil_moisture = _spin_up(model, max_iter=max_iter, max_diff=max_diff)
    states = model.calculate_soil_moisture(initial_soil_moisture)
    fluxes = (attrgetter(name)(model) for name in _FLUX_ATTRIBUTES.values())
    return (*states, *fluxes)


def _splash_block(
    sunshine_fraction: NDArray[np.float64],
    temperature: NDArray[np.float64],
    precipitation: NDArray[np.float64],
    elevation: NDArray[np.float64],
    latitude: NDArray[np.float64],
    max_soil_moisture: NDArray[np.float64],
    pixel_labels: NDArray,
    initial_soil_moisture: NDArray[np.float64] | None = None,
    *,
    dates: pyrealm.core.calendar.Calendar,
    max_iter: int,
    max_diff: float,
) -> tuple[NDArray[np.float64], ...]:
    """Run SPLASH on a whole pixel-block, deciding each pixel from its own inputs.

    ``apply_ufunc`` places the ``time`` core dim last, so the climate inputs arrive as
    ``(pixel, time)`` and the rest as ``(pixel,)``. Outputs are returned as
    ``(pixel, time)`` for ``apply_ufunc``, in the order of `_OUTPUT_FIELDS`.

    A pixel's outcome must not depend on which other pixels share its block, or the
    outcome would change with the block size:

    - Masked: a NaN static or initial state, or no climate data at all. The output is
      NaN, and pyrealm never sees the pixel.
    - Gap: some climate values are missing. This raises, because soil moisture carries
      over from one day to the next, so everything after the gap would be lost.
    - Model failure: complete inputs but NaN soil moisture. The output is NaN and this
      warns, because a rare failure in pyrealm should not stop a large run.
    """
    climate = np.stack(
        [
            np.moveaxis(np.asarray(values, dtype=float), -1, 0)
            for values in (sunshine_fraction, temperature, precipitation)
        ]
    )
    n_days, n_pixels = climate.shape[1:]
    statics = np.stack(
        [
            np.broadcast_to(np.asarray(values, dtype=float), (n_pixels,))
            for values in (elevation, latitude, max_soil_moisture)
        ]
        + (
            []
            if initial_soil_moisture is None
            else [np.asarray(initial_soil_moisture, dtype=float).ravel()]
        )
    )

    missing = np.isnan(climate)
    masked = missing.all(axis=(0, 1)) | np.isnan(statics).any(axis=0)
    gaps = missing.any(axis=(0, 1)) & ~masked
    if gaps.any():
        raise ValueError(
            "SPLASH inputs have missing values in "
            f"{_describe_pixels(pixel_labels[gaps])}. Soil moisture carries over from "
            "day to day, so everything after a gap would be lost. Fill the gaps, or "
            "mask each pixel by making all of its climate inputs NaN."
        )

    outputs = np.full((len(_OUTPUT_FIELDS), n_pixels, n_days), np.nan)
    valid = ~masked
    if not valid.any():
        return tuple(outputs)

    sf, tc, pn = climate[:, :, valid]
    elv, lat, capacity = statics[:3, valid]
    results = _run_pyrealm(
        sf,
        tc,
        pn,
        elv,
        lat,
        capacity,
        statics[3, valid] if initial_soil_moisture is not None else None,
        dates=dates,
        max_iter=max_iter,
        max_diff=max_diff,
    )
    for output, result in zip(outputs, results, strict=True):
        output[valid] = np.moveaxis(result, 0, -1)

    moisture = outputs[_OUTPUT_FIELDS.index("soil_moisture_daily")]
    failed = valid & np.isnan(moisture).any(axis=-1)
    if failed.any():
        warnings.warn(
            "SPLASH returned NaN soil moisture for "
            f"{_describe_pixels(pixel_labels[failed])} despite complete inputs. "
            "pyrealm's AET calculation can fail at high latitudes.",
            stacklevel=2,
        )
    return tuple(outputs)


def _splash(
    sunshine_fraction_daily: DataArray,
    temperature_daily: DataArray,
    precipitation_daily: DataArray,
    elevation: DataArray,
    latitude: DataArray,
    max_soil_moisture: DataArray,
    dates_daily: DatetimeIndex,
    *,
    initial_soil_moisture: DataArray | None = None,
    soil_moisture_init_max_iter: int = 10,
    soil_moisture_init_max_diff: float = 1.0,
) -> SplashOut:
    """Apply SPLASH over the ``(time, pixel)`` block via `xarray.apply_ufunc`.

    SPLASH is sequential along ``time`` (soil moisture carries state day to day) but
    embarrassingly parallel over ``pixel``, so ``time`` is the input/output core dim and
    ``pixel`` the broadcast (mapped) dim. ``vectorize`` is left ``False`` so the whole
    pixel-block reaches pyrealm in one call and is vectorised internally; with
    ``dask="parallelized"`` a future dask backend can still chunk over ``pixel``.
    Unset, ``initial_soil_moisture`` means the initial state comes from the spin-up.
    """
    prescribed = [] if initial_soil_moisture is None else [initial_soil_moisture]
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
        sunshine_fraction_daily["pixel"],
        *prescribed,
        input_core_dims=[["time"]] * 3 + [[]] * (4 + len(prescribed)),
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


_INIT_METHODS = ("equilibrate", "prescribed")


class SplashConfig(TypedDict):
    """Settings for the `splash` node, as returned by `splash_config`.

    The descriptions and defaults live on `splash_config`, which is what the
    pipeline config populates.
    """

    soil_moisture_init_method: str
    soil_moisture_init_max_iter: int
    soil_moisture_init_max_diff: float


def splash_config(
    *,
    soil_moisture_init_method: str = "equilibrate",
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
    soil_moisture_init_method
        Where the initial soil moisture comes from: ``equilibrate`` repeats the
        first year until soil moisture stabilises, and ``prescribed`` takes it
        from the ``initial_soil_moisture`` input; see
        `soil_moisture_init_inputs__prescribed`.
    soil_moisture_init_max_iter
        Maximum number of one-year iterations used to estimate initial soil
        moisture. If any pixel in a block with a non-NaN result has not converged
        by then, the run stops with an error. Only used by ``equilibrate``.
    soil_moisture_init_max_diff
        Maximum acceptable difference between year-start and year-end soil
        moisture (millimetres). Only used by ``equilibrate``.

    Returns
    -------
    SplashConfig
        The settings for the model call.

    Raises
    ------
    ValueError
        If ``soil_moisture_init_method`` is not a known method. Without this a
        misspelt ``prescribed`` would silently select ``equilibrate``.
    """
    if soil_moisture_init_method not in _INIT_METHODS:
        raise ValueError(
            f"Unknown soil_moisture_init_method {soil_moisture_init_method!r}; "
            f"expected one of {', '.join(map(repr, _INIT_METHODS))}."
        )
    return SplashConfig(
        soil_moisture_init_method=soil_moisture_init_method,
        soil_moisture_init_max_iter=soil_moisture_init_max_iter,
        soil_moisture_init_max_diff=soil_moisture_init_max_diff,
    )


@config.when_not_in(soil_moisture_init_method=["prescribed"])
def soil_moisture_init_inputs__equilibrate() -> dict[str, DataArray]:
    """No extra inputs, for ``soil_moisture_init_method = "equilibrate"``."""
    return {}


@config.when_in(soil_moisture_init_method=["prescribed"])
@declare_units
def soil_moisture_init_inputs__prescribed(
    initial_soil_moisture: Annotated[DataArray, "mm"],
) -> dict[str, DataArray]:
    """Supply the starting state, for ``soil_moisture_init_method = "prescribed"``.

    Parameters
    ----------
    initial_soil_moisture
        Soil moisture on the day before the first date (millimetres), one value
        per pixel, between 0 and ``max_soil_moisture``.
    """
    return {"initial_soil_moisture": initial_soil_moisture}


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
    soil_moisture_init_inputs: dict[str, DataArray],
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
        Maximum soil moisture capacity (millimetres). Sets the runoff threshold and
        scales the rate at which the soil supplies water for evaporation.
    soil_moisture_init_inputs
        Extra inputs for the configured ``soil_moisture_init_method``.
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
        initial_soil_moisture=soil_moisture_init_inputs.get("initial_soil_moisture"),
        soil_moisture_init_max_iter=splash_config["soil_moisture_init_max_iter"],
        soil_moisture_init_max_diff=splash_config["soil_moisture_init_max_diff"],
    )
