"""ACM-GPP-ET canopy model interface for the SatTerC pipeline.

ACM-GPP-ET (Smallman & Williams, 2019) is a Fortran model of daily canopy
photosynthesis and evapotranspiration, coupled to a layered soil-water bucket.
It is called through an f2py extension, `satterc.models._acm_gpp_et`, which is
compiled from the upstream source by ``just build-acm`` and is not shipped with
the package; importing this module without it raises `ImportError`.
"""

from typing import Annotated, TypedDict, cast

import numpy as np
import pandas as pd
import xarray as xr
from hamilton.function_modifiers import extract_fields
from numpy.typing import NDArray
from xarray import DataArray
from xarray_annotated.temporal import declare_freq
from xarray_annotated.units import declare_units

from ..temporal import DAILY, time_index

try:
    from ._acm_gpp_et import acm_pixel
except ImportError as exc:
    raise ImportError(
        "satterc.models.acm_gpp_et needs the compiled ACM-GPP-ET extension, "
        "which is not built by default. Build it with `just build-acm` "
        "(requires gfortran)."
    ) from exc


class AcmGppEtOut(TypedDict):
    """Outputs of the `acm_gpp_et` node, at daily resolution.

    Water fluxes are reported as ACM-GPP-ET reports them, as mass of water per
    unit ground area per day, which is numerically millimetres per day.
    """

    canopy_gpp_daily: Annotated[DataArray, "g m-2 d-1", DAILY]
    """Gross primary productivity (grams of carbon per square metre per day)."""
    transpiration_daily: Annotated[DataArray, "kg m-2 d-1", DAILY]
    """Canopy transpiration (kilograms of water per square metre per day)."""
    wet_canopy_evaporation_daily: Annotated[DataArray, "kg m-2 d-1", DAILY]
    """Evaporation of rainfall intercepted by the canopy (kilograms of water per
    square metre per day)."""
    soil_evaporation_daily: Annotated[DataArray, "kg m-2 d-1", DAILY]
    """Evaporation from the soil surface, including snow sublimation (kilograms
    of water per square metre per day)."""
    surface_runoff_daily: Annotated[DataArray, "kg m-2 d-1", DAILY]
    """Surface runoff (kilograms of water per square metre per day)."""
    drainage_daily: Annotated[DataArray, "kg m-2 d-1", DAILY]
    """Drainage from the bottom of the soil column (kilograms of water per
    square metre per day)."""
    leaf_water_potential_daily: Annotated[DataArray, "MPa", DAILY]
    """Mean leaf water potential (megapascals)."""
    internal_co2_daily: Annotated[DataArray, "umol mol-1", DAILY]
    """Leaf-internal CO2 concentration (micromoles per mole)."""
    topsoil_water_daily: Annotated[DataArray, "mm", DAILY]
    """Water held in the top 10 cm soil layer (millimetres)."""
    weighted_soil_water_potential_daily: Annotated[DataArray, "MPa", DAILY]
    """Soil water potential weighted by root uptake across layers
    (megapascals)."""


# Columns of the shim's ``fluxes`` output, in order, followed by its
# ``soil_water`` and ``wswp`` outputs.
_ACM_OUTPUT_NAMES: tuple[str, ...] = (
    "canopy_gpp_daily",
    "transpiration_daily",
    "wet_canopy_evaporation_daily",
    "soil_evaporation_daily",
    "surface_runoff_daily",
    "drainage_daily",
    "leaf_water_potential_daily",
    "internal_co2_daily",
    "topsoil_water_daily",
    "weighted_soil_water_potential_daily",
)

#: Dimension along which soil inputs may vary by layer, top layer first.
SOIL_LAYER_DIM = "soil_layer"

# ACM-GPP-ET's soil column has four layers: two fixed topsoil layers (0-10 cm,
# 10-30 cm), the rest of the rooting zone, and the column below it.
_N_SOIL_LAYERS = 4

_STATUS_MESSAGES = {
    1: (
        "ACM-GPP-ET was already run in this process on a series of a different "
        "length. It sizes some module-level arrays on the first call and never "
        "resizes them, so every call in one process must use the same number "
        "of days."
    ),
    2: "ACM-GPP-ET's soil layer count no longer matches the f2py shim.",
}


def _soil_layers(values: float | NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Broadcast one pixel's soil property onto ACM-GPP-ET's four layers.

    Accepts a scalar or a length-1 array (one value for the whole column) or a
    length-4 array (one value per layer).
    """
    values = np.asarray(values, dtype=np.float64)
    try:
        return np.array(np.broadcast_to(values, (_N_SOIL_LAYERS,)))
    except ValueError:
        raise ValueError(
            f"{name} has {values.size} values along {SOIL_LAYER_DIM!r}; "
            f"ACM-GPP-ET needs 1 (applied to every layer) or {_N_SOIL_LAYERS}."
        ) from None


def _acm_gpp_et_1px(
    temperature_min: NDArray[np.float64],
    temperature_max: NDArray[np.float64],
    temperature: NDArray[np.float64],
    shortwave_radiation: NDArray[np.float64],
    co2: NDArray[np.float64],
    precipitation: NDArray[np.float64],
    wind_speed: NDArray[np.float64],
    vpd: NDArray[np.float64],
    lai: NDArray[np.float64],
    fine_root_carbon: NDArray[np.float64],
    latitude: float,
    clay_content: float | NDArray[np.float64],
    sand_content: float | NDArray[np.float64],
    *,
    run_day: NDArray[np.float64],
    day_of_year: NDArray[np.float64],
    pars: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ...]:
    """Run ACM-GPP-ET for a single pixel.

    The driver arguments are 1D ``(time,)`` arrays for one pixel; ``latitude``
    is a scalar, and the soil contents are scalars or ``(soil_layer,)`` arrays,
    see `_soil_layers`. ``run_day``, ``day_of_year``
    and ``pars`` are pixel-invariant and passed through unchanged. Returns one
    ``(time,)`` array per output, ordered as `_ACM_OUTPUT_NAMES`.
    """
    # The 12 drivers, in CARBON_MODEL's row order. Fortran is column-major, so
    # building the array in that order up front saves f2py a silent copy.
    met = np.asfortranarray(
        np.stack(
            [
                run_day,
                temperature_min,
                temperature_max,
                shortwave_radiation,
                co2,
                day_of_year,
                precipitation / 86400.0,  # mm d-1 -> kg m-2 s-1
                temperature,
                wind_speed,
                vpd,
                lai,
                fine_root_carbon,
            ]
        ),
        dtype=np.float64,
    )
    fluxes, soil_water, wswp, status = acm_pixel(
        met,
        pars,
        float(latitude),
        _soil_layers(clay_content, "clay_content"),
        _soil_layers(sand_content, "sand_content"),
    )
    if status != 0:
        raise RuntimeError(_STATUS_MESSAGES[status])
    return (*fluxes.T, soil_water, wswp)


def _acm_gpp_et(
    temperature_min_daily: DataArray,
    temperature_max_daily: DataArray,
    temperature_daily: DataArray,
    shortwave_radiation_daily: DataArray,
    co2_daily: DataArray,
    precipitation_daily: DataArray,
    wind_speed_daily: DataArray,
    vpd_daily: DataArray,
    lai_daily: DataArray,
    fine_root_carbon_daily: DataArray,
    latitude: DataArray,
    clay_content: DataArray,
    sand_content: DataArray,
    dates_daily: pd.DatetimeIndex,
    *,
    foliar_nitrogen: float,
    min_leaf_water_potential: float,
    root_biomass_half_depth: float,
    max_root_depth: float,
    canopy_height: float,
) -> AcmGppEtOut:
    """Map `_acm_gpp_et_1px` over the stacked ``pixel`` dimension.

    The same `xarray.apply_ufunc` seam as `satterc.models.rothc`: ``time`` is
    the core dimension, ``pixel`` is mapped, and ``vectorize=True`` loops over
    pixels in Python because the Fortran runs one pixel per call. A soil input
    with a ``soil_layer`` dimension gets it as a second core dimension, so the
    kernel receives that pixel's whole profile.

    The extension keeps state in Fortran module variables, so it must never run
    two pixels at once in one process. That holds today because an f2py wrapper
    holds the GIL for the whole call unless its signature opts out with
    ``threadsafe``, which this one must not. A threaded dask scheduler therefore
    serialises these calls; parallelism has to come from processes.
    """
    n_days = len(dates_daily)
    pars = np.array(
        [
            foliar_nitrogen,
            min_leaf_water_potential,
            root_biomass_half_depth,
            max_root_depth,
            canopy_height,
        ],
        dtype=np.float64,
    )

    outputs = xr.apply_ufunc(
        _acm_gpp_et_1px,
        temperature_min_daily,
        temperature_max_daily,
        temperature_daily,
        shortwave_radiation_daily,
        co2_daily,
        precipitation_daily,
        wind_speed_daily,
        vpd_daily,
        lai_daily,
        fine_root_carbon_daily,
        latitude,
        clay_content,
        sand_content,
        input_core_dims=[["time"]] * 10
        + [[]]
        + [
            [SOIL_LAYER_DIM] if SOIL_LAYER_DIM in da.dims else []
            for da in (clay_content, sand_content)
        ],
        output_core_dims=[["time"]] * len(_ACM_OUTPUT_NAMES),
        kwargs={
            "run_day": np.arange(1, n_days + 1, dtype=np.float64),
            # Missing from the pandas type stubs, as in `satterc.models.rothc`.
            "day_of_year": dates_daily.day_of_year.to_numpy(dtype=np.float64),  # type: ignore[reportAttributeAccessIssue]
            "pars": pars,
        },
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float] * len(_ACM_OUTPUT_NAMES),
    )

    # apply_ufunc drops the `time` coordinate (a core dim) and orders outputs as
    # (pixel, time); reattach the coordinate and restore the canonical (time, pixel).
    time_coord = temperature_daily.coords["time"]
    return cast(
        AcmGppEtOut,
        {
            name: da.assign_coords(time=time_coord).transpose("time", "pixel")
            for name, da in zip(_ACM_OUTPUT_NAMES, outputs, strict=True)
        },
    )


class AcmGppEtConfig(TypedDict):
    """Settings for the `acm_gpp_et` node, as returned by `acm_gpp_et_config`.

    The descriptions and defaults live on `acm_gpp_et_config`, which is what the
    pipeline config populates.
    """

    foliar_nitrogen: float
    min_leaf_water_potential: float
    root_biomass_half_depth: float
    max_root_depth: float
    canopy_height: float


def acm_gpp_et_config(
    *,
    foliar_nitrogen: float = 1.89,
    min_leaf_water_potential: float = -1.990842,
    root_biomass_half_depth: float = 100.0,
    max_root_depth: float = 2.0,
    canopy_height: float = 9.0,
) -> AcmGppEtConfig:
    """Collect the ACM-GPP-ET settings from the pipeline config.

    These are the five entries of the model's ``pars`` vector. The defaults are
    those of upstream's single-site test, except `min_leaf_water_potential`,
    whose default is the value CARBON_MODEL falls back to when given none.

    Parameters
    ----------
    foliar_nitrogen
        Foliar nitrogen per unit leaf area (grams of nitrogen per square metre).
        One value for every pixel, although it varies with vegetation type and
        site fertility, and GPP is directly proportional to it through the
        nitrogen-use efficiency. A spatially varying run needs this to become a
        per-pixel input.
    min_leaf_water_potential
        Minimum leaf water potential, below which stomata close (megapascals).
    root_biomass_half_depth
        Root biomass at which roots reach half of `max_root_depth` (grams of
        biomass per square metre).
    max_root_depth
        Maximum rooting depth (metres).
    canopy_height
        Canopy height (metres). The drivers are taken to be measured 2 m above
        it.

    Returns
    -------
    AcmGppEtConfig
        The settings, ready to unpack into the model call.
    """
    return AcmGppEtConfig(
        foliar_nitrogen=foliar_nitrogen,
        min_leaf_water_potential=min_leaf_water_potential,
        root_biomass_half_depth=root_biomass_half_depth,
        max_root_depth=max_root_depth,
        canopy_height=canopy_height,
    )


@extract_fields()
@declare_units
@declare_freq
def acm_gpp_et(
    temperature_min_daily: Annotated[DataArray, "degC", DAILY],
    temperature_max_daily: Annotated[DataArray, "degC", DAILY],
    temperature_daily: Annotated[DataArray, "degC", DAILY],
    shortwave_radiation_daily: Annotated[DataArray, "MJ m-2 d-1", DAILY],
    co2_daily: Annotated[DataArray, "ppm", DAILY],
    precipitation_daily: Annotated[DataArray, "mm d-1", DAILY],
    wind_speed_daily: Annotated[DataArray, "m s-1", DAILY],
    vpd_daily: Annotated[DataArray, "Pa", DAILY],
    lai_daily: Annotated[DataArray, "1", DAILY],
    fine_root_carbon_daily: Annotated[DataArray, "g m-2", DAILY],
    latitude: Annotated[DataArray, "degrees_north"],
    clay_content: Annotated[DataArray, "percent"],
    sand_content: Annotated[DataArray, "percent"],
    acm_gpp_et_config: AcmGppEtConfig,
) -> AcmGppEtOut:
    """
    Run the Aggregated Canopy Model for GPP and evapotranspiration.

    Daily resolution input data.

    Parameters
    ----------
    temperature_min_daily
        Daily minimum air temperature (degrees Celsius).
    temperature_max_daily
        Daily maximum air temperature (degrees Celsius).
    temperature_daily
        Daily mean air temperature (degrees Celsius).
    shortwave_radiation_daily
        Daily total incoming shortwave radiation (megajoules per square metre
        per day).
    co2_daily
        Atmospheric CO2 concentration (parts per million).
    precipitation_daily
        Daily precipitation (millimetres per day).
    wind_speed_daily
        Daily mean wind speed (metres per second).
    vpd_daily
        Daily mean vapour pressure deficit (pascals).
    lai_daily
        Leaf area index (square metres of leaf per square metre of ground).
    fine_root_carbon_daily
        Fine root carbon stock (grams of carbon per square metre).
    latitude
        Latitude (degrees north).
    clay_content
        Soil clay content (percent). Either one value per pixel, applied to
        every soil layer, or one per layer along a ``soil_layer`` dimension of
        length 4, top layer first.
    sand_content
        Soil sand content (percent). Shaped as `clay_content`, independently of
        it.
    acm_gpp_et_config
        Model settings; see `acm_gpp_et_config`.

    Returns
    -------
    AcmGppEtOut
        Dictionary of daily outputs; see `AcmGppEtOut` for per-output detail.
    """
    return _acm_gpp_et(
        temperature_min_daily=temperature_min_daily,
        temperature_max_daily=temperature_max_daily,
        temperature_daily=temperature_daily,
        shortwave_radiation_daily=shortwave_radiation_daily,
        co2_daily=co2_daily,
        precipitation_daily=precipitation_daily,
        wind_speed_daily=wind_speed_daily,
        vpd_daily=vpd_daily,
        lai_daily=lai_daily,
        fine_root_carbon_daily=fine_root_carbon_daily,
        latitude=latitude,
        clay_content=clay_content,
        sand_content=sand_content,
        dates_daily=time_index(temperature_daily, "temperature_daily"),
        **acm_gpp_et_config,
    )
