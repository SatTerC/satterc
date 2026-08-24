"""
Satterc-compatable interface to PyRealm's 'P-model'.

This module provides the pmodel node, which wraps the pyrealm P-Model
to calculate gross primary productivity (GPP), light use efficiency (LUE),
and intrinsic water use efficiency (IWUE) from environmental inputs.
"""

from typing import Annotated, TypedDict, cast

import pyrealm.pmodel
import xarray as xr
from conduit import declare_units
from conduit.io import sole_time_dim
from hamilton.function_modifiers import extract_fields
from numpy.typing import NDArray
from xarray import DataArray
from xarray_annotated.temporal import declare_freq

from ..temporal import DAILY, WEEKLY


class PModelOut(TypedDict):
    """Outputs of the `pmodel` node, at weekly resolution."""

    gpp_flux_weekly: Annotated[DataArray, "ug m-2 s-1", WEEKLY]
    """Carbon fixed by photosynthesis, as the instantaneous flux pyrealm reports
    (micrograms of carbon per square metre per second).

    Not a daily rate. A consumer wanting ``g m-2 d-1`` multiplies by 0.0864
    (86400 s d-1 x 1e-6 g ug-1). The example config does that in its
    ``gpp_weekly`` node."""
    lue_photon_weekly: Annotated[DataArray, "g mol-1", WEEKLY]
    """Light use efficiency, as carbon fixed per *mole of absorbed photons*
    (grams of carbon per mole).

    pyrealm defines LUE against PPFD, a photon flux, so the denominator counts
    photons rather than energy. SGAM wants ``g MJ-1`` instead, and the two differ
    by the energy content of PAR, roughly 4.57 mol MJ-1."""
    iwue_weekly: Annotated[DataArray, "umol mol-1", WEEKLY]
    """Intrinsic water use efficiency (micromoles of CO2 per mole of air).

    pyrealm computes ``(5/8)(ca - ci) / P`` with the partial pressures in Pa and
    ``P`` in megapascals, which lands in umol mol-1. That is a mixing ratio, not
    the pressure the units once claimed."""


def _pmodel_block(
    temperature_weekly: NDArray,
    vpd_weekly: NDArray,
    co2_weekly: NDArray,
    pressure_weekly: NDArray,
    fapar_weekly: NDArray,
    ppfd_weekly: NDArray,
    mean_growth_temperature: NDArray,
    aridity_index: NDArray,
    volumetric_water_content_weekly: NDArray,
    *,
    method_optchi: str,
    method_jmaxlim: str,
    method_kphio: str,
    method_arrhenius: str,
) -> tuple[NDArray, NDArray, NDArray]:
    """Run the P-Model on a whole numpy block (element-wise over every cell).

    pyrealm's P-Model is vectorised over the spatial axis, so this kernel runs on the
    full array — or a single dask chunk — in one call; there is no per-pixel loop.
    Returns ``(gpp, lue, iwue)`` arrays matching the input shape, ordered as the fields
    of `PModelOut`, in pyrealm's own units. Nothing is rescaled here. This is the
    unit mapped over the block by `_pmodel` via `xarray.apply_ufunc`.
    """
    # Environmental drivers computed upon instantiation of PModelEnvironment
    env = pyrealm.pmodel.PModelEnvironment(
        tc=temperature_weekly,
        vpd=vpd_weekly,
        co2=co2_weekly,
        patm=pressure_weekly,
        fapar=fapar_weekly,
        ppfd=ppfd_weekly,
        theta=volumetric_water_content_weekly,
        mean_growth_temperature=mean_growth_temperature,
        aridity_index=aridity_index,
    )
    # P-model fit performed upon instantiation of Pmodel
    model = pyrealm.pmodel.PModel(
        env=env,
        method_optchi=method_optchi,
        method_kphio=method_kphio,
        method_arrhenius=method_arrhenius,
        method_jmaxlim=method_jmaxlim,
    )

    return model.gpp, model.lue, model.iwue


def _pmodel(
    temperature_weekly: DataArray,
    vpd_weekly: DataArray,
    co2_weekly: DataArray,
    pressure_weekly: DataArray,
    fapar_weekly: DataArray,
    ppfd_weekly: DataArray,
    mean_growth_temperature: DataArray,
    aridity_index: DataArray,
    volumetric_water_content_weekly: DataArray,
    method_optchi: str,
    method_jmaxlim: str,
    method_kphio: str,
    method_arrhenius: str,
) -> PModelOut:
    """Apply the P-Model to a ``(time, pixel)`` block via `xarray.apply_ufunc`.

    The P-Model is element-wise (every cell independent), so no core dimensions are
    declared and ``apply_ufunc`` broadcasts over both ``time`` and ``pixel``; pyrealm
    still vectorises within each call. ``dask="parallelized"`` is a no-op for eager
    numpy inputs but keeps the node compatible with a future dask-backed (chunked)
    execution strategy.
    """
    # `mean_growth_temperature` and `aridity_index` are climatological: one value
    # per pixel, no time axis. apply_ufunc would hand them to the kernel with a
    # length-1 time axis, and pyrealm's `check_input_shapes` rejects that rather
    # than broadcasting it, so expand them here.
    mean_growth_temperature = mean_growth_temperature.broadcast_like(temperature_weekly)
    aridity_index = aridity_index.broadcast_like(temperature_weekly)

    gpp, lue, iwue = xr.apply_ufunc(
        _pmodel_block,
        temperature_weekly,
        vpd_weekly,
        co2_weekly,
        pressure_weekly,
        fapar_weekly,
        ppfd_weekly,
        mean_growth_temperature,
        aridity_index,
        volumetric_water_content_weekly,
        kwargs={
            "method_optchi": method_optchi,
            "method_jmaxlim": method_jmaxlim,
            "method_kphio": method_kphio,
            "method_arrhenius": method_arrhenius,
        },
        output_core_dims=[[], [], []],
        dask="parallelized",
        output_dtypes=[float, float, float],
    )
    return cast(
        PModelOut,
        {
            "gpp_flux_weekly": gpp,
            "lue_photon_weekly": lue,
            "iwue_weekly": iwue,
        },
    )


@extract_fields()
@declare_units
@declare_freq
def pmodel(
    temperature_weekly: Annotated[DataArray, "degC", WEEKLY],
    vpd_weekly: Annotated[DataArray, "Pa", WEEKLY],
    co2_weekly: Annotated[DataArray, "ppm", WEEKLY],
    pressure_weekly: Annotated[DataArray, "Pa", WEEKLY],
    fapar_weekly: Annotated[DataArray, "1", WEEKLY],
    ppfd_weekly: Annotated[DataArray, "umol m-2 s-1", WEEKLY],
    mean_growth_temperature: Annotated[DataArray, "degC"],
    aridity_index: Annotated[DataArray, "1"],
    volumetric_water_content_weekly: Annotated[DataArray, "m3 m-3", WEEKLY],
    *,
    method_optchi: str = "prentice14",
    method_jmaxlim: str = "wang17",
    method_kphio: str = "temperature",
    method_arrhenius: str = "simple",
) -> PModelOut:
    """Run the P-Model to calculate GPP, LUE, and IWUE.

    Parameters
    ----------
    temperature_weekly
        Air temperature (degrees Celsius).
    vpd_weekly
        Vapour pressure deficit (pascals).
    co2_weekly
        Atmospheric CO2 concentration (parts per million).
    pressure_weekly
        Atmospheric pressure (pascals).
    fapar_weekly
        Fraction of absorbed photosynthetically active radiation
        (dimensionless, 0-1).
    ppfd_weekly
        Photosynthetic photon flux density (micromoles per square metre per
        second).
    mean_growth_temperature
        Mean growth temperature (degrees Celsius).
    aridity_index
        Climatological aridity index, PET/P: potential evapotranspiration over
        precipitation, both accumulated over the record. Values above 1 mean
        demand exceeds supply. Note the orientation — pyrealm wants PET/P, so a
        *higher* value is a *drier* site.
    volumetric_water_content_weekly
        Volumetric soil water content: the fraction of soil volume occupied by
        water (cubic metres of water per cubic metre of soil). This is pyrealm's
        ``theta``, not SPLASH's soil moisture, which is a depth of water in
        millimetres — see the ``volumetric_water_content`` node in the example
        config for the conversion.
    method_optchi
        Method for calculating optimal chi (leaf-internal CO2 compensation
        point).
    method_jmaxlim
        Method for Jmax limitation.
    method_kphio
        Method for calculating the quantum yield efficiency (phi0).
    method_arrhenius
        Method for Arrhenius temperature scaling.

    Returns
    -------
    PModelOut
        Dictionary of weekly outputs:

        - gpp_flux_weekly: gross primary productivity (micrograms of carbon per
          square metre per second)
        - lue_photon_weekly: light use efficiency (grams of carbon per mole of
          absorbed photons)
        - iwue_weekly: intrinsic water use efficiency (micromoles per mole)

        See `PModelOut` for per-output detail.
    """
    return _pmodel(
        temperature_weekly=temperature_weekly,
        vpd_weekly=vpd_weekly,
        co2_weekly=co2_weekly,
        pressure_weekly=pressure_weekly,
        fapar_weekly=fapar_weekly,
        ppfd_weekly=ppfd_weekly,
        mean_growth_temperature=mean_growth_temperature,
        aridity_index=aridity_index,
        volumetric_water_content_weekly=volumetric_water_content_weekly,
        method_optchi=method_optchi,
        method_jmaxlim=method_jmaxlim,
        method_kphio=method_kphio,
        method_arrhenius=method_arrhenius,
    )


@declare_units
def mean_growth_temperature(
    temperature_daily: Annotated[DataArray, "degC", DAILY],
) -> Annotated[DataArray, "degC"]:
    """Mean temperature over the growing days of the record, one value per pixel.

    pyrealm consumes this as an *acclimation* quantity: the ``sandoval`` quantum
    yield method uses it to set the temperature at which the highest kphio is
    attained, and pairs it with `aridity_index`, which pyrealm likewise documents
    as climatological. The instantaneous temperature response is a separate term,
    driven by ``tc``.

    So this reduces over the whole record rather than over a window. Computing it
    weekly (as this node once did) returned NaN for any week in which no day rose
    above 0 degC — an ordinary winter week, not an error — and that NaN
    propagated through kphio into GPP.

    A pixel that never rises above 0 degC over the whole record still yields NaN.
    That is a real gap rather than an artifact of the window, so it is left to
    propagate.
    """
    growing_days = temperature_daily.where(temperature_daily > 0.0)
    return growing_days.mean(sole_time_dim(temperature_daily, "temperature_daily"))
