"""
Satterc-compatable interface to PyRealm's 'P-model'.

This module provides the pmodel node, which wraps the pyrealm P-Model and returns
every output pyrealm declares: gross primary productivity (GPP), light use efficiency
(LUE), intrinsic water use efficiency (IWUE), Vcmax and Jmax at growth and standard
temperature, and stomatal conductance.

Some of pyrealm's methods need inputs beyond the core drivers. Each method axis
that has any such method gets an ``<axis>_inputs`` node, defined once per set of
requirements and selected by ``@config.when`` on the configured method, so a
pipeline only asks for what its chosen methods consume. Each variant returns the
extra inputs keyed by their pyrealm ``PModelEnvironment`` argument names, and
`pmodel` passes the union through.
"""

from collections.abc import Mapping
from typing import Annotated, Any, TypedDict, cast

import pyrealm.pmodel
import xarray as xr
from conduit.io import sole_time_dim
from hamilton.function_modifiers import config, extract_fields
from numpy.typing import NDArray
from pyrealm.pmodel.arrhenius import ARRHENIUS_METHOD_REGISTRY
from pyrealm.pmodel.optimal_chi import OPTIMAL_CHI_CLASS_REGISTRY
from pyrealm.pmodel.quantum_yield import QUANTUM_YIELD_CLASS_REGISTRY
from xarray import DataArray
from xarray_annotated.temporal import declare_freq
from xarray_annotated.units import declare_units

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
    ``P`` in megapascals, so this is a mixing ratio rather than a pressure."""
    vcmax_weekly: Annotated[DataArray, "umol m-2 s-1", WEEKLY]
    """Maximum rate of carboxylation at the growth temperature."""
    vcmax25_weekly: Annotated[DataArray, "umol m-2 s-1", WEEKLY]
    """Maximum rate of carboxylation at 25 degC, scaled from Vcmax by
    ``method_arrhenius``."""
    jmax_weekly: Annotated[DataArray, "umol m-2 s-1", WEEKLY]
    """Maximum rate of electron transport at the growth temperature."""
    jmax25_weekly: Annotated[DataArray, "umol m-2 s-1", WEEKLY]
    """Maximum rate of electron transport at 25 degC, scaled from Jmax by
    ``method_arrhenius``."""
    gs_weekly: Annotated[DataArray, "umol m-2 s-1", WEEKLY]
    """Stomatal conductance, NaN where VPD is 0 or ``ca - ci`` is not positive.

    The units are the ones pyrealm declares. pyrealm computes this as
    ``A_c / (ca - ci)`` with the partial pressures in Pa, which is dimensionally
    also per pascal."""


#: `PModelOut` field -> the `pyrealm.pmodel.PModel` attribute holding it.
_OUTPUT_ATTRIBUTES = {
    "gpp_flux_weekly": "gpp",
    "lue_photon_weekly": "lue",
    "iwue_weekly": "iwue",
    "vcmax_weekly": "vcmax",
    "vcmax25_weekly": "vcmax25",
    "jmax_weekly": "jmax",
    "jmax25_weekly": "jmax25",
    "gs_weekly": "gs",
}


def _pmodel_block(
    temperature_weekly: NDArray,
    vpd_weekly: NDArray,
    co2_weekly: NDArray,
    pressure_weekly: NDArray,
    fapar_weekly: NDArray,
    ppfd_weekly: NDArray,
    *extra_values: NDArray,
    extra_names: tuple[str, ...],
    method_optchi: str,
    method_jmaxlim: str,
    method_kphio: str,
    method_arrhenius: str,
    reference_kphio: float | None = None,
) -> tuple[NDArray, ...]:
    """Run the P-Model on a whole numpy block (element-wise over every cell).

    pyrealm's P-Model is vectorised over the spatial axis, so this kernel runs on the
    full array — or a single dask chunk — in one call; there is no per-pixel loop.
    ``extra_values`` are the method-dependent inputs, passed positionally because
    `xarray.apply_ufunc` only maps over positional arguments, and named by
    ``extra_names`` in the same order.
    Returns one array per entry of `_OUTPUT_ATTRIBUTES`, in that order, matching the
    input shape, in pyrealm's own units. Nothing is rescaled here. This is the unit
    mapped over the block by `_pmodel` via `xarray.apply_ufunc`.
    """
    # Typed `Any` because pyrealm takes these through `**kwargs`, alongside
    # keyword arguments of other types that a `dict[str, NDArray]` would clash with.
    extras: dict[str, Any] = dict(zip(extra_names, extra_values, strict=True))
    # Environmental drivers computed upon instantiation of PModelEnvironment
    env = pyrealm.pmodel.PModelEnvironment(
        tc=temperature_weekly,
        vpd=vpd_weekly,
        co2=co2_weekly,
        patm=pressure_weekly,
        fapar=fapar_weekly,
        ppfd=ppfd_weekly,
        **extras,
    )
    # P-model fit performed upon instantiation of Pmodel
    model = pyrealm.pmodel.PModel(
        env=env,
        method_optchi=method_optchi,
        method_kphio=method_kphio,
        method_arrhenius=method_arrhenius,
        method_jmaxlim=method_jmaxlim,
        reference_kphio=reference_kphio,
    )

    return tuple(getattr(model, name) for name in _OUTPUT_ATTRIBUTES.values())


def _pmodel(
    temperature_weekly: DataArray,
    vpd_weekly: DataArray,
    co2_weekly: DataArray,
    pressure_weekly: DataArray,
    fapar_weekly: DataArray,
    ppfd_weekly: DataArray,
    extra_inputs: dict[str, DataArray],
    method_optchi: str,
    method_jmaxlim: str,
    method_kphio: str,
    method_arrhenius: str,
    reference_kphio: float | None = None,
) -> PModelOut:
    """Apply the P-Model to a ``(time, pixel)`` block via `xarray.apply_ufunc`.

    The P-Model is element-wise (every cell independent), so no core dimensions are
    declared and ``apply_ufunc`` broadcasts over both ``time`` and ``pixel``; pyrealm
    still vectorises within each call. ``dask="parallelized"`` is a no-op for eager
    numpy inputs but keeps the node compatible with a future dask-backed (chunked)
    execution strategy.

    ``extra_inputs`` holds the method-dependent inputs, keyed by pyrealm's
    ``PModelEnvironment`` argument names, as the ``<axis>_inputs`` nodes return
    them.
    """
    # Some extra inputs (`mean_growth_temperature`, `aridity_index`) are
    # climatological: one value per pixel, no time axis. apply_ufunc would hand
    # them to the kernel with a length-1 time axis, and pyrealm's
    # `check_input_shapes` rejects that rather than broadcasting it, so expand them
    # here. Broadcasting an input that already has a time axis is a no-op.
    extra_names = tuple(extra_inputs)
    extra_values = [
        extra_inputs[name].broadcast_like(temperature_weekly) for name in extra_names
    ]

    n_outputs = len(_OUTPUT_ATTRIBUTES)
    outputs = xr.apply_ufunc(
        _pmodel_block,
        temperature_weekly,
        vpd_weekly,
        co2_weekly,
        pressure_weekly,
        fapar_weekly,
        ppfd_weekly,
        *extra_values,
        kwargs={
            "extra_names": extra_names,
            "method_optchi": method_optchi,
            "method_jmaxlim": method_jmaxlim,
            "method_kphio": method_kphio,
            "method_arrhenius": method_arrhenius,
            "reference_kphio": reference_kphio,
        },
        output_core_dims=[[]] * n_outputs,
        dask="parallelized",
        output_dtypes=[float] * n_outputs,
    )
    return cast(PModelOut, dict(zip(_OUTPUT_ATTRIBUTES, outputs, strict=True)))


class PModelConfig(TypedDict):
    """Settings for the `pmodel` node, as returned by `pmodel_config`.

    The descriptions and defaults live on `pmodel_config`, which is what the
    pipeline config populates.
    """

    method_optchi: str
    method_jmaxlim: str
    method_kphio: str
    method_arrhenius: str
    reference_kphio: float | None


def pmodel_config(
    *,
    method_optchi: str = "prentice14",
    method_jmaxlim: str = "wang17",
    method_kphio: str = "temperature",
    method_arrhenius: str = "simple",
    reference_kphio: float | None = None,
) -> PModelConfig:
    """Collect the P-model settings from the pipeline config.

    Grouping the settings into one node keeps them out of `pmodel`'s data
    inputs. It does not namespace them: conduit merges every module section's
    params into one flat driver config, so these are still configured as
    top-level keys of ``[pmodel]``.

    Each setting selects one of pyrealm's alternative formulations; the
    accepted values are pyrealm's, and the
    [pyrealm P-model documentation](https://pyrealm.readthedocs.io/en/latest/users/pmodel/module_overview.html)
    is the authoritative list. pyrealm's own warnings reach you unfiltered: it
    warns that some of these methods are experimental, and recommends ``simple``
    over the other ``method_arrhenius`` choices.

    Parameters
    ----------
    method_optchi
        Formulation for optimal chi, the ratio of leaf-internal to ambient CO2:
        ``prentice14``, ``prentice14_rootzonestress``, ``c4``,
        ``c4_rootzonestress``, ``c4_no_gamma``, ``c4_no_gamma_rootzonestress``,
        ``lavergne20_c3`` or ``lavergne20_c4``. The ``lavergne20`` methods need
        soil moisture and the ``rootzonestress`` methods a root zone stress
        factor; see `optchi_inputs__theta` and `optchi_inputs__rootzonestress`.
    method_jmaxlim
        Jmax limitation: ``wang17``, ``smith19`` or ``none``.
    method_kphio
        Temperature dependence of the quantum yield efficiency (phi0):
        ``temperature``, ``fixed`` or ``sandoval``. ``sandoval`` needs
        climatological inputs; see `kphio_inputs__sandoval`.
    method_arrhenius
        Arrhenius temperature scaling of ``vcmax25_weekly`` and ``jmax25_weekly``,
        the only outputs it affects: ``simple`` or ``kattge_knorr``.
        ``kattge_knorr`` needs the mean growth temperature; see
        `arrhenius_inputs__kattge_knorr`.
    reference_kphio
        Reference value of the quantum yield efficiency, as one number for the
        whole run. Unset, ``method_kphio`` uses pyrealm's default for that method.

    Returns
    -------
    PModelConfig
        The settings, ready to unpack into the model call.
    """
    return PModelConfig(
        method_optchi=method_optchi,
        method_jmaxlim=method_jmaxlim,
        method_kphio=method_kphio,
        method_arrhenius=method_arrhenius,
        reference_kphio=reference_kphio,
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
    optchi_inputs: dict[str, DataArray],
    kphio_inputs: dict[str, DataArray],
    arrhenius_inputs: dict[str, DataArray],
    pmodel_config: PModelConfig,
) -> PModelOut:
    """Run the P-Model, returning every output pyrealm declares.

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
    optchi_inputs
        Extra inputs for the configured ``method_optchi``.
    kphio_inputs
        Extra inputs for the configured ``method_kphio``.
    arrhenius_inputs
        Extra inputs for the configured ``method_arrhenius``.
    pmodel_config
        Model settings; see `pmodel_config`.

    Returns
    -------
    PModelOut
        Dictionary of weekly outputs:

        - gpp_flux_weekly: gross primary productivity (micrograms of carbon per
          square metre per second)
        - lue_photon_weekly: light use efficiency (grams of carbon per mole of
          absorbed photons)
        - iwue_weekly: intrinsic water use efficiency (micromoles per mole)
        - vcmax_weekly, vcmax25_weekly: maximum carboxylation rate at the growth
          temperature and at 25 degC (micromoles per square metre per second)
        - jmax_weekly, jmax25_weekly: maximum electron transport rate at the growth
          temperature and at 25 degC (micromoles per square metre per second)
        - gs_weekly: stomatal conductance

        See `PModelOut` for per-output detail.
    """
    # Two axes can ask for the same input (`mean_growth_temperature`), and then
    # both entries are the same node's value, so the merge loses nothing.
    return _pmodel(
        temperature_weekly=temperature_weekly,
        vpd_weekly=vpd_weekly,
        co2_weekly=co2_weekly,
        pressure_weekly=pressure_weekly,
        fapar_weekly=fapar_weekly,
        ppfd_weekly=ppfd_weekly,
        extra_inputs={**optchi_inputs, **kphio_inputs, **arrhenius_inputs},
        **pmodel_config,
    )


def _methods_requiring(
    registry: Mapping[str, type], attribute: str, *requirements: str
) -> tuple[str, ...]:
    """Select the methods in a pyrealm registry declaring exactly these requirements."""
    # No getattr default: if pyrealm renames the attribute, fail on import rather
    # than read every method as needing nothing.
    return tuple(
        method
        for method, cls in registry.items()
        if getattr(cls, attribute) == list(requirements)
    )


# Methods that need inputs beyond the core drivers, grouped by what they need, as
# pyrealm declares it. A method in none of these gets the empty
# `<axis>_inputs__none` variant, and so does an unset key: the config's defaults live
# on `pmodel_config`, which `config.when` cannot see, so this only holds while every
# default method needs nothing extra. `tests/test_pmodel_methods.py` checks that
# invariant, and fails on a requirement set that has no variant here.
_OPTCHI_THETA = _methods_requiring(OPTIMAL_CHI_CLASS_REGISTRY, "requires", "theta")
_OPTCHI_ROOTZONESTRESS = _methods_requiring(
    OPTIMAL_CHI_CLASS_REGISTRY, "requires", "rootzonestress"
)
_KPHIO_SANDOVAL = _methods_requiring(
    QUANTUM_YIELD_CLASS_REGISTRY,
    "requires",
    "aridity_index",
    "mean_growth_temperature",
)
_ARRHENIUS_KATTGE_KNORR = _methods_requiring(
    ARRHENIUS_METHOD_REGISTRY, "required_env_variables", "mean_growth_temperature"
)


@config.when_not_in(method_optchi=[*_OPTCHI_THETA, *_OPTCHI_ROOTZONESTRESS])
def optchi_inputs__none() -> dict[str, DataArray]:
    """No extra inputs, for the ``method_optchi`` choices that need none."""
    return {}


@config.when_in(method_optchi=list(_OPTCHI_THETA))
@declare_units
@declare_freq
def optchi_inputs__theta(
    volumetric_water_content_weekly: Annotated[DataArray, "m3 m-3", WEEKLY],
) -> dict[str, DataArray]:
    """Soil moisture, for ``method_optchi = "lavergne20_c3"`` or ``"lavergne20_c4"``.

    Parameters
    ----------
    volumetric_water_content_weekly
        Volumetric soil water content: the fraction of soil volume occupied by
        water (cubic metres of water per cubic metre of soil). This is pyrealm's
        ``theta``, not SPLASH's soil moisture, which is a depth of water in
        millimetres — see the ``volumetric_water_content`` node in the example
        config for the conversion.
    """
    return {"theta": volumetric_water_content_weekly}


@config.when_in(method_optchi=list(_OPTCHI_ROOTZONESTRESS))
@declare_units
@declare_freq
def optchi_inputs__rootzonestress(
    rootzonestress_weekly: Annotated[DataArray, "1", WEEKLY],
) -> dict[str, DataArray]:
    """Root zone stress, for the ``*_rootzonestress`` ``method_optchi`` choices.

    Parameters
    ----------
    rootzonestress_weekly
        Root zone stress factor (dimensionless, 0-1), which penalises the
        cost ratio in the optimal chi calculation. Experimental in pyrealm.
    """
    return {"rootzonestress": rootzonestress_weekly}


@config.when_not_in(method_kphio=list(_KPHIO_SANDOVAL))
def kphio_inputs__none() -> dict[str, DataArray]:
    """No extra inputs, for the ``method_kphio`` choices that need none."""
    return {}


@config.when_in(method_kphio=list(_KPHIO_SANDOVAL))
@declare_units
def kphio_inputs__sandoval(
    aridity_index: Annotated[DataArray, "1"],
    mean_growth_temperature: Annotated[DataArray, "degC"],
) -> dict[str, DataArray]:
    """Climatological inputs, for ``method_kphio = "sandoval"``.

    Parameters
    ----------
    aridity_index
        Climatological aridity index, PET/P: potential evapotranspiration over
        precipitation, both accumulated over the record. Values above 1 mean
        demand exceeds supply. Note the orientation — pyrealm wants PET/P, so a
        *higher* value is a *drier* site.
    mean_growth_temperature
        Mean growth temperature (degrees Celsius); see
        `mean_growth_temperature`.
    """
    return {
        "aridity_index": aridity_index,
        "mean_growth_temperature": mean_growth_temperature,
    }


@config.when_not_in(method_arrhenius=list(_ARRHENIUS_KATTGE_KNORR))
def arrhenius_inputs__none() -> dict[str, DataArray]:
    """No extra inputs, for the ``method_arrhenius`` choices that need none."""
    return {}


@config.when_in(method_arrhenius=list(_ARRHENIUS_KATTGE_KNORR))
@declare_units
def arrhenius_inputs__kattge_knorr(
    mean_growth_temperature: Annotated[DataArray, "degC"],
) -> dict[str, DataArray]:
    """Mean growth temperature, for ``method_arrhenius = "kattge_knorr"``.

    Parameters
    ----------
    mean_growth_temperature
        Mean growth temperature (degrees Celsius); see
        `mean_growth_temperature`.
    """
    return {"mean_growth_temperature": mean_growth_temperature}


@config(
    lambda cfg: (
        cfg.get("method_kphio") in _KPHIO_SANDOVAL
        or cfg.get("method_arrhenius") in _ARRHENIUS_KATTGE_KNORR
    ),
    config_used=["method_kphio", "method_arrhenius"],
)
@declare_units
def mean_growth_temperature(
    temperature_daily: Annotated[DataArray, "degC", DAILY],
) -> Annotated[DataArray, "degC"]:
    """Mean temperature over the growing days of the record, one value per pixel.

    pyrealm consumes this as an *acclimation* quantity: the ``sandoval`` quantum
    yield method uses it to set the temperature at which the highest kphio is
    attained, and the ``kattge_knorr`` Arrhenius method to set the activation
    entropy of its temperature scaling. The instantaneous temperature response is a
    separate term, driven by ``tc``.

    The mean is taken over the whole record, not a rolling window, and over days
    above 0 degC only. A pixel that never rises above 0 degC over the record
    yields NaN. Under ``sandoval`` that propagates through kphio into GPP; under
    ``kattge_knorr`` it reaches only ``vcmax25`` and ``jmax25``.

    Parameters
    ----------
    temperature_daily
        Daily mean air temperature (degrees Celsius).
    """
    # Reduce over the whole record, not per week: a winter week with no day above
    # 0 degC is an ordinary week, and windowing turned it into a NaN in GPP.
    growing_days = temperature_daily.where(temperature_daily > 0.0)
    return growing_days.mean(sole_time_dim(temperature_daily, "temperature_daily"))
