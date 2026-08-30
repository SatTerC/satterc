"""Simplified Growth and Allocation Model (SGAM) interface for the SatTerC pipeline."""

from typing import Annotated, TypedDict, cast

import numpy as np
import pandas as pd
import xarray as xr
from hamilton.function_modifiers import extract_fields
from numpy.typing import NDArray
from sgam import Disturbances, Sgam
from sgam.pft import PftParams, PlantFunctionalType, get_default_pft_params
from xarray_annotated.temporal import declare_freq
from xarray_annotated.units import declare_units

from ..temporal import DAILY, WEEKLY, time_index

# SGAM output node names, in the order they are returned by `_sgam_1px` and mapped
# onto the DAG node names below.
_SGAM_OUTPUT_NAMES: tuple[str, ...] = (
    "leaf_pool_weekly",
    "stem_pool_weekly",
    "root_pool_weekly",
    "litter_pool_weekly",
    "removed_pool_weekly",
    "npp_leaf_weekly",
    "npp_stem_weekly",
    "npp_root_weekly",
    "turnover_leaf_weekly",
    "turnover_stem_weekly",
    "turnover_root_weekly",
    "respiration_leaf_weekly",
    "respiration_stem_weekly",
    "respiration_root_weekly",
    "disturbance_leaf_weekly",
    "disturbance_stem_weekly",
    "disturbance_root_weekly",
    "cue_weekly",
    "allocation_leaf_weekly",
    "allocation_stem_weekly",
    "allocation_root_weekly",
    "drought_modifier_weekly",
    "lue_score_weekly",
    "iwue_score_weekly",
)


class SgamOut(TypedDict):
    """Outputs of the `sgam` node, at weekly resolution.

    Carbon pools are standing stocks; the npp/turnover/respiration/disturbance
    quantities are weekly fluxes recorded as the carbon amount per weekly
    timestep. Both pools and fluxes are carbon mass per unit ground area
    (grams of carbon per square metre); the remaining diagnostics are
    dimensionless.
    """

    leaf_pool_weekly: Annotated[xr.DataArray, "g m-2", WEEKLY]
    """Standing leaf carbon pool (grams of carbon per square metre)."""
    stem_pool_weekly: Annotated[xr.DataArray, "g m-2", WEEKLY]
    """Standing stem carbon pool (grams of carbon per square metre)."""
    root_pool_weekly: Annotated[xr.DataArray, "g m-2", WEEKLY]
    """Standing root carbon pool (grams of carbon per square metre)."""
    litter_pool_weekly: Annotated[xr.DataArray, "g m-2", WEEKLY]
    """Litter carbon pool, fed by turnover and (for non-crop disturbance)
    disturbance losses; accumulate-only, since decomposition is RothC's role
    (grams of carbon per square metre)."""
    removed_pool_weekly: Annotated[xr.DataArray, "g m-2", WEEKLY]
    """Cumulative carbon removed from the system by disturbance/harvest
    (grams of carbon per square metre)."""
    npp_leaf_weekly: Annotated[xr.DataArray, "g m-2", WEEKLY]
    """Net primary productivity directed to the leaf pool, as the weekly growth
    flux (grams of carbon per square metre per week)."""
    npp_stem_weekly: Annotated[xr.DataArray, "g m-2", WEEKLY]
    """Net primary productivity directed to the stem pool, as the weekly growth
    flux (grams of carbon per square metre per week)."""
    npp_root_weekly: Annotated[xr.DataArray, "g m-2", WEEKLY]
    """Net primary productivity directed to the root pool, as the weekly growth
    flux (grams of carbon per square metre per week)."""
    turnover_leaf_weekly: Annotated[xr.DataArray, "g m-2", WEEKLY]
    """Leaf litterfall flux to the litter pool (grams of carbon per square metre
    per week)."""
    turnover_stem_weekly: Annotated[xr.DataArray, "g m-2", WEEKLY]
    """Stem litterfall flux to the litter pool (grams of carbon per square metre
    per week)."""
    turnover_root_weekly: Annotated[xr.DataArray, "g m-2", WEEKLY]
    """Root litterfall flux to the litter pool (grams of carbon per square metre
    per week)."""
    respiration_leaf_weekly: Annotated[xr.DataArray, "g m-2", WEEKLY]
    """Autotrophic respiration attributed to the leaf pool (grams of carbon per
    square metre per week)."""
    respiration_stem_weekly: Annotated[xr.DataArray, "g m-2", WEEKLY]
    """Autotrophic respiration attributed to the stem pool (grams of carbon per
    square metre per week)."""
    respiration_root_weekly: Annotated[xr.DataArray, "g m-2", WEEKLY]
    """Autotrophic respiration attributed to the root pool (grams of carbon per
    square metre per week)."""
    disturbance_leaf_weekly: Annotated[xr.DataArray, "g m-2", WEEKLY]
    """Carbon lost from the leaf pool to disturbance, as a positive flux. For
    non-crop PFTs this transfers to litter; for crops it transfers to the
    removed pool (grams of carbon per square metre per week)."""
    disturbance_stem_weekly: Annotated[xr.DataArray, "g m-2", WEEKLY]
    """Carbon lost from the stem pool to disturbance, as a positive flux.
    Non-zero only for crops, where it transfers to the removed pool (grams of
    carbon per square metre per week)."""
    disturbance_root_weekly: Annotated[xr.DataArray, "g m-2", WEEKLY]
    """Carbon lost from the root pool to disturbance, as a positive flux.
    Non-zero only for crops, where it transfers to litter (grams of carbon per
    square metre per week)."""
    cue_weekly: Annotated[xr.DataArray, "1", WEEKLY]
    """Carbon use efficiency: the fraction of GPP retained as biomass, in
    [0.2, 0.7] (dimensionless)."""
    allocation_leaf_weekly: Annotated[xr.DataArray, "1", WEEKLY]
    """Fraction of NPP allocated to the leaf pool, in (0, 1). The three
    allocation fractions sum to 1 at every timestep (dimensionless)."""
    allocation_stem_weekly: Annotated[xr.DataArray, "1", WEEKLY]
    """Fraction of NPP allocated to the stem pool, in (0, 1). The three
    allocation fractions sum to 1 at every timestep (dimensionless)."""
    allocation_root_weekly: Annotated[xr.DataArray, "1", WEEKLY]
    """Fraction of NPP allocated to the root pool, in (0, 1). The three
    allocation fractions sum to 1 at every timestep (dimensionless)."""
    drought_modifier_weekly: Annotated[xr.DataArray, "1", WEEKLY]
    """Combined drought stress scalar in [0, 1] (1.0 = no stress, 0.0 = maximum
    stress) (dimensionless)."""
    lue_score_weekly: Annotated[xr.DataArray, "1", WEEKLY]
    """Light use efficiency relative to its PFT-specific maximum, clipped to
    [0, 1] (dimensionless)."""
    iwue_score_weekly: Annotated[xr.DataArray, "1", WEEKLY]
    """Intrinsic water use efficiency relative to its PFT-specific maximum,
    clipped to [0, 1] (dimensionless)."""


def _pft_int_to_enum(value: int) -> PlantFunctionalType:
    """Map a ``plant_type`` integer onto `sgam.pft.PlantFunctionalType`.

    The integer is a 0-based index into the enum's declaration order
    (0=tree, 1=grass, 2=shrub, 3=crop).
    """
    members = list(PlantFunctionalType)
    if not 0 <= value < len(members):
        raise ValueError(
            f"plant_type {value} is not a valid plant functional type: "
            f"expected an integer in [0, {len(members) - 1}] "
            f"({', '.join(f'{i}={m.name.lower()}' for i, m in enumerate(members))})."
        )
    return members[value]


def _build_pft_params_dataset(plant_type: xr.DataArray) -> xr.Dataset:
    field_names = [
        "leaf_base_allocation",
        "stem_base_allocation",
        "root_base_allocation",
        "leaf_turnover_rate",
        "stem_turnover_rate",
        "root_turnover_rate",
        "lue_max",
        "iwue_max",
        "disturbance_threshold",
        "disturbance_leaf_loss_frac",
        "leaf_carbon_area",
        "wilting_point",
        "field_capacity",
        "vpd_threshold",
        "vpd_sensitivity",
        "temp_optimum",
        "temp_sensitivity",
    ]

    pft_vars: dict[str, xr.DataArray] = {}
    for field_name in field_names:
        values = []
        for pft_int in plant_type.values:
            pft_enum = _pft_int_to_enum(int(pft_int))
            params = get_default_pft_params(pft_enum)
            values.append(getattr(params, field_name))
        pft_vars[field_name] = xr.DataArray(data=np.array(values), dims=["pixel"])

    return xr.Dataset(pft_vars)


def _pft_params_from_dataset(ds: xr.Dataset, pixel_idx: int) -> PftParams:
    return PftParams(
        leaf_base_allocation=ds["leaf_base_allocation"].values[pixel_idx],
        stem_base_allocation=ds["stem_base_allocation"].values[pixel_idx],
        root_base_allocation=ds["root_base_allocation"].values[pixel_idx],
        leaf_turnover_rate=ds["leaf_turnover_rate"].values[pixel_idx],
        stem_turnover_rate=ds["stem_turnover_rate"].values[pixel_idx],
        root_turnover_rate=ds["root_turnover_rate"].values[pixel_idx],
        lue_max=ds["lue_max"].values[pixel_idx],
        iwue_max=ds["iwue_max"].values[pixel_idx],
        disturbance_threshold=ds["disturbance_threshold"].values[pixel_idx],
        disturbance_leaf_loss_frac=ds["disturbance_leaf_loss_frac"].values[pixel_idx],
        leaf_carbon_area=ds["leaf_carbon_area"].values[pixel_idx],
        wilting_point=ds["wilting_point"].values[pixel_idx],
        field_capacity=ds["field_capacity"].values[pixel_idx],
        vpd_threshold=ds["vpd_threshold"].values[pixel_idx],
        vpd_sensitivity=ds["vpd_sensitivity"].values[pixel_idx],
        temp_optimum=ds["temp_optimum"].values[pixel_idx],
        temp_sensitivity=ds["temp_sensitivity"].values[pixel_idx],
    )


def pft_params(plant_type: xr.DataArray) -> xr.Dataset:
    """Get PFT parameters for each pixel based on plant_type.

    Parameters
    ----------
    plant_type
        Plant functional type as an integer (0=tree, 1=grass, 2=shrub, 3=crop).
        Dims: ["pixel"].

    Returns
    -------
    xr.Dataset
        Dataset with dimension (pixel) containing PFT parameters for each pixel.
    """
    return _build_pft_params_dataset(plant_type)


def _disturbances_block(
    temperature: NDArray[np.float64],
    gpp: NDArray[np.float64],
    lai: NDArray[np.float64],
    disturbance_threshold: NDArray[np.float64],
    *,
    growing_season_limit: float,
) -> NDArray[np.float64]:
    """Run disturbance detection on a whole pixel-block (vectorised over pixels).

    ``apply_ufunc`` places the ``time`` core dim last, so each time series arrives
    as ``(pixel, time)`` (or ``(time,)`` with no pixel broadcast) and the per-pixel
    ``disturbance_threshold`` as ``(pixel,)`` (or a 0-d scalar).
    `Disturbances.forward` diffs GPP/LAI along axis 0, so ``time`` is moved to
    the front for the call and the result moved back to ``(pixel, time)``.
    ``moveaxis`` is a no-op on a 1D ``(time,)`` array, so the single-pixel case is
    handled too.

    With ``time`` leading, the time series are ``(time, pixel)`` and the threshold
    ``(pixel,)`` broadcasts against them along the trailing axis, so the whole
    block is still one call: `Disturbances.forward` only ever uses
    ``disturbance_threshold`` in an elementwise numpy comparison, so an array of
    per-pixel thresholds works exactly like a scalar and no ``vectorize=True``
    map over pixels is needed.
    """
    temp = np.moveaxis(np.asarray(temperature, dtype=float), -1, 0)
    g = np.moveaxis(np.asarray(gpp, dtype=float), -1, 0)
    la = np.moveaxis(np.asarray(lai, dtype=float), -1, 0)
    detector = Disturbances(
        growing_season_limit=growing_season_limit,
        # Per-pixel array rather than the declared scalar; see the docstring.
        disturbance_threshold=np.asarray(disturbance_threshold, dtype=float),  # type: ignore[reportArgumentType]
    )
    result = detector.forward(temp, g, la, aggregate=False)
    return np.moveaxis(result, 0, -1)


def _disturbances_daily(
    temperature_daily: xr.DataArray,
    gpp_daily: xr.DataArray,
    lai_daily: xr.DataArray,
    disturbance_threshold: xr.DataArray,
    growing_season_limit: float,
) -> xr.DataArray:
    """Apply disturbance detection over the ``(time, pixel)`` block via apply_ufunc.

    Disturbance detection diffs along ``time`` (so ``time`` is the input/output core
    dim) but is otherwise element-wise over ``pixel`` (the broadcast/mapped dim).
    ``disturbance_threshold`` is a per-pixel ``(pixel,)`` input with no core dim, so
    it broadcasts alongside the time series.
    ``vectorize`` is left ``False`` so the whole pixel-block reaches the numpy kernel
    in one call; ``dask="parallelized"`` keeps a future dask-backed (chunked-``pixel``)
    run reachable.
    """
    out = xr.apply_ufunc(
        _disturbances_block,
        temperature_daily,
        gpp_daily,
        lai_daily,
        disturbance_threshold,
        input_core_dims=[["time"]] * 3 + [[]],
        output_core_dims=[["time"]],
        kwargs={"growing_season_limit": growing_season_limit},
        dask="parallelized",
        output_dtypes=[float],
    )

    # apply_ufunc drops the `time` coord (a core dim) and orders the output as
    # (pixel, time); reattach the coord and restore the canonical (time, pixel).
    time_coord = temperature_daily.coords["time"]
    return out.assign_coords(time=time_coord).transpose("time", "pixel")


class DisturbancesConfig(TypedDict):
    """Settings for `disturbances_daily`, as returned by `disturbances_config`.

    The descriptions and defaults live on `disturbances_config`, which is what
    the pipeline config populates.
    """

    growing_season_limit: float


def disturbances_config(
    *,
    growing_season_limit: float = 10.0,
) -> DisturbancesConfig:
    """Collect the disturbance-detection settings from the pipeline config.

    Grouping the settings into one node keeps them out of
    `disturbances_daily`'s data inputs. It does not namespace them: conduit
    merges every module section's params into one flat driver config, so these
    are still configured as top-level keys of ``[sgam]``.

    Parameters
    ----------
    growing_season_limit
        Minimum daily mean temperature for a day to count as within the growing
        season (degrees Celsius). This is a temperature gate, not a calendar
        one, so it needs no hemisphere correction: the seasonal phase is already
        carried by ``temperature_daily``.

    Returns
    -------
    DisturbancesConfig
        The settings, ready to unpack into the detection call.
    """
    return DisturbancesConfig(growing_season_limit=growing_season_limit)


@declare_units
@declare_freq
def disturbances_daily(
    temperature_daily: Annotated[xr.DataArray, "degC", DAILY],
    gpp_daily: Annotated[xr.DataArray, "g m-2 d-1", DAILY],
    lai_daily: Annotated[xr.DataArray, "1", DAILY],
    pft_params: xr.Dataset,
    disturbances_config: DisturbancesConfig,
) -> Annotated[xr.DataArray, "1", DAILY]:
    """Detect daily disturbance events from anomalous declines in GPP and LAI.

    A disturbance is flagged on a day that is (a) within the growing season, i.e.
    warmer than ``growing_season_limit``, and (b) shows a relative decline in
    *both* GPP and LAI larger than the PFT's ``disturbance_threshold``. The
    threshold is taken per pixel from the ``pft_params`` node, so crops (0.1)
    register far more events than trees (0.3), with shrubs (0.25) and grass
    (0.2) in between.

    Parameters
    ----------
    temperature_daily
        Daily mean air temperature (degrees Celsius).
    gpp_daily
        Daily gross primary productivity (grams of carbon per square metre per
        day).
    lai_daily
        Daily leaf area index (dimensionless).
    pft_params
        PFT parameters for each pixel. Output of the `pft_params` node; only
        ``disturbance_threshold`` is used here.
    disturbances_config
        Detection settings; see `disturbances_config`.

    Returns
    -------
    xr.DataArray
        Daily disturbance severity: the relative decline in GPP/LAI used to flag
        a disturbance, in [0, 1] where 0 is no disturbance and 1 is a total loss
        (dimensionless).
    """
    return _disturbances_daily(
        temperature_daily,
        gpp_daily,
        lai_daily,
        pft_params["disturbance_threshold"],
        **disturbances_config,
    )


def _sgam_1px(
    temperature: NDArray[np.float64],
    gpp: NDArray[np.float64],
    volumetric_water_content: NDArray[np.float64],
    vpd: NDArray[np.float64],
    lue: NDArray[np.float64],
    iwue: NDArray[np.float64],
    disturbances: NDArray[np.float64],
    plant_type: int,
    pft_params: PftParams,
    latitude: float,
    leaf_init: float,
    stem_init: float,
    root_init: float,
    litter_init: float,
    removed_init: float,
    *,
    week_of_year: NDArray[np.int_],
    use_dynamic_allocation: bool,
    strict_mass_balance: bool,
) -> tuple[NDArray[np.float64], ...]:
    """Run SGAM for a single pixel.

    The climate/driver arguments are 1D ``(time,)`` arrays for one pixel;
    ``plant_type``, ``latitude`` and the init pools are per-pixel scalars and
    ``pft_params`` is the per-pixel `PftParams` object (threaded
    through ``apply_ufunc`` as one element of an object-dtype ``(pixel,)`` array).
    ``week_of_year`` depends only on the date range, so it is computed once in
    `_sgam` and passed through unchanged. Returns one ``(time,)`` array per
    output, ordered as `_SGAM_OUTPUT_NAMES`. This is the per-pixel kernel mapped
    over ``pixel`` by `_sgam` via `xarray.apply_ufunc`.
    """
    output = Sgam(
        plant_type=_pft_int_to_enum(int(plant_type)),
        pft_params=pft_params,
        use_dynamic_allocation=use_dynamic_allocation,
        hemisphere="NH" if latitude >= 0 else "SH",
    )(
        gpp=gpp,
        temperature=temperature,
        soil_moisture=volumetric_water_content,
        vpd=vpd,
        lue=lue,
        iwue=iwue,
        week_of_year=week_of_year,  # type: ignore[reportArgumentType]  # int weeks ok
        disturbances=disturbances,
        leaf_pool_init=float(leaf_init),
        stem_pool_init=float(stem_init),
        root_pool_init=float(root_init),
        litter_pool_init=float(litter_init),
        removed_init=float(removed_init),
        strict_mass_balance=strict_mass_balance,
    )
    return tuple(
        np.asarray(v, dtype=float)
        for v in (
            output.pools.leaf,
            output.pools.stem,
            output.pools.root,
            output.pools.litter,
            output.pools.removed,
            output.npp.leaf,
            output.npp.stem,
            output.npp.root,
            output.turnover.leaf,
            output.turnover.stem,
            output.turnover.root,
            output.respiration.leaf,
            output.respiration.stem,
            output.respiration.root,
            output.disturbance.leaf,
            output.disturbance.stem,
            output.disturbance.root,
            output.diagnostics.cue,
            output.diagnostics.allocation_leaf,
            output.diagnostics.allocation_stem,
            output.diagnostics.allocation_root,
            output.diagnostics.drought_modifier,
            output.diagnostics.lue_score,
            output.diagnostics.iwue_score,
        )
    )


def _sgam(
    plant_type: xr.DataArray,
    pft_params: xr.Dataset,
    temperature_weekly: xr.DataArray,
    gpp_weekly: xr.DataArray,
    volumetric_water_content_weekly: xr.DataArray,
    vpd_weekly: xr.DataArray,
    lue_weekly: xr.DataArray,
    iwue_weekly: xr.DataArray,
    dates_weekly: pd.Index,
    disturbances_weekly: xr.DataArray,
    leaf_pool_init: xr.DataArray,
    stem_pool_init: xr.DataArray,
    root_pool_init: xr.DataArray,
    latitude: xr.DataArray,
    litter_pool_init: xr.DataArray | None = None,
    removed_init: xr.DataArray | None = None,
    use_dynamic_allocation: bool = True,
    strict_mass_balance: bool = False,
) -> SgamOut:
    """Map `_sgam_1px` over the stacked ``pixel`` dimension.

    The per-pixel SGAM kernel is applied via `xarray.apply_ufunc` with ``time`` as
    the input/output core dimension and ``pixel`` as the broadcast (mapped) dim. The 2D
    ``(time, pixel)`` climate/driver inputs declare ``time`` as their core dim; the 1D
    ``(pixel,)`` metadata and init-pool inputs declare no core dim (so each call gets a
    per-pixel scalar). The structured per-pixel `PftParams` are passed
    as one object-dtype ``(pixel,)`` array. ``week_of_year`` and the boolean flags are
    pixel-invariant constants passed through ``kwargs``. ``dask="parallelized"`` is a
    no-op for eager numpy inputs but keeps the node compatible with a future dask-backed
    (chunked-``pixel``) execution strategy.
    """
    # Week index, from 1-52 — depends only on the date range, so compute once and share.
    # isocalendar() exists on the DatetimeIndex passed at runtime but is missing from
    # pandas Index type stubs, hence the type: ignore.
    week_of_year = dates_weekly.isocalendar().week.values  # type: ignore[reportAttributeAccessIssue]

    # Pack the per-pixel PftParams into a single object-dtype (pixel,) array so each
    # apply_ufunc call receives one PftParams without exploding its 17 fields into args.
    n_pixels = pft_params.sizes["pixel"]
    pft_objs = xr.DataArray(
        np.array(
            [_pft_params_from_dataset(pft_params, i) for i in range(n_pixels)],
            dtype=object,
        ),
        dims=["pixel"],
    )
    # Under a dask-backed (chunked-pixel) run, this object-dtype array must be
    # explicitly chunked to match the inputs' pixel chunking: dask cannot auto-estimate
    # the byte size of object dtype, so an unchunked object array triggers an
    # auto-rechunk error inside apply_ufunc. Mirror the pixel chunks of the
    # (dask-backed) reference input; stay eager numpy otherwise (a no-op when eager).
    if temperature_weekly.chunks is not None:
        pft_objs = pft_objs.chunk({"pixel": temperature_weekly.chunksizes["pixel"]})

    # apply_ufunc inputs must be real DataArrays; substitute zeros for omitted pools.
    if litter_pool_init is None:
        litter_pool_init = xr.zeros_like(leaf_pool_init)
    if removed_init is None:
        removed_init = xr.zeros_like(leaf_pool_init)

    outputs = xr.apply_ufunc(
        _sgam_1px,
        temperature_weekly,
        gpp_weekly,
        volumetric_water_content_weekly,
        vpd_weekly,
        lue_weekly,
        iwue_weekly,
        disturbances_weekly,
        plant_type,
        pft_objs,
        latitude,
        leaf_pool_init,
        stem_pool_init,
        root_pool_init,
        litter_pool_init,
        removed_init,
        input_core_dims=[["time"]] * 7 + [[]] * 8,
        output_core_dims=[["time"]] * 24,
        kwargs={
            "week_of_year": week_of_year,
            "use_dynamic_allocation": use_dynamic_allocation,
            "strict_mass_balance": strict_mass_balance,
        },
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float] * 24,
    )

    # apply_ufunc drops the `time` coordinate (a core dim) and orders outputs as
    # (pixel, time); reattach the coordinate and restore the canonical (time, pixel).
    time_coord = temperature_weekly.coords["time"]
    return cast(
        SgamOut,
        {
            name: da.assign_coords(time=time_coord).transpose("time", "pixel")
            for name, da in zip(_SGAM_OUTPUT_NAMES, outputs, strict=True)
        },
    )


class SgamConfig(TypedDict):
    """Settings for the `sgam` node, as returned by `sgam_config`.

    The descriptions and defaults live on `sgam_config`, which is what the
    pipeline config populates.
    """

    use_dynamic_allocation: bool
    strict_mass_balance: bool


def sgam_config(
    *,
    use_dynamic_allocation: bool = True,
    strict_mass_balance: bool = False,
) -> SgamConfig:
    """Collect the SGAM settings from the pipeline config.

    Grouping the settings into one node keeps them out of `sgam`'s data inputs.
    It does not namespace them: conduit merges every module section's params
    into one flat driver config, so these are still configured as top-level keys
    of ``[sgam]``.

    Parameters
    ----------
    use_dynamic_allocation
        Whether allocation fractions vary with environmental conditions. When
        False, the PFT's fixed base allocations are used instead.
    strict_mass_balance
        Whether a mass balance violation raises `RuntimeError`. When False, it
        issues a warning instead.

    Returns
    -------
    SgamConfig
        The settings, ready to unpack into the model call.
    """
    return SgamConfig(
        use_dynamic_allocation=use_dynamic_allocation,
        strict_mass_balance=strict_mass_balance,
    )


@extract_fields()
@declare_units
@declare_freq
def sgam(
    plant_type: xr.DataArray,
    pft_params: xr.Dataset,
    temperature_weekly: Annotated[xr.DataArray, "degC", WEEKLY],
    gpp_weekly: Annotated[xr.DataArray, "g m-2 d-1", WEEKLY],
    volumetric_water_content_weekly: Annotated[xr.DataArray, "m3 m-3", WEEKLY],
    vpd_weekly: Annotated[xr.DataArray, "Pa", WEEKLY],
    lue_weekly: Annotated[xr.DataArray, "g MJ-1", WEEKLY],
    iwue_weekly: Annotated[xr.DataArray, "umol mol-1", WEEKLY],
    disturbances_weekly: xr.DataArray,
    leaf_pool_init: Annotated[xr.DataArray, "g m-2"],
    stem_pool_init: Annotated[xr.DataArray, "g m-2"],
    root_pool_init: Annotated[xr.DataArray, "g m-2"],
    latitude: xr.DataArray,
    sgam_config: SgamConfig,
    litter_pool_init: xr.DataArray | None = None,
    removed_init: xr.DataArray | None = None,
) -> SgamOut:
    """Run the Simplified Growth and Allocation Model (SGAM) vegetation model.

    Parameters
    ----------
    plant_type
        Plant functional type as an integer (0=tree, 1=grass, 2=shrub, 3=crop).
    pft_params
        PFT parameters for each pixel. Output of the `pft_params` node.
    temperature_weekly
        Weekly mean air temperature (degrees Celsius).
    gpp_weekly
        Weekly gross primary productivity (grams of carbon per square metre per
        day).
    volumetric_water_content_weekly
        Weekly mean volumetric soil water content: the fraction of soil volume
        occupied by water (cubic metres of water per cubic metre of soil). SGAM
        compares this against the PFT's ``wilting_point`` and ``field_capacity``,
        which are volume fractions, so it is *not* SPLASH's soil moisture — that
        is a depth of water in millimetres. See the
        ``volumetric_water_content_weekly`` node in the example config for the
        conversion.
    vpd_weekly
        Weekly mean vapour pressure deficit (pascals).
    lue_weekly
        Weekly mean light use efficiency (grams of carbon per megajoule).
    iwue_weekly
        Weekly mean intrinsic water use efficiency (micromoles per mole).
    disturbances_weekly
        Weekly disturbance severity: the maximum daily relative decline observed
        during the week, in [0, 1] (dimensionless).
    leaf_pool_init
        Initial leaf carbon pool (grams of carbon per square metre).
    stem_pool_init
        Initial stem carbon pool (grams of carbon per square metre).
    root_pool_init
        Initial root carbon pool (grams of carbon per square metre).
    latitude
        Latitude for each pixel (degrees; used to determine hemisphere).
    sgam_config
        Model settings; see `sgam_config`.
    litter_pool_init
        Initial litter carbon pool (grams of carbon per square metre). Defaults
        to zero.
    removed_init
        Initial removed-carbon pool (grams of carbon per square metre). Defaults
        to zero.

    Returns
    -------
    SgamOut
        Dictionary of weekly carbon pools, fluxes (NPP, turnover, respiration,
        disturbance losses), and dimensionless diagnostics. See `SgamOut`
        for the full list of outputs and their units.
    """
    return _sgam(
        plant_type=plant_type,
        pft_params=pft_params,
        temperature_weekly=temperature_weekly,
        gpp_weekly=gpp_weekly,
        volumetric_water_content_weekly=volumetric_water_content_weekly,
        vpd_weekly=vpd_weekly,
        lue_weekly=lue_weekly,
        iwue_weekly=iwue_weekly,
        dates_weekly=time_index(temperature_weekly, "temperature_weekly"),
        disturbances_weekly=disturbances_weekly,
        leaf_pool_init=leaf_pool_init,
        stem_pool_init=stem_pool_init,
        root_pool_init=root_pool_init,
        latitude=latitude,
        litter_pool_init=litter_pool_init,
        removed_init=removed_init,
        **sgam_config,
    )
