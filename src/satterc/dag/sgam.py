"""Storage Gap Model (SGAM) vegetation model interface for the SatTerC pipeline."""

import numpy as np
import pandas as pd
import xarray as xr
from hamilton.function_modifiers import extract_fields
from numpy.typing import NDArray
from sgam import Disturbances, Sgam
from sgam.pft import PftParams, PlantFunctionalType, get_default_pft_params

from ._utils import xarray_io


def _pft_int_to_enum(value: int) -> PlantFunctionalType:
    return list(PlantFunctionalType)[value]


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
    plant_type : xr.DataArray
        Plant functional type as integer (0=tree, 1=grass, 2=shrub, 3=crop).
        Dims: ["pixel"].

    Returns
    -------
    xr.Dataset
        Dataset with dimension (pixel) containing PFT parameters for each pixel.
    """
    return _build_pft_params_dataset(plant_type)


@xarray_io()
def _disturbances_daily(
    temperature_celcius_daily: NDArray,
    gpp_daily: NDArray,
    lai_daily: NDArray,
    plant_type: NDArray,
    latitude: NDArray,
) -> NDArray:
    # TODO: upgrade growing_season_limit to a function of pft and latitude!
    # TODO: upgrade disturbance_threshold to a function of pft!
    return Disturbances(growing_season_limit=10.0, disturbance_threshold=0.3)(
        temperature_celcius_daily, gpp_daily, lai_daily, aggregate=False
    )


def disturbances_daily(
    temperature_celcius_daily: xr.DataArray,
    gpp_daily: xr.DataArray,
    lai_daily: xr.DataArray,
    plant_type: xr.DataArray,
    latitude: xr.DataArray,
) -> xr.DataArray:
    """Calculate daily disturbance events.

    Parameters
    ----------
    temperature_celcius_daily : xr.DataArray
        Daily air temperature (degrees Celsius).
    gpp_daily : xr.DataArray
        Daily gross primary productivity (gC/m²).
    lai_daily : xr.DataArray
        Daily leaf area index.
    plant_type: xr.DataArray
        Plant functional type.
    latitude: xr.DataArray
        Latitude.

    Returns
    -------
    xr.DataArray
        Daily disturbance indicators.
    """
    return _disturbances_daily(
        temperature_celcius_daily, gpp_daily, lai_daily, plant_type.values, latitude
    )


@xarray_io()
def _sgam(
    plant_type: NDArray[np.int_],
    pft_params: xr.Dataset,
    temperature_celcius_weekly: NDArray[np.float64],
    gpp_weekly: NDArray[np.float64],
    soil_moisture_weekly: NDArray[np.float64],
    vpd_pa_weekly: NDArray[np.float64],
    lue_weekly: NDArray[np.float64],
    iwue_weekly: NDArray[np.float64],
    dates_weekly: pd.DatetimeIndex,
    disturbances_weekly: NDArray[np.float64],
    leaf_pool_init: NDArray[np.float64],
    stem_pool_init: NDArray[np.float64],
    root_pool_init: NDArray[np.float64],
    latitude: NDArray[np.float64],
    litter_pool_init: NDArray[np.float64] | None = None,
    removed_init: NDArray[np.float64] | None = None,
    use_dynamic_allocation: bool = True,
    strict_mass_balance: bool = False,
) -> dict[str, NDArray]:
    # Week index, from 1-52
    week_of_year = dates_weekly.isocalendar().week.values

    # TODO: manual loop might benefit from an apply_ufunc or something.
    results_all_pixels = []

    for i in range(len(plant_type)):
        pft_enum = _pft_int_to_enum(int(plant_type[i]))
        params = _pft_params_from_dataset(pft_params, i)
        hemisphere = "NH" if latitude[i] >= 0 else "SH"

        output = Sgam(
            plant_type=pft_enum,
            pft_params=params,
            use_dynamic_allocation=use_dynamic_allocation,
            hemisphere=hemisphere,
        )(
            gpp=gpp_weekly[:, i],
            temperature=temperature_celcius_weekly[:, i],
            soil_moisture=soil_moisture_weekly[:, i],
            vpd=vpd_pa_weekly[:, i],
            lue=lue_weekly[:, i],
            iwue=iwue_weekly[:, i],
            week_of_year=week_of_year,
            disturbances=disturbances_weekly[:, i],
            leaf_pool_init=leaf_pool_init[i],
            stem_pool_init=stem_pool_init[i],
            root_pool_init=root_pool_init[i],
            litter_pool_init=litter_pool_init[i]
            if litter_pool_init is not None
            else 0.0,
            removed_init=removed_init[i] if removed_init is not None else 0.0,
            strict_mass_balance=strict_mass_balance,
        )

        results_all_pixels.append(
            {
                "leaf_pool_weekly": output.pools.leaf,
                "stem_pool_weekly": output.pools.stem,
                "root_pool_weekly": output.pools.root,
                "litter_pool_weekly": output.pools.litter,
                "removed_pool_weekly": output.pools.removed,
                "npp_leaf_weekly": output.npp.leaf,
                "npp_stem_weekly": output.npp.stem,
                "npp_root_weekly": output.npp.root,
                "turnover_leaf_weekly": output.turnover.leaf,
                "turnover_stem_weekly": output.turnover.stem,
                "turnover_root_weekly": output.turnover.root,
                "respiration_leaf_weekly": output.respiration.leaf,
                "respiration_stem_weekly": output.respiration.stem,
                "respiration_root_weekly": output.respiration.root,
                "disturbance_leaf_weekly": output.disturbance.leaf,
                "disturbance_stem_weekly": output.disturbance.stem,
                "disturbance_root_weekly": output.disturbance.root,
                "cue_weekly": output.diagnostics.cue,
                "allocation_leaf_weekly": output.diagnostics.allocation_leaf,
                "allocation_stem_weekly": output.diagnostics.allocation_stem,
                "allocation_root_weekly": output.diagnostics.allocation_root,
                "drought_modifier_weekly": output.diagnostics.drought_modifier,
                "lue_score_weekly": output.diagnostics.lue_score,
                "iwue_score_weekly": output.diagnostics.iwue_score,
            }
        )

    # Stack list of dicts of 1D arrays (time) into dict of 2D arrays (time, pixel)
    keys = results_all_pixels[0].keys()
    results_stacked = {
        key: np.vstack([d[key] for d in results_all_pixels]) for key in keys
    }

    return results_stacked


@extract_fields(
    [
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
    ]
)
def sgam(
    plant_type: xr.DataArray,
    pft_params: xr.Dataset,
    temperature_celcius_weekly: xr.DataArray,
    gpp_weekly: xr.DataArray,
    soil_moisture_weekly: xr.DataArray,
    vpd_pa_weekly: xr.DataArray,
    lue_weekly: xr.DataArray,
    iwue_weekly: xr.DataArray,
    disturbances_weekly: xr.DataArray,
    dates_weekly: pd.Index,
    leaf_pool_init: xr.DataArray,
    stem_pool_init: xr.DataArray,
    root_pool_init: xr.DataArray,
    latitude: xr.DataArray,
    litter_pool_init: xr.DataArray | None = None,
    removed_init: xr.DataArray | None = None,
    use_dynamic_allocation: bool = True,
    strict_mass_balance: bool = False,
) -> dict[str, xr.DataArray]:
    """Run the Storage Gap Model (SGAM) vegetation model.

    Parameters
    ----------
    plant_type : xr.DataArray
        Plant functional type as integer (0=tree, 1=grass, 2=shrub, 3=crop).
    pft_params : xr.Dataset
        PFT parameters for each pixel. Output of pft_params node.
    temperature_celcius_weekly : xr.DataArray
        Weekly air temperature (degrees Celsius).
    gpp_weekly : xr.DataArray
        Weekly gross primary productivity (gC/m²).
    soil_moisture_weekly : xr.DataArray
        Weekly soil moisture (mm).
    vpd_pa_weekly : xr.DataArray
        Weekly vapor pressure deficit (Pa).
    lue_weekly : xr.DataArray
        Weekly light use efficiency (gC/MJ).
    iwue_weekly : xr.DataArray
        Weekly intrinsic water use efficiency (Pa).
    disturbances_weekly : xr.DataArray
        Weekly disturbance indicators.
    dates_weekly : pd.Index
        Weekly datetime index.
    leaf_pool_init : xr.DataArray
        Initial leaf pool size.
    stem_pool_init : xr.DataArray
        Initial stem pool size.
    root_pool_init : xr.DataArray
        Initial root pool size.
    latitude : xr.DataArray
        Latitude for each pixel (used to determine hemisphere).
    litter_pool_init : xr.DataArray, optional
        Initial litter pool size. Defaults to 0.0.
    removed_init : xr.DataArray, optional
        Initial removed-carbon pool size. Defaults to 0.0.
    use_dynamic_allocation : bool, optional
        If True (default), allocation fractions vary with environmental
        conditions. If False, use fixed base allocations from pft_params.
    strict_mass_balance : bool, optional
        If True, raise RuntimeError on a mass balance violation.
        If False, issue a warning instead. Defaults to False.

    Returns
    -------
    dict[str, xr.DataArray]
        Dictionary containing vegetation pool sizes, fluxes, and diagnostics.
    """
    return _sgam(
        plant_type=plant_type.values,
        pft_params=pft_params,
        temperature_celcius_weekly=temperature_celcius_weekly,
        gpp_weekly=gpp_weekly,
        soil_moisture_weekly=soil_moisture_weekly,
        vpd_pa_weekly=vpd_pa_weekly,
        lue_weekly=lue_weekly,
        iwue_weekly=iwue_weekly,
        dates_weekly=dates_weekly,
        disturbances_weekly=disturbances_weekly,
        leaf_pool_init=leaf_pool_init,
        stem_pool_init=stem_pool_init,
        root_pool_init=root_pool_init,
        latitude=latitude,
        litter_pool_init=litter_pool_init,
        removed_init=removed_init,
        use_dynamic_allocation=use_dynamic_allocation,
        strict_mass_balance=strict_mass_balance,
    )
