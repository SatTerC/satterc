from pathlib import Path
from os import PathLike

import xarray as xr


def _save_dataset(ds: xr.Dataset, path: str | PathLike) -> None:
    """
    Saves a dataset to a NetCDF file or Zarr store based on the file extension.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to save.
    path : str | PathLike
        The destination path.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in [".nc", ".netcdf"]:
        ds.to_netcdf(path, engine="netcdf4")

    elif suffix == ".zarr" or (not suffix and p.is_dir()):
        # 'mode' defaults to 'w' (write), but can be overridden in kwargs
        ds.to_zarr(path)

    else:
        raise ValueError(
            f"Unsupported file extension: '{suffix}'. Use '.nc', '.netcdf', or '.zarr'."
        )


def daily_outputs_stacked(
    actual_evapotranspiration_daily: xr.DataArray,
    soil_moisture_daily: xr.DataArray,
    runoff_daily: xr.DataArray,
) -> xr.Dataset:
    return xr.merge(
        [actual_evapotranspiration_daily, soil_moisture_daily, runoff_daily]
    )


def daily_outputs(daily_outputs_stacked: xr.Dataset) -> xr.Dataset:
    return daily_outputs_stacked.unstack("pixel")


def saved_daily_outputs(
    daily_outputs: xr.Dataset, daily_outputs_path: str | PathLike
) -> None:
    _save_dataset(daily_outputs, daily_outputs_path)


def weekly_outputs_stacked(
    gpp_weekly: xr.DataArray,
    lue_weekly: xr.DataArray,
    iwue_weekly: xr.DataArray,
    leaf_pool_size_weekly: xr.DataArray,
    stem_pool_size_weekly: xr.DataArray,
    root_pool_size_weekly: xr.DataArray,
    leaf_respiration_loss_weekly: xr.DataArray,
    stem_respiration_loss_weekly: xr.DataArray,
    root_respiration_loss_weekly: xr.DataArray,
    litter_to_soil_weekly: xr.DataArray,
    disturbance_loss_weekly: xr.DataArray,
    leaf_area_index_weekly: xr.DataArray,
    npp_weekly: xr.DataArray,
    cue_weekly: xr.DataArray,
) -> xr.Dataset:
    return xr.merge(
        [
            gpp_weekly,
            lue_weekly,
            iwue_weekly,
            leaf_pool_size_weekly,
            stem_pool_size_weekly,
            root_pool_size_weekly,
            leaf_respiration_loss_weekly,
            stem_respiration_loss_weekly,
            root_respiration_loss_weekly,
            litter_to_soil_weekly,
            disturbance_loss_weekly,
            leaf_area_index_weekly,
            npp_weekly,
            cue_weekly,
        ]
    )


def weekly_outputs(weekly_outputs_stacked: xr.Dataset) -> xr.Dataset:
    return weekly_outputs_stacked.unstack("pixel")


def saved_weekly_outputs(weekly_outputs: xr.Dataset, weekly_outputs_path: Path) -> None:
    _save_dataset(weekly_outputs, weekly_outputs_path)


def monthly_outputs_stacked(
    decomposable_plant_material_monthly: xr.DataArray,
    resistant_plant_material_monthly: xr.DataArray,
    microbial_biomass_monthly: xr.DataArray,
    humified_organic_matter_monthly: xr.DataArray,
    soil_organic_carbon_monthly: xr.DataArray,
) -> xr.Dataset:
    return xr.merge(
        [
            decomposable_plant_material_monthly,
            resistant_plant_material_monthly,
            microbial_biomass_monthly,
            humified_organic_matter_monthly,
            soil_organic_carbon_monthly,
        ]
    )


def monthly_outputs(monthly_outputs_stacked: xr.Dataset) -> xr.Dataset:
    return monthly_outputs_stacked.unstack("pixel")


def saved_monthly_outputs(
    monthly_outputs: xr.Dataset, monthly_outputs_path: Path
) -> None:
    _save_dataset(monthly_outputs, monthly_outputs_path)
