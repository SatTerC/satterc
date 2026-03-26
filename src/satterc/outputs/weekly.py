import xarray as xr

from ._utils import _save_dataset


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
    """Merge weekly output data arrays into a single dataset.

    Parameters
    ----------
    gpp_weekly : xr.DataArray
        Weekly gross primary productivity.
    lue_weekly : xr.DataArray
        Weekly light use efficiency.
    iwue_weekly : xr.DataArray
        Weekly intrinsic water use efficiency.
    leaf_pool_size_weekly : xr.DataArray
        Weekly leaf pool size.
    stem_pool_size_weekly : xr.DataArray
        Weekly stem pool size.
    root_pool_size_weekly : xr.DataArray
        Weekly root pool size.
    leaf_respiration_loss_weekly : xr.DataArray
        Weekly leaf respiration loss.
    stem_respiration_loss_weekly : xr.DataArray
        Weekly stem respiration loss.
    root_respiration_loss_weekly : xr.DataArray
        Weekly root respiration loss.
    litter_to_soil_weekly : xr.DataArray
        Weekly litter to soil flux.
    disturbance_loss_weekly : xr.DataArray
        Weekly disturbance loss.
    leaf_area_index_weekly : xr.DataArray
        Weekly leaf area index.
    npp_weekly : xr.DataArray
        Weekly net primary productivity.
    cue_weekly : xr.DataArray
        Weekly carbon use efficiency.

    Returns
    -------
    xr.Dataset
        Merged dataset with stacked spatial dimensions.
    """
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
    """Unstack spatial dimensions from weekly outputs.

    Parameters
    ----------
    weekly_outputs_stacked : xr.Dataset
        Weekly outputs with stacked spatial dimensions.

    Returns
    -------
    xr.Dataset
        Weekly outputs with original spatial dimensions restored.
    """
    return weekly_outputs_stacked.unstack("pixel")


def saved_weekly_outputs(weekly_outputs: xr.Dataset, weekly_outputs_path: str) -> None:
    """Save weekly outputs to file.

    Parameters
    ----------
    weekly_outputs : xr.Dataset
        Weekly outputs dataset.
    weekly_outputs_path : str
        Path to save the dataset.
    """
    _save_dataset(weekly_outputs, weekly_outputs_path)
