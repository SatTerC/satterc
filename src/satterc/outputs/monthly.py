import xarray as xr

from ._utils import _save_dataset


def monthly_outputs_stacked(
    decomposable_plant_material_monthly: xr.DataArray,
    resistant_plant_material_monthly: xr.DataArray,
    microbial_biomass_monthly: xr.DataArray,
    humified_organic_matter_monthly: xr.DataArray,
    soil_organic_carbon_monthly: xr.DataArray,
) -> xr.Dataset:
    """Merge monthly output data arrays into a single dataset.

    Parameters
    ----------
    decomposable_plant_material_monthly : xr.DataArray
        Monthly decomposable plant material (DPM).
    resistant_plant_material_monthly : xr.DataArray
        Monthly resistant plant material (RPM).
    microbial_biomass_monthly : xr.DataArray
        Monthly microbial biomass (BIO).
    humified_organic_matter_monthly : xr.DataArray
        Monthly humified organic matter (HUM).
    soil_organic_carbon_monthly : xr.DataArray
        Monthly soil organic carbon (SOC).

    Returns
    -------
    xr.Dataset
        Merged dataset with stacked spatial dimensions.
    """
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
    """Unstack spatial dimensions from monthly outputs.

    Parameters
    ----------
    monthly_outputs_stacked : xr.Dataset
        Monthly outputs with stacked spatial dimensions.

    Returns
    -------
    xr.Dataset
        Monthly outputs with original spatial dimensions restored.
    """
    return monthly_outputs_stacked.unstack("pixel")


def saved_monthly_outputs(
    monthly_outputs: xr.Dataset, monthly_outputs_path: str
) -> None:
    """Save monthly outputs to file.

    Parameters
    ----------
    monthly_outputs : xr.Dataset
        Monthly outputs dataset.
    monthly_outputs_path : str
        Path to save the dataset.
    """
    _save_dataset(monthly_outputs, monthly_outputs_path)
