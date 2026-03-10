from hamilton.function_modifiers import extract_fields
import numpy as np
from numpy.typing import NDArray
from pandas import DatetimeIndex
from xarray import DataArray

from rothc_py import RothC, percent_modern_c

from ._utils import xarray_io


@xarray_io()
def _rothc(
    temperature_celcius_monthly: NDArray[np.float64],
    precipitation_mm_monthly: NDArray[np.float64],
    evaporation_monthly: NDArray[np.float64],
    plant_cover_monthly: NDArray[np.bool],
    dpm_rpm_ratio_monthly: NDArray[np.float64],
    soil_carbon_input_monthly: NDArray[np.float64],
    farmyard_manure_input_monthly: NDArray[np.float64],
    clay_content: NDArray[np.float64],
    soil_depth: NDArray[np.float64],
    inert_organic_matter: NDArray[np.float64],
    n_years_spinup: int,
    dates_monthly: DatetimeIndex,
) -> dict[str, NDArray]:
    n_months, n_pixels = temperature_celcius_monthly.shape
    n_spinup_months = n_years_spinup * 12
    start_date = dates_monthly.values[0]

    t_mod = percent_modern_c(start_date=start_date, n_months=n_months)

    pixel_outputs = []
    for i in range(n_pixels):
        model = RothC(
            clay=clay_content[i], depth=soil_depth[i], iom=inert_organic_matter[i]
        )
        data = dict(
            t_tmp=temperature_celcius_monthly[:, i].tolist(),
            t_rain=precipitation_mm_monthly[:, i].tolist(),
            t_evap=evaporation_monthly[:, i].tolist(),
            t_PC=plant_cover_monthly[:, i].astype(int).tolist(),
            t_DPM_RPM=dpm_rpm_ratio_monthly[:, i].tolist(),
            t_C_Inp=soil_carbon_input_monthly[:, i].tolist(),
            t_FYM_Inp=farmyard_manure_input_monthly[:, i].tolist(),
            t_mod=t_mod,
        )

        spinup_data = dict(
            t_tmp=data["t_tmp"][:n_spinup_months],
            t_rain=data["t_rain"][:n_spinup_months],
            t_evap=data["t_evap"][:n_spinup_months],
            t_PC=data["t_PC"][:n_spinup_months],
            t_DPM_RPM=data["t_DPM_RPM"][:n_spinup_months],
            t_C_Inp=data["t_C_Inp"][:n_spinup_months],
            t_FYM_Inp=data["t_FYM_Inp"][:n_spinup_months],
            t_mod=t_mod[:n_spinup_months],
        )

        _, outputs = model(data, spinup_data)
        pixel_outputs.append(outputs)

    return dict(
        decomposable_plant_material_monthly=np.column_stack(
            [out["DPM_t_C_ha"] for out in pixel_outputs]
        ),
        resistant_plant_material_monthly=np.column_stack(
            [out["RPM_t_C_ha"] for out in pixel_outputs]
        ),
        microbial_biomass_monthly=np.column_stack(
            [out["BIO_t_C_ha"] for out in pixel_outputs]
        ),
        humified_organic_matter_monthly=np.column_stack(
            [out["HUM_t_C_ha"] for out in pixel_outputs]
        ),
        soil_organic_carbon_monthly=np.column_stack(
            [out["SOC_t_C_ha"] for out in pixel_outputs]
        ),
    )


def rothc_parameters(n_years_spinup: int) -> tuple[int]:
    """Static parameters for the Rothamsted Carbon model.

    Parameters
    ----------
    n_years_spinup
        Number of years to use for model spin-up.

    Returns
    -------
    Tuple containing these parameters.
    """
    return (n_years_spinup,)


@extract_fields(
    [
        "decomposable_plant_material_monthly",
        "resistant_plant_material_monthly",
        "microbial_biomass_monthly",
        "humified_organic_matter_monthly",
        "soil_organic_carbon_monthly",
    ]
)
def rothc(
    temperature_celcius_monthly: DataArray,
    precipitation_mm_monthly: DataArray,
    evaporation_monthly: DataArray,
    plant_cover_monthly: DataArray,
    dpm_rpm_ratio_monthly: DataArray,
    soil_carbon_input_monthly: DataArray,
    farmyard_manure_input_monthly: DataArray,
    clay_content: DataArray,
    inert_organic_matter: DataArray,
    soil_depth: DataArray,
    rothc_parameters: tuple[int],
) -> dict[str, DataArray]:
    """
    Rothamsted Carbon model.

    Monthly resolution input data.

    Parameters
    ----------
    temperature_celcius_monthly
        Monthly mean temperature in degrees Celsius.
    precipitation_mm_monthly
        Monthly precipitation in mm.
    evaporation_monthly
        Monthly evaporation in mm.
    plant_cover_monthly
        Monthly plant cover as boolean (True = covered).
    dpm_rpm_ratio_monthly
        Ratio of decomposable to resistant plant material.
    soil_carbon_input_monthly
        Carbon input in tC/ha/month.
    farmyard_manure_input_monthly
        Farmyard manure input in tC/ha/month.
    clay_content
        Clay content percentage.
    soil_depth
        Soil depth in cm.
    inert_organic_matter
        Inert organic matter in tC/ha.
    rothc_parameters
        Tuple of parameters.

    Returns
    -------
    dict
        Dictionary containing monthly model outputs:
        - decomposable_plant_material_monthly: DPM pool (tC/ha)
        - resistant_plant_material_monthly: RPM pool (tC/ha)
        - microbial_biomass_monthly: Microbial biomass pool (tC/ha)
        - humified_organic_matter_monthly: HUM pool (tC/ha)
        - soil_organic_carbon_monthly: Total SOC (tC/ha)

    Note
    ----
    All outputs have units tC/ha (tonnes of Carbon per hectare).
    All outputs are at monthly resolution.
    """
    (n_years_spinup,) = rothc_parameters

    dates_monthly = temperature_celcius_monthly.get_index("time")

    return _rothc(
        temperature_celcius_monthly=temperature_celcius_monthly,
        precipitation_mm_monthly=precipitation_mm_monthly,
        evaporation_monthly=evaporation_monthly,
        plant_cover_monthly=plant_cover_monthly,
        dpm_rpm_ratio_monthly=dpm_rpm_ratio_monthly,
        soil_carbon_input_monthly=soil_carbon_input_monthly,
        farmyard_manure_input_monthly=farmyard_manure_input_monthly,
        clay_content=clay_content,
        soil_depth=soil_depth,
        inert_organic_matter=inert_organic_matter,
        n_years_spinup=n_years_spinup,
        dates_monthly=dates_monthly,
    )


def evaporation_monthly(
    actual_evapotranspiration_monthly: DataArray,
) -> DataArray:
    return actual_evapotranspiration_monthly
    # BUG: this is not quite correct!!
    # RothC expects monthly *open pan evaporation* NOT actual evapotranspiration.


# Temporary bridge
def soil_carbon_input_monthly(litter_to_soil_monthly: DataArray) -> DataArray:
    return litter_to_soil_monthly


def inert_organic_matter(organic_carbon_stocks: DataArray) -> DataArray:
    return 0.049 * organic_carbon_stocks**1.139
    # NOTE: taken from https://github.com/vmyrgiotis/coupled-ecosystem-carbon-model/blob/v0/notebooks/notebook_v4.ipynb
