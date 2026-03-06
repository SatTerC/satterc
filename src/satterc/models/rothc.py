from hamilton.function_modifiers import extract_fields
import numpy as np
from numpy.typing import NDArray
from xarray import DataArray

from rothc_py import RothC, percent_modern_c

from ..utils import xarray_io


@xarray_io(flatten_spatial=True, inject_time="dates_monthly")
def _rothc(
    temperature_celcius_monthly: NDArray[np.float64],
    precipitation_mm_monthly: NDArray[np.float64],
    evaporation_monthly: NDArray[np.float64],
    plant_cover_monthly: NDArray[np.bool],
    dpm_rpm_ratio_monthly: NDArray[np.float64],
    carbon_input_monthly: NDArray[np.float64],
    farmyard_manure_input_monthly: NDArray[np.float64],
    dates_monthly,
    clay: float,
    soil_depth: float,
    inert_organic_matter: float,
    n_years_spinup: int,
) -> dict[str, NDArray]:
    n_months, n_pixels = temperature_celcius_monthly.shape
    n_spinup_months = n_years_spinup * 12

    t_mod = percent_modern_c(start_date=dates_monthly[0], n_months=n_months)

    model = RothC(clay=clay, depth=soil_depth, iom=inert_organic_matter)

    pixel_outputs = []
    for i in range(n_pixels):
        data = dict(
            t_tmp=temperature_celcius_monthly[:, i].tolist(),
            t_rain=precipitation_mm_monthly[:, i].tolist(),
            t_evap=evaporation_monthly[:, i].tolist(),
            t_PC=plant_cover_monthly[:, i].astype(int).tolist(),
            t_DPM_RPM=dpm_rpm_ratio_monthly[:, i].tolist(),
            t_C_Inp=carbon_input_monthly[:, i].tolist(),
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


def rothc_parameters(
    clay: float,
    soil_depth: float,
    inert_organic_matter: float,
    n_years_spinup: int,
) -> tuple[float, float, float, int]:
    """Static parameters for the Rothamsted Carbon model.

    Parameters
    ----------
    clay
        Clay content percentage.
    soil_depth
        Soil depth in cm.
    inert_organic_matter
        Inert organic matter in tC/ha.
    n_years_spinup
        Number of years to use for model spin-up.

    Returns
    -------
    Tuple containing these parameters.
    """
    return (clay, soil_depth, inert_organic_matter, n_years_spinup)


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
    carbon_input_monthly: DataArray,
    farmyard_manure_input_monthly: DataArray,
    dates_monthly: DataArray,
    rothc_parameters: tuple[float, float, float, int],
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
    carbon_input_monthly
        Carbon input in tC/ha/month.
    farmyard_manure_input_monthly
        Farmyard manure input in tC/ha/month.
    dates_monthly
        Monthly dates.
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
    clay, soil_depth, inert_organic_matter, n_years_spinup = rothc_parameters

    return _rothc(
        temperature_celcius_monthly=temperature_celcius_monthly,
        precipitation_mm_monthly=precipitation_mm_monthly,
        evaporation_monthly=evaporation_monthly,
        plant_cover_monthly=plant_cover_monthly,
        dpm_rpm_ratio_monthly=dpm_rpm_ratio_monthly,
        carbon_input_monthly=carbon_input_monthly,
        farmyard_manure_input_monthly=farmyard_manure_input_monthly,
        dates_monthly=dates_monthly,
        clay=clay,
        soil_depth=soil_depth,
        inert_organic_matter=inert_organic_matter,
        n_years_spinup=n_years_spinup,
    )
