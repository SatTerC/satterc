from hamilton.function_modifiers import extract_fields
import numpy as np
from numpy.typing import NDArray

from rothc_py import RothC, percent_modern_c


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
    temperature_celcius_monthly: NDArray[np.float64],
    precipitation_mm_monthly: NDArray[np.float64],
    evaporation_monthly: NDArray[np.float64],
    plant_cover_monthly: NDArray[np.bool],
    dpm_rpm_ratio_monthly: NDArray[np.float64],
    carbon_input_monthly: NDArray[np.float64],
    farmyard_manure_input_monthly: NDArray[np.float64],
    dates_monthly: NDArray[np.datetime64],
    rothc_parameters: tuple[float, float, float, int],
) -> dict[str, NDArray]:
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
        Monthly dates as numpy datetime64.
    rothc_parameters
        Tuple of parameters.

    Returns
    -------
    dict[str, NDArray[np.float64]]
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
    n_spinup_months = n_years_spinup * 12

    t_mod = percent_modern_c(start_date=dates_monthly[0], n_months=len(dates_monthly))

    data = dict(
        t_tmp=temperature_celcius_monthly,
        t_rain=precipitation_mm_monthly,
        t_evap=evaporation_monthly,
        t_PC=plant_cover_monthly,
        t_DPM_RPM=dpm_rpm_ratio_monthly,
        t_C_Inp=carbon_input_monthly,
        t_FYM_Inp=farmyard_manure_input_monthly,
        t_mod=t_mod,
    )

    spinup_data = dict(
        t_tmp=temperature_celcius_monthly[:n_spinup_months],
        t_rain=precipitation_mm_monthly[:n_spinup_months],
        t_evap=evaporation_monthly[:n_spinup_months],
        t_PC=plant_cover_monthly[:n_spinup_months],
        t_DPM_RPM=dpm_rpm_ratio_monthly[:n_spinup_months],
        t_C_Inp=carbon_input_monthly[:n_spinup_months],
        t_FYM_Inp=farmyard_manure_input_monthly[:n_spinup_months],
        t_mod=t_mod[:n_spinup_months],
    )

    model = RothC(clay=clay, depth=soil_depth, iom=inert_organic_matter)

    _, outputs = model(data, spinup_data)

    return dict(
        decomposable_plant_material_monthly=np.array(outputs["DPM_t_C_ha"]),
        resistant_plant_material_monthly=np.array(outputs["RPM_t_C_ha"]),
        microbial_biomass_monthly=np.array(outputs["BIO_t_C_ha"]),
        humified_organic_matter_monthly=np.array(outputs["HUM_t_C_ha"]),
        soil_organic_carbon_monthly=np.array(outputs["SOC_t_C_ha"]),
    )
