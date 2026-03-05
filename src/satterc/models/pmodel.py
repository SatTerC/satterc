"""
Satterc-compatable interface to PyRealm's 'P-model'.

This module provides the pmodel node, which wraps the pyrealm P-Model
to calculate gross primary productivity (GPP), light use efficiency (LUE),
and intrinsic water use efficiency (IWUE) from environmental inputs.
"""

from hamilton.function_modifiers import unpack_fields
import numpy as np
from numpy.typing import NDArray
import pyrealm.pmodel


def pmodel_parameters(
    method_optchi: str = "prentice14",
    method_jmaxlim: str = "wang17",
    method_kphio: str = "temperature",
    method_arrhenius: str = "simple",
) -> tuple[str, str, str, str]:
    """
    Parameters for the P-model.

    Parameters
    ----------
    method_optchi
        Method for calculating optimal chi (leaf-internal CO2 compensation point).
    method_jmaxlim
        Method for Jmax limitation.
    method_kphio
        Method for calculating the quantum yield efficiency (phi0).
    method_arrhenius
        Method for Arrhenius temperature scaling.

    Returns
    -------
    Tuple containing these parameters in the order that they appear in the signature.
    """
    return (method_optchi, method_jmaxlim, method_kphio, method_arrhenius)


@unpack_fields("gpp_weekly", "lue_weekly", "iwue_weekly")
def pmodel(
    temperature_celcius_weekly: NDArray[np.float64],
    vpd_pa_weekly: NDArray[np.float64],
    co2_ppm_weekly: NDArray[np.float64],
    pressure_pa_weekly: NDArray[np.float64],
    fapar_weekly: NDArray[np.float64],
    ppfd_umol_m2_s1_weekly: NDArray[np.float64],
    mean_growth_temperature_weekly: NDArray[np.float64],
    aridity_index_weekly: NDArray[np.float64],
    soil_moisture_weekly: NDArray[np.float64],
    pmodel_parameters: tuple[str, str, str, str],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Run the P-Model to calculate GPP, LUE, and IWUE.

    Parameters
    ----------
    temperature_celcius_weekly
        Air temperature (degrees Celsius).
    vpd_pa_weekly
        Vapor pressure deficit (Pascals).
    co2_ppm_weekly
        Atmospheric CO2 concentration (parts per million).
    pressure_pa_weekly
        Atmospheric pressure (Pascals).
    fapar_weekly
        Fraction of absorbed photosynthetically active radiation (dimensionless, 0-1).
    ppfd_umol_m2_s1_weekly
        Photosynthetic photon flux density (micromoles per square meter per second).
    soil_moisture_weekly
        Soil moisture content (mm).
    mean_growth_temperature_weekly
        Mean growth temperature (degrees Celsius).
    aridity_index_weekly
        Aridity index (dimensionless, ratio of actual evapotranspiration to precipitation).
    pmodel_parameters
        Tuple of parameters for the P-model.

    Returns
    -------
    tuple
        Tuple of weekly outputs:
        - gpp_weekly: Gross primary productivity (gC per m2 per day)
        - lue_weekly: Light use efficiency (gC per MJ PAR)
        - iwue_weekly: Intrinsic water use efficiency (Pa)
    """
    method_optchi, method_jmaxlim, method_kphio, method_arrhenius = pmodel_parameters

    # Environmental drivers computed upon instantiation of PModelEnvironment
    env = pyrealm.pmodel.PModelEnvironment(
        tc=temperature_celcius_weekly,
        vpd=vpd_pa_weekly,
        co2=co2_ppm_weekly,
        patm=pressure_pa_weekly,
        fapar=fapar_weekly,
        ppfd=ppfd_umol_m2_s1_weekly,
        theta=soil_moisture_weekly / 300,  # TODO: figure out how to remove this factor!
        mean_growth_temperature=mean_growth_temperature_weekly,
        aridity_index=aridity_index_weekly,
    )

    # P-model fit performed upon instantiation of Pmodel
    model = pyrealm.pmodel.PModel(
        env=env,
        method_optchi=method_optchi,
        method_kphio=method_kphio,
        method_arrhenius=method_arrhenius,
        method_jmaxlim=method_jmaxlim,
    )

    return (model.gpp, model.lue, model.iwue)
