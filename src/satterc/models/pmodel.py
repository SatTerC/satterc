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


@unpack_fields("gpp", "lue", "iwue")
def pmodel(
    temperature_celcius: NDArray[np.float64],
    vpd_pa: NDArray[np.float64],
    co2_ppm: NDArray[np.float64],
    pressure_pa: NDArray[np.float64],
    fapar: NDArray[np.float64],
    ppfd_umol_m2_s1: NDArray[np.float64],
    mean_growth_temperature: NDArray[np.float64],
    aridity_index: NDArray[np.float64],
    soil_moisture: NDArray[np.float64],
    method_optchi: str = "prentice14",
    method_jmaxlim: str = "wang17",
    method_kphio: str = "temperature",
    method_arrhenius: str = "simple",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Run the P-Model to calculate GPP, LUE, and IWUE.

    This function is intended to act as a node in a Hamilton DAG.

    Parameters
    ----------
    temperature_celcius
        Air temperature (degrees Celsius).
    vpd_pa
        Vapor pressure deficit (Pascals).
    co2_ppm
        Atmospheric CO2 concentration (parts per million).
    pressure_pa
        Atmospheric pressure (Pascals).
    fapar
        Fraction of absorbed photosynthetically active radiation (dimensionless, 0-1).
    ppfd_umol_m2_s1
        Photosynthetic photon flux density (micromoles per square meter per second).
    soil_moisture
        Soil moisture content (mm).
    mean_growth_temperature
        Mean growth temperature (degrees Celsius).
    aridity_index
        Aridity index (dimensionless, ratio of actual evapotranspiration to precipitation).
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
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        A tuple containing (i) Gross primary productivity (gC per m2 per day), (ii) \
        Light use efficiency (gC per MJ PAR), (iii) Intrinsic water use efficiency (Pa).
    """

    # Environmental drivers computed upon instantiation of PModelEnvironment
    env = pyrealm.pmodel.PModelEnvironment(
        tc=temperature_celcius,
        vpd=vpd_pa,
        co2=co2_ppm,
        patm=pressure_pa,
        fapar=fapar,
        ppfd=ppfd_umol_m2_s1,
        theta=soil_moisture / 300,  # TODO: figure out how to remove this factor!
        mean_growth_temperature=mean_growth_temperature,
        aridity_index=aridity_index,
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
