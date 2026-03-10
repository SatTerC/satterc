"""
Satterc-compatable interface to PyRealm's 'P-model'.

This module provides the pmodel node, which wraps the pyrealm P-Model
to calculate gross primary productivity (GPP), light use efficiency (LUE),
and intrinsic water use efficiency (IWUE) from environmental inputs.
"""

from hamilton.function_modifiers import unpack_fields
from numpy.typing import NDArray
import xarray as xr
from xarray import DataArray
import pyrealm.pmodel

from ..utils import xarray_io


@xarray_io()
def _pmodel(
    temperature_celcius_weekly: NDArray,
    vpd_pa_weekly: NDArray,
    co2_ppm_weekly: NDArray,
    pressure_pa_weekly: NDArray,
    fapar_weekly: NDArray,
    ppfd_umol_m2_s1_weekly: NDArray,
    mean_growth_temperature_weekly: NDArray,
    aridity_index_weekly: NDArray,
    soil_moisture_weekly: NDArray,
    method_optchi: str,
    method_jmaxlim: str,
    method_kphio: str,
    method_arrhenius: str,
) -> tuple[NDArray, NDArray, NDArray]:
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
    temperature_celcius_weekly: DataArray,
    vpd_pa_weekly: DataArray,
    co2_ppm_weekly: DataArray,
    pressure_pa_weekly: DataArray,
    fapar_weekly: DataArray,
    ppfd_umol_m2_s1_weekly: DataArray,
    mean_growth_temperature_weekly: DataArray,
    aridity_index_weekly: DataArray,
    soil_moisture_weekly: DataArray,
    pmodel_parameters: tuple[str, str, str, str],
) -> tuple[DataArray, DataArray, DataArray]:
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

    return _pmodel(
        temperature_celcius_weekly=temperature_celcius_weekly,
        vpd_pa_weekly=vpd_pa_weekly,
        co2_ppm_weekly=co2_ppm_weekly,
        pressure_pa_weekly=pressure_pa_weekly,
        fapar_weekly=fapar_weekly,
        ppfd_umol_m2_s1_weekly=ppfd_umol_m2_s1_weekly,
        mean_growth_temperature_weekly=mean_growth_temperature_weekly,
        aridity_index_weekly=aridity_index_weekly,
        soil_moisture_weekly=soil_moisture_weekly,
        method_optchi=method_optchi,
        method_jmaxlim=method_jmaxlim,
        method_kphio=method_kphio,
        method_arrhenius=method_arrhenius,
    )


def aridity_index_daily(
    actual_evapotranspiration_daily: xr.DataArray,
    precipitation_mm_daily: xr.DataArray,
) -> xr.DataArray:
    """Calculate the aridity index.

    This is a dimensionless ratio of actual evapotranspiration to precipitation.

    This function is intended to act as a node in a Hamilton DAG.

    Parameters
    ----------
    actual_evapotranspiration_daily
        Actual evapotranspiration (mm).
    precipitation_mm_daily
        Precipitation (mm).

    Returns
    -------
    xr.DataArray
        Aridity index (dimensionless ratio of actual evapotranspiration to precipitation).
    """
    return actual_evapotranspiration_daily / precipitation_mm_daily
