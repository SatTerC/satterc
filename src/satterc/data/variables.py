import xarray as xr


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
