"""Generate synthetic daily input data."""

from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray


def _generate_seasonal_cycle(
    n_days: int, amplitude: float, phase_shift: float, baseline: float
) -> NDArray[np.float64]:
    """Generate a sinusoidal seasonal cycle."""
    t = np.arange(n_days)
    return baseline + amplitude * np.sin(2 * np.pi * t / 365.25 + phase_shift)


def time_coord(n_days: int, start_date: str = "2020-01-01") -> NDArray[np.datetime64]:
    """Create time coordinate.

    Parameters
    ----------
    n_days : int
        Number of days.
    start_date : str
        Start date string.

    Returns
    -------
    NDArray[np.datetime64]
        Time coordinate array.
    """
    start = np.datetime64(start_date)
    return start + np.arange(n_days)


def lat(n_lat: int) -> NDArray[np.float64]:
    """Latitude coordinates.

    Parameters
    ----------
    n_lat : int
        Number of latitude points.

    Returns
    -------
    NDArray[np.float64]
        Latitude array.
    """
    return np.linspace(50.0, 54.0, n_lat)


def lon(n_lon: int) -> NDArray[np.float64]:
    """Longitude coordinates.

    Parameters
    ----------
    n_lon : int
        Number of longitude points.

    Returns
    -------
    NDArray[np.float64]
        Longitude array.
    """
    return np.linspace(-4.0, 2.0, n_lon)


def temperature_celcius_daily(
    time_coord: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
) -> xr.DataArray:
    """Daily air temperature in degrees Celsius for UK conditions.

    Parameters
    ----------
    time_coord : NDArray[np.datetime64]
        Time coordinate.
    lat : NDArray[np.float64]
        Latitude array.
    lon : NDArray[np.float64]
        Longitude array.

    Returns
    -------
    xr.DataArray
        Daily temperature data array.
    """
    n_days = len(time_coord)
    n_lat, n_lon = len(lat), len(lon)

    base_temp = 10.0
    seasonal_amp = 10.0

    data = np.zeros((n_days, n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            lat_effect = (lat[i] - 52.0) * 0.5
            lon_effect = (lon[j] + 1.0) * 0.3
            baseline = base_temp + lat_effect + lon_effect
            seasonal = _generate_seasonal_cycle(n_days, seasonal_amp, -np.pi / 2, 0)
            daily_variation = np.random.uniform(-3, 3, n_days)
            data[:, i, j] = baseline + seasonal + daily_variation

    return xr.DataArray(
        data=data,
        dims=["time", "y", "x"],
        coords={"time": time_coord, "y": lat, "x": lon},
        attrs={"units": "degrees_C", "long_name": "air temperature"},
        name="temperature_celcius",
    )


def precipitation_mm_daily(
    time_coord: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
) -> xr.DataArray:
    """Daily precipitation in mm for UK conditions.

    Parameters
    ----------
    time_coord : NDArray[np.datetime64]
        Time coordinate.
    lat : NDArray[np.float64]
        Latitude array.
    lon : NDArray[np.float64]
        Longitude array.

    Returns
    -------
    xr.DataArray
        Daily precipitation data array.
    """
    n_days = len(time_coord)
    n_lat, n_lon = len(lat), len(lon)

    data = np.zeros((n_days, n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            base_precip = 2.5 + (54 - lat[i]) * 0.3
            seasonal = _generate_seasonal_cycle(n_days, 1.0, 0, 0)
            daily_precip = np.random.exponential(base_precip + seasonal, n_days)
            wet_days = np.random.random(n_days) < 0.6
            data[:, i, j] = np.where(wet_days, daily_precip, 0.0)

    return xr.DataArray(
        data=data,
        dims=["time", "y", "x"],
        coords={"time": time_coord, "y": lat, "x": lon},
        attrs={"units": "mm", "long_name": "precipitation"},
        name="precipitation_mm",
    )


def sunshine_fraction_daily(
    time_coord: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
) -> xr.DataArray:
    """Daily sunshine fraction (0-1) for UK conditions.

    Parameters
    ----------
    time_coord : NDArray[np.datetime64]
        Time coordinate.
    lat : NDArray[np.float64]
        Latitude array.
    lon : NDArray[np.float64]
        Longitude array.

    Returns
    -------
    xr.DataArray
        Daily sunshine fraction data array.
    """
    n_days = len(time_coord)
    n_lat, n_lon = len(lat), len(lon)

    data = np.zeros((n_days, n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            seasonal = _generate_seasonal_cycle(n_days, 0.3, 0, 0.5)
            noise = np.random.uniform(-0.15, 0.15, n_days)
            data[:, i, j] = np.clip(seasonal + noise, 0.0, 1.0)

    return xr.DataArray(
        data=data,
        dims=["time", "y", "x"],
        coords={"time": time_coord, "y": lat, "x": lon},
        attrs={"units": "dimensionless", "long_name": "sunshine fraction"},
        name="sunshine_fraction",
    )


def lai_daily(
    time_coord: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
) -> xr.DataArray:
    """Daily Leaf Area Index (m²/m²) for UK conditions.

    Parameters
    ----------
    time_coord : NDArray[np.datetime64]
        Time coordinate.
    lat : NDArray[np.float64]
        Latitude array.
    lon : NDArray[np.float64]
        Longitude array.

    Returns
    -------
    xr.DataArray
        Daily LAI data array.
    """
    n_days = len(time_coord)
    n_lat, n_lon = len(lat), len(lon)

    data = np.zeros((n_days, n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            seasonal = _generate_seasonal_cycle(n_days, 2.5, -np.pi / 3, 3.0)
            noise = np.random.uniform(-0.3, 0.3, n_days)
            data[:, i, j] = np.clip(seasonal + noise, 0.1, 6.0)

    return xr.DataArray(
        data=data,
        dims=["time", "y", "x"],
        coords={"time": time_coord, "y": lat, "x": lon},
        attrs={"units": "m2/m2", "long_name": "leaf area index"},
        name="lai",
    )


def gpp_daily(
    time_coord: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    temperature_celcius_daily: xr.DataArray,
) -> xr.DataArray:
    """Daily Gross Primary Productivity (gC/m²/d).

    Parameters
    ----------
    time_coord : NDArray[np.datetime64]
        Time coordinate.
    lat : NDArray[np.float64]
        Latitude array.
    lon : NDArray[np.float64]
        Longitude array.
    temperature_celcius_daily : xr.DataArray
        Temperature data array.

    Returns
    -------
    xr.DataArray
        Daily GPP data array.
    """
    n_days = len(time_coord)
    n_lat, n_lon = len(lat), len(lon)

    data = np.zeros((n_days, n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            base_gpp = 8.0
            seasonal = _generate_seasonal_cycle(n_days, 5.0, -np.pi / 3, 0)
            temp_factor = (
                np.maximum(temperature_celcius_daily.values[:, i, j] - 5, 0) / 15.0
            )
            noise = np.random.uniform(-1, 1, n_days)
            data[:, i, j] = np.maximum(base_gpp + seasonal * temp_factor + noise, 0.1)

    return xr.DataArray(
        data=data,
        dims=["time", "y", "x"],
        coords={"time": time_coord, "y": lat, "x": lon},
        attrs={"units": "gC/m2/d", "long_name": "gross primary productivity"},
        name="gpp",
    )


def dummy_variable_daily(
    time_coord: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
) -> xr.DataArray:
    """Monthly dummy variable (NaN array).

    Parameters
    ----------
    time_coord : NDArray[np.datetime64]
        Time coordinate.
    lat : NDArray[np.float64]
        Latitude array.
    lon : NDArray[np.float64]
        Longitude array.

    Returns
    -------
    xr.DataArray
        Dummy variable data array (NaN).
    """
    return xr.DataArray(
        data=np.full((len(time_coord), len(lat), len(lon)), np.nan),
        dims=["time", "y", "x"],
        coords={"time": time_coord, "y": lat, "x": lon},
        attrs={"units": "dimensionless", "long_name": "dummy variable"},
        name="dummy_variable",
    )


def co2_ppm_daily(
    time_coord: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
) -> xr.DataArray:
    """Atmospheric CO2 concentration in ppm.

    Parameters
    ----------
    time_coord : NDArray[np.datetime64]
        Time coordinate.
    lat : NDArray[np.float64]
        Latitude array.
    lon : NDArray[np.float64]
        Longitude array.

    Returns
    -------
    xr.DataArray
        CO2 concentration data array.
    """
    n_days = len(time_coord)
    n_lat, n_lon = len(lat), len(lon)

    baseline = 412.0
    trend = np.linspace(0, 5, n_days)
    seasonal = _generate_seasonal_cycle(n_days, 3, 0, 0)
    noise = np.random.normal(0, 1, n_days)

    data_1d = baseline + trend + seasonal + noise
    data = np.broadcast_to(data_1d[:, np.newaxis, np.newaxis], (n_days, n_lat, n_lon))

    return xr.DataArray(
        data=data,
        dims=["time", "y", "x"],
        coords={"time": time_coord, "y": lat, "x": lon},
        attrs={"units": "ppm", "long_name": "atmospheric CO2 concentration"},
        name="co2_ppm",
    )


def fapar_daily(
    time_coord: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
) -> xr.DataArray:
    """Fraction of absorbed photosynthetically active radiation (0-1).

    Parameters
    ----------
    time_coord : NDArray[np.datetime64]
        Time coordinate.
    lat : NDArray[np.float64]
        Latitude array.
    lon : NDArray[np.float64]
        Longitude array.

    Returns
    -------
    xr.DataArray
        FAPAR data array.
    """
    n_days = len(time_coord)
    n_lat, n_lon = len(lat), len(lon)

    data = np.zeros((n_days, n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            seasonal = _generate_seasonal_cycle(n_days, 0.25, 0, 0.55)
            noise = np.random.uniform(-0.1, 0.1, n_days)
            data[:, i, j] = np.clip(seasonal + noise, 0.05, 0.95)

    return xr.DataArray(
        data=data,
        dims=["time", "y", "x"],
        coords={"time": time_coord, "y": lat, "x": lon},
        attrs={"units": "dimensionless", "long_name": "fAPAR"},
        name="fapar",
    )


def ppfd_umol_m2_s1_daily(
    time_coord: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
) -> xr.DataArray:
    """Photosynthetic photon flux density in µmol/m²/s.

    Parameters
    ----------
    time_coord : NDArray[np.datetime64]
        Time coordinate.
    lat : NDArray[np.float64]
        Latitude array.
    lon : NDArray[np.float64]
        Longitude array.

    Returns
    -------
    xr.DataArray
        PPFD data array.
    """
    n_days = len(time_coord)
    n_lat, n_lon = len(lat), len(lon)

    data = np.zeros((n_days, n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            day_of_year = np.arange(n_days) % 365.25
            max_ppfd = 1200 * np.abs(np.sin(np.pi * day_of_year / 182.6))
            cloud_effect = 0.4 + np.random.uniform(0.2, 0.6, n_days)
            data[:, i, j] = max_ppfd * cloud_effect

    return xr.DataArray(
        data=data,
        dims=["time", "y", "x"],
        coords={"time": time_coord, "y": lat, "x": lon},
        attrs={"units": "umol/m2/s", "long_name": "photosynthetic photon flux density"},
        name="ppfd_umol_m2_s1",
    )


def pressure_pa_daily(
    time_coord: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    elevation: xr.DataArray,
) -> xr.DataArray:
    """Atmospheric pressure in Pascals.

    Parameters
    ----------
    time_coord : NDArray[np.datetime64]
        Time coordinate.
    lat : NDArray[np.float64]
        Latitude array.
    lon : NDArray[np.float64]
        Longitude array.
    elevation : xr.DataArray
        Elevation data array.

    Returns
    -------
    xr.DataArray
        Pressure data array.
    """
    n_days = len(time_coord)
    n_lat, n_lon = len(lat), len(lon)

    baseline_pressure = 101325.0
    data = np.zeros((n_days, n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            elevation_effect = -elevation.values[i, j] * 10.0
            seasonal = _generate_seasonal_cycle(n_days, 500, 0, 0)
            noise = np.random.normal(0, 300, n_days)
            data[:, i, j] = baseline_pressure + elevation_effect + seasonal + noise

    return xr.DataArray(
        data=data,
        dims=["time", "y", "x"],
        coords={"time": time_coord, "y": lat, "x": lon},
        attrs={"units": "Pa", "long_name": "atmospheric pressure"},
        name="pressure_pa",
    )


def vpd_pa_daily(
    time_coord: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    temperature_celcius_daily: xr.DataArray,
) -> xr.DataArray:
    """Vapor pressure deficit in Pascals.

    Parameters
    ----------
    time_coord : NDArray[np.datetime64]
        Time coordinate.
    lat : NDArray[np.float64]
        Latitude array.
    lon : NDArray[np.float64]
        Longitude array.
    temperature_celcius_daily : xr.DataArray
        Temperature data array.

    Returns
    -------
    xr.DataArray
        VPD data array.
    """
    temp_da = temperature_celcius_daily
    data = np.zeros_like(temp_da.values)

    for i in range(temp_da.shape[1]):
        for j in range(temp_da.shape[2]):
            temp = temp_da.values[:, i, j]
            svp = 610.78 * np.exp(temp / (temp + 237.3) * 17.27)
            rh = np.clip(0.5 + np.random.uniform(-0.2, 0.2, len(temp)), 0.1, 0.95)
            vpd = svp * (1 - rh)
            data[:, i, j] = np.clip(vpd, 50, 3000)

    return xr.DataArray(
        data=data,
        dims=["time", "y", "x"],
        coords={"time": time_coord, "y": lat, "x": lon},
        attrs={"units": "Pa", "long_name": "vapor pressure deficit"},
        name="vpd_pa",
    )
