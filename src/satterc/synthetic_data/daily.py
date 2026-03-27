"""Generate synthetic daily input data."""

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray


def _generate_seasonal_cycle(
    n_days: int, amplitude: float, phase_shift: float, baseline: float
) -> NDArray[np.float64]:
    """Generate a sinusoidal seasonal cycle."""
    t = np.arange(n_days)
    return baseline + amplitude * np.sin(2 * np.pi * t / 365.25 + phase_shift)


def time_coord(n_days: int, start_date: str = "2020-01-01") -> NDArray[np.datetime64]:
    """Create time coordinate."""
    start = np.datetime64(start_date)
    return start + np.arange(n_days)


def temperature_celcius_daily(
    time_coord: NDArray[np.datetime64],
    pixel_coords: pd.MultiIndex,
) -> xr.DataArray:
    """Daily air temperature in degrees Celsius."""
    n_days = len(time_coord)
    n_pixels = len(pixel_coords)

    data = np.zeros((n_days, n_pixels))
    for p in range(n_pixels):
        lat_val = pixel_coords.get_level_values("y")[p]
        lon_val = pixel_coords.get_level_values("x")[p]
        lat_effect = (lat_val - 52.0) * 0.5
        lon_effect = (lon_val + 1.0) * 0.3
        baseline = 10.0 + lat_effect + lon_effect
        seasonal = _generate_seasonal_cycle(n_days, 10.0, -np.pi / 2, 0)
        daily_variation = np.random.uniform(-3, 3, n_days)
        data[:, p] = baseline + seasonal + daily_variation

    return xr.DataArray(
        data=data,
        dims=["time", "pixel"],
        coords={"time": time_coord, "pixel": pixel_coords},
        attrs={"units": "degrees_C", "long_name": "air temperature"},
        name="temperature_celcius",
    )


def precipitation_mm_daily(
    time_coord: NDArray[np.datetime64],
    pixel_coords: pd.MultiIndex,
) -> xr.DataArray:
    """Daily precipitation in mm."""
    n_days = len(time_coord)
    n_pixels = len(pixel_coords)
    lat_values = pixel_coords.get_level_values("y")

    data = np.zeros((n_days, n_pixels))
    for p in range(n_pixels):
        lat_val = lat_values[p]
        base_precip = 2.5 + (54 - lat_val) * 0.3
        seasonal = _generate_seasonal_cycle(n_days, 1.0, 0, 0)
        daily_precip = np.random.exponential(base_precip + seasonal, n_days)
        wet_days = np.random.random(n_days) < 0.6
        data[:, p] = np.where(wet_days, daily_precip, 0.0)

    return xr.DataArray(
        data=data,
        dims=["time", "pixel"],
        coords={"time": time_coord, "pixel": pixel_coords},
        attrs={"units": "mm", "long_name": "precipitation"},
        name="precipitation_mm",
    )


def sunshine_fraction_daily(
    time_coord: NDArray[np.datetime64],
    pixel_coords: pd.MultiIndex,
) -> xr.DataArray:
    """Daily sunshine fraction (0-1)."""
    n_days = len(time_coord)
    n_pixels = len(pixel_coords)

    data = np.zeros((n_days, n_pixels))
    for p in range(n_pixels):
        seasonal = _generate_seasonal_cycle(n_days, 0.3, 0, 0.5)
        noise = np.random.uniform(-0.15, 0.15, n_days)
        data[:, p] = np.clip(seasonal + noise, 0.0, 1.0)

    return xr.DataArray(
        data=data,
        dims=["time", "pixel"],
        coords={"time": time_coord, "pixel": pixel_coords},
        attrs={"units": "dimensionless", "long_name": "sunshine fraction"},
        name="sunshine_fraction",
    )


def lai_daily(
    time_coord: NDArray[np.datetime64],
    pixel_coords: pd.MultiIndex,
) -> xr.DataArray:
    """Daily Leaf Area Index (m2/m2)."""
    n_days = len(time_coord)
    n_pixels = len(pixel_coords)

    data = np.zeros((n_days, n_pixels))
    for p in range(n_pixels):
        seasonal = _generate_seasonal_cycle(n_days, 2.5, -np.pi / 3, 3.0)
        noise = np.random.uniform(-0.3, 0.3, n_days)
        data[:, p] = np.clip(seasonal + noise, 0.1, 6.0)

    return xr.DataArray(
        data=data,
        dims=["time", "pixel"],
        coords={"time": time_coord, "pixel": pixel_coords},
        attrs={"units": "m2/m2", "long_name": "leaf area index"},
        name="lai",
    )


def gpp_daily(
    time_coord: NDArray[np.datetime64],
    pixel_coords: pd.MultiIndex,
    temperature_celcius_daily: xr.DataArray,
) -> xr.DataArray:
    """Daily Gross Primary Productivity (gC/m2/d)."""
    n_days = len(time_coord)
    n_pixels = len(pixel_coords)
    temp_vals = temperature_celcius_daily.values

    data = np.zeros((n_days, n_pixels))
    for p in range(n_pixels):
        base_gpp = 8.0
        seasonal = _generate_seasonal_cycle(n_days, 5.0, -np.pi / 3, 0)
        temp_factor = np.maximum(temp_vals[:, p] - 5, 0) / 15.0
        noise = np.random.uniform(-1, 1, n_days)
        data[:, p] = np.maximum(base_gpp + seasonal * temp_factor + noise, 0.1)

    return xr.DataArray(
        data=data,
        dims=["time", "pixel"],
        coords={"time": time_coord, "pixel": pixel_coords},
        attrs={"units": "gC/m2/d", "long_name": "gross primary productivity"},
        name="gpp",
    )


def dummy_variable_daily(
    time_coord: NDArray[np.datetime64],
    pixel_coords: pd.MultiIndex,
) -> xr.DataArray:
    """Daily dummy variable (NaN array)."""
    n_days = len(time_coord)
    n_pixels = len(pixel_coords)

    return xr.DataArray(
        data=np.full((n_days, n_pixels), np.nan),
        dims=["time", "pixel"],
        coords={"time": time_coord, "pixel": pixel_coords},
        attrs={"units": "dimensionless", "long_name": "dummy variable"},
        name="dummy_variable",
    )


def co2_ppm_daily(
    time_coord: NDArray[np.datetime64],
    pixel_coords: pd.MultiIndex,
) -> xr.DataArray:
    """Atmospheric CO2 concentration in ppm."""
    n_days = len(time_coord)
    n_pixels = len(pixel_coords)

    baseline = 412.0
    trend = np.linspace(0, 5, n_days)
    seasonal = _generate_seasonal_cycle(n_days, 3, 0, 0)
    noise = np.random.normal(0, 1, n_days)
    data_1d = baseline + trend + seasonal + noise
    data = np.broadcast_to(data_1d[:, np.newaxis], (n_days, n_pixels))

    return xr.DataArray(
        data=data,
        dims=["time", "pixel"],
        coords={"time": time_coord, "pixel": pixel_coords},
        attrs={"units": "ppm", "long_name": "atmospheric CO2 concentration"},
        name="co2_ppm",
    )


def fapar_daily(
    time_coord: NDArray[np.datetime64],
    pixel_coords: pd.MultiIndex,
) -> xr.DataArray:
    """Fraction of absorbed photosynthetically active radiation (0-1)."""
    n_days = len(time_coord)
    n_pixels = len(pixel_coords)

    data = np.zeros((n_days, n_pixels))
    for p in range(n_pixels):
        seasonal = _generate_seasonal_cycle(n_days, 0.25, 0, 0.55)
        noise = np.random.uniform(-0.1, 0.1, n_days)
        data[:, p] = np.clip(seasonal + noise, 0.05, 0.95)

    return xr.DataArray(
        data=data,
        dims=["time", "pixel"],
        coords={"time": time_coord, "pixel": pixel_coords},
        attrs={"units": "dimensionless", "long_name": "fAPAR"},
        name="fapar",
    )


def ppfd_umol_m2_s1_daily(
    time_coord: NDArray[np.datetime64],
    pixel_coords: pd.MultiIndex,
) -> xr.DataArray:
    """Photosynthetic photon flux density in umol/m2/s."""
    n_days = len(time_coord)
    n_pixels = len(pixel_coords)

    data = np.zeros((n_days, n_pixels))
    for p in range(n_pixels):
        day_of_year = np.arange(n_days) % 365.25
        max_ppfd = 1200 * np.abs(np.sin(np.pi * day_of_year / 182.6))
        cloud_effect = 0.4 + np.random.uniform(0.2, 0.6, n_days)
        data[:, p] = max_ppfd * cloud_effect

    return xr.DataArray(
        data=data,
        dims=["time", "pixel"],
        coords={"time": time_coord, "pixel": pixel_coords},
        attrs={"units": "umol/m2/s", "long_name": "photosynthetic photon flux density"},
        name="ppfd_umol_m2_s1",
    )


def pressure_pa_daily(
    time_coord: NDArray[np.datetime64],
    pixel_coords: pd.MultiIndex,
    elevation: xr.DataArray,
) -> xr.DataArray:
    """Atmospheric pressure in Pascals."""
    n_days = len(time_coord)
    n_pixels = len(pixel_coords)
    elevation_vals = elevation.values

    data = np.zeros((n_days, n_pixels))
    for p in range(n_pixels):
        elevation_effect = -elevation_vals[p] * 10.0
        seasonal = _generate_seasonal_cycle(n_days, 500, 0, 0)
        noise = np.random.normal(0, 300, n_days)
        data[:, p] = 101325.0 + elevation_effect + seasonal + noise

    return xr.DataArray(
        data=data,
        dims=["time", "pixel"],
        coords={"time": time_coord, "pixel": pixel_coords},
        attrs={"units": "Pa", "long_name": "atmospheric pressure"},
        name="pressure_pa",
    )


def vpd_pa_daily(
    time_coord: NDArray[np.datetime64],
    pixel_coords: pd.MultiIndex,
    temperature_celcius_daily: xr.DataArray,
) -> xr.DataArray:
    """Vapor pressure deficit in Pascals."""
    n_days = len(time_coord)
    n_pixels = len(pixel_coords)
    temp_vals = temperature_celcius_daily.values

    data = np.zeros((n_days, n_pixels))
    for p in range(n_pixels):
        temp = temp_vals[:, p]
        svp = 610.78 * np.exp(temp / (temp + 237.3) * 17.27)
        rh = np.clip(0.5 + np.random.uniform(-0.2, 0.2, n_days), 0.1, 0.95)
        vpd = svp * (1 - rh)
        data[:, p] = np.clip(vpd, 50, 3000)

    return xr.DataArray(
        data=data,
        dims=["time", "pixel"],
        coords={"time": time_coord, "pixel": pixel_coords},
        attrs={"units": "Pa", "long_name": "vapor pressure deficit"},
        name="vpd_pa",
    )
