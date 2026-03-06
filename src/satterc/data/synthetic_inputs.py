"""
Synthetic input data for Hamilton DAG testing.

This module provides daily resolution DataArray inputs for the pmodel, splash,
and rothc models. All DataArrays share a 2x2 lat-lon grid with 2 years of daily
data, representing reasonable UK conditions for integration testing.
"""

import numpy as np
import xarray as xr
from xarray import DataArray


def _create_time_coord(n_days: int = 731) -> np.ndarray:
    """Create time coordinate for 2 years of daily data starting 2020-01-01."""
    return np.arange("2020-01-01", "2022-01-01", dtype="datetime64[D]")


def _create_spatial_grid() -> tuple[np.ndarray, np.ndarray]:
    """Create 2x2 lat-lon grid representing UK region."""
    lat = np.array([50.0, 52.0])
    lon = np.array([-2.0, 0.0])
    return lat, lon


def _generate_seasonal_cycle(
    n_days: int, amplitude: float, phase_shift: float, baseline: float
) -> np.ndarray:
    """Generate a sinusoidal seasonal cycle."""
    t = np.arange(n_days)
    return baseline + amplitude * np.sin(2 * np.pi * t / 365.25 + phase_shift)


def dates() -> DataArray:
    """Daily dates for the time series."""
    time = _create_time_coord()
    return DataArray(
        data=time,
        dims=["time"],
        coords={"time": time},
        name="dates",
    )


def latitude() -> DataArray:
    """Latitude coordinates of the grid cells (2D for consistency with elevation)."""
    lat, lon = _create_spatial_grid()
    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing="ij")
    return DataArray(
        data=lat_2d,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={"units": "degrees_north", "long_name": "latitude"},
    )


def longitude() -> DataArray:
    """Longitude coordinates of the grid cells (2D for consistency with elevation)."""
    lat, lon = _create_spatial_grid()
    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing="ij")
    return DataArray(
        data=lon_2d,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={"units": "degrees_east", "long_name": "longitude"},
    )


def elevation() -> DataArray:
    """Elevation of the grid cells in meters."""
    lat, lon = _create_spatial_grid()
    n_lat, n_lon = len(lat), len(lon)
    elevation_data = np.full((n_lat, n_lon), 150.0)
    elevation_data[0, 0] = 100.0
    elevation_data[0, 1] = 200.0
    elevation_data[1, 0] = 250.0
    elevation_data[1, 1] = 300.0
    return DataArray(
        data=elevation_data,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={"units": "m", "long_name": "elevation"},
    )


def max_soil_moisture() -> DataArray:
    """Maximum soil moisture capacity in mm."""
    lat, lon = _create_spatial_grid()
    n_lat, n_lon = len(lat), len(lon)
    data = np.full((n_lat, n_lon), 200.0)
    return DataArray(
        data=data,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={"units": "mm", "long_name": "maximum soil moisture"},
    )


def temperature_celcius_daily() -> DataArray:
    """Daily air temperature in degrees Celsius for UK conditions."""
    lat, lon = _create_spatial_grid()
    n_lat, n_lon = len(lat), len(lon)
    n_days = 731

    base_temp = 10.0
    seasonal_amp = 10.0

    data = np.zeros((n_days, n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            lat_effect = (lat[i] - 51.0) * 0.5
            lon_effect = (lon[j] + 1.0) * 0.3
            baseline = base_temp + lat_effect + lon_effect
            seasonal = _generate_seasonal_cycle(n_days, seasonal_amp, -np.pi / 2, 0)
            daily_variation = np.random.uniform(-3, 3, n_days)
            data[:, i, j] = baseline + seasonal + daily_variation

    time = _create_time_coord()
    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"units": "degrees_C", "long_name": "air temperature"},
    )


def precipitation_mm_daily() -> DataArray:
    """Daily precipitation in mm for UK conditions."""
    lat, lon = _create_spatial_grid()
    n_lat, n_lon = len(lat), len(lon)
    n_days = 731

    np.random.seed(42)
    data = np.zeros((n_days, n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            base_precip = 2.5 + (52 - lat[i]) * 0.3
            seasonal = _generate_seasonal_cycle(n_days, 1.0, 0, 0)
            daily_precip = np.random.exponential(base_precip + seasonal, n_days)
            wet_days = np.random.random(n_days) < 0.6
            data[:, i, j] = np.where(wet_days, daily_precip, 0.0)

    time = _create_time_coord()
    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"units": "mm", "long_name": "precipitation"},
    )


def sunshine_fraction_daily() -> DataArray:
    """Daily sunshine fraction (0-1) for UK conditions."""
    lat, lon = _create_spatial_grid()
    n_lat, n_lon = len(lat), len(lon)
    n_days = 731

    data = np.zeros((n_days, n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            seasonal = _generate_seasonal_cycle(n_days, 0.3, 0, 0.5)
            noise = np.random.uniform(-0.15, 0.15, n_days)
            data[:, i, j] = np.clip(seasonal + noise, 0.0, 1.0)

    time = _create_time_coord()
    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"units": "dimensionless", "long_name": "sunshine fraction"},
    )


def pressure_pa_daily() -> DataArray:
    """Daily atmospheric pressure in Pascals."""
    lat, lon = _create_spatial_grid()
    n_lat, n_lon = len(lat), len(lon)
    n_days = 731

    baseline_pressure = 101325.0
    data = np.zeros((n_days, n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            elevation_effect = -elevation().values[i, j] * 10.0
            seasonal = _generate_seasonal_cycle(n_days, 500, 0, 0)
            noise = np.random.normal(0, 300, n_days)
            data[:, i, j] = baseline_pressure + elevation_effect + seasonal + noise

    time = _create_time_coord()
    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"units": "Pa", "long_name": "atmospheric pressure"},
    )


def vpd_pa_daily() -> DataArray:
    """Daily vapor pressure deficit in Pascals."""
    temp_da = temperature_celcius_daily()
    data = np.zeros_like(temp_da.values)

    for i in range(temp_da.shape[1]):
        for j in range(temp_da.shape[2]):
            temp = temp_da.values[:, i, j]
            svp = 610.78 * np.exp(temp / (temp + 237.3) * 17.27)
            rh = np.clip(0.5 + np.random.uniform(-0.2, 0.2, len(temp)), 0.1, 0.95)
            vpd = svp * (1 - rh)
            data[:, i, j] = np.clip(vpd, 50, 3000)

    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={
            "time": temp_da.coords["time"],
            "lat": temp_da.coords["lat"],
            "lon": temp_da.coords["lon"],
        },
        attrs={"units": "Pa", "long_name": "vapor pressure deficit"},
    )


def co2_ppm_daily() -> DataArray:
    """Daily atmospheric CO2 concentration in ppm."""
    lat, lon = _create_spatial_grid()
    n_days = 731

    baseline = 412.0
    trend = np.linspace(0, 5, n_days)
    seasonal = _generate_seasonal_cycle(n_days, 3, 0, 0)
    noise = np.random.normal(0, 1, n_days)

    data_1d = baseline + trend + seasonal + noise
    data = np.broadcast_to(
        data_1d[:, np.newaxis, np.newaxis], (n_days, len(lat), len(lon))
    )

    time = _create_time_coord()
    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"units": "ppm", "long_name": "atmospheric CO2 concentration"},
    )


def fapar_daily() -> DataArray:
    """Daily fraction of absorbed photosynthetically active radiation (0-1)."""
    lat, lon = _create_spatial_grid()
    n_lat, n_lon = len(lat), len(lon)
    n_days = 731

    data = np.zeros((n_days, n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            seasonal = _generate_seasonal_cycle(n_days, 0.25, 0, 0.55)
            noise = np.random.uniform(-0.1, 0.1, n_days)
            data[:, i, j] = np.clip(seasonal + noise, 0.05, 0.95)

    time = _create_time_coord()
    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"units": "dimensionless", "long_name": "fAPAR"},
    )


def ppfd_umol_m2_s1_daily() -> DataArray:
    """Daily photosynthetic photon flux density in µmol/m²/s."""
    lat, lon = _create_spatial_grid()
    n_lat, n_lon = len(lat), len(lon)
    n_days = 731

    data = np.zeros((n_days, n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            day_of_year = np.arange(n_days) % 365.25
            lat_rad = np.deg2rad(lat[i])
            max_ppfd = 1200 * np.sin(np.pi * day_of_year / 182.6)
            cloud_effect = 0.4 + np.random.uniform(0.2, 0.6, n_days)
            data[:, i, j] = max_ppfd * cloud_effect

    time = _create_time_coord()
    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"units": "umol/m2/s", "long_name": "photosynthetic photon flux density"},
    )


def evaporation_daily() -> DataArray:
    """Daily evaporation in mm."""
    temp_da = temperature_celcius_daily()
    sunshine_da = sunshine_fraction_daily()
    data = np.zeros_like(temp_da.values)

    for i in range(temp_da.shape[1]):
        for j in range(temp_da.shape[2]):
            temp = temp_da.values[:, i, j]
            sun = sunshine_da.values[:, i, j]
            base_evap = 0.1 + 0.3 * sun
            temp_effect = np.maximum(temp, 0) / 25.0
            data[:, i, j] = base_evap * temp_effect + np.random.uniform(
                0, 0.3, len(temp)
            )

    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={
            "time": temp_da.coords["time"],
            "lat": temp_da.coords["lat"],
            "lon": temp_da.coords["lon"],
        },
        attrs={"units": "mm", "long_name": "evaporation"},
    )

    """Daily aridity index (AET/P ratio)."""
    evap_da = evaporation_daily()
    precip_da = precipitation_mm_daily()
    precip_safe = np.where(precip_da.values < 0.1, 0.1, precip_da.values)
    data = evap_da.values / precip_safe
    data = np.clip(data, 0.01, 2.0)

    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={
            "time": evap_da.coords["time"],
            "lat": evap_da.coords["lat"],
            "lon": evap_da.coords["lon"],
        },
        attrs={"units": "dimensionless", "long_name": "aridity index"},
    )


def mean_growth_temperature_daily() -> DataArray:
    """Daily mean growth temperature in degrees Celsius."""
    temp_da = temperature_celcius_daily()
    data = temp_da.values + np.random.uniform(-1, 1, temp_da.shape)

    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={
            "time": temp_da.coords["time"],
            "lat": temp_da.coords["lat"],
            "lon": temp_da.coords["lon"],
        },
        attrs={"units": "degrees_C", "long_name": "mean growth temperature"},
    )


def plant_cover_daily() -> DataArray:
    """Daily plant cover as boolean (True = covered)."""
    temp_da = temperature_celcius_daily()
    data = temp_da.values > 3.0

    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={
            "time": temp_da.coords["time"],
            "lat": temp_da.coords["lat"],
            "lon": temp_da.coords["lon"],
        },
        attrs={"units": "dimensionless", "long_name": "plant cover"},
    )


def dpm_rpm_ratio_daily() -> DataArray:
    """Daily ratio of decomposable to resistant plant material."""
    temp_da = temperature_celcius_daily()
    n_days = temp_da.shape[0]
    lat, lon = _create_spatial_grid()

    data = np.zeros_like(temp_da.values)
    for i in range(temp_da.shape[1]):
        for j in range(temp_da.shape[2]):
            base = 0.5
            seasonal = _generate_seasonal_cycle(n_days, 0.2, np.pi, 0)
            noise = np.random.uniform(-0.1, 0.1, n_days)
            data[:, i, j] = np.clip(base + seasonal + noise, 0.1, 2.0)

    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={
            "time": temp_da.coords["time"],
            "lat": temp_da.coords["lat"],
            "lon": temp_da.coords["lon"],
        },
        attrs={"units": "dimensionless", "long_name": "DPM/RPM ratio"},
    )


def carbon_input_daily() -> DataArray:
    """Daily carbon input in tC/ha/day."""
    temp_da = temperature_celcius_daily()
    n_days = temp_da.shape[0]
    lat, lon = _create_spatial_grid()

    data = np.zeros((n_days, len(lat), len(lon)))
    for i in range(len(lat)):
        for j in range(len(lon)):
            base = 0.05
            seasonal = _generate_seasonal_cycle(n_days, 0.03, -np.pi / 4, 0)
            noise = np.random.exponential(0.01, n_days)
            data[:, i, j] = np.maximum(base + seasonal + noise, 0)

    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={"time": temp_da.coords["time"], "lat": lat, "lon": lon},
        attrs={"units": "tC/ha", "long_name": "carbon input"},
    )


def farmyard_manure_input_daily() -> DataArray:
    """Daily farmyard manure input in tC/ha/day."""
    temp_da = temperature_celcius_daily()
    n_days = temp_da.shape[0]
    lat, lon = _create_spatial_grid()

    np.random.seed(123)
    data = np.random.exponential(0.005, (n_days, len(lat), len(lon)))
    data = np.where(data > 0.02, 0.0, data)

    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={"time": temp_da.coords["time"], "lat": lat, "lon": lon},
        attrs={"units": "tC/ha", "long_name": "farmyard manure input"},
    )
