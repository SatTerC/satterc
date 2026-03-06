"""
Synthetic input data for Hamilton DAG testing.

This module provides daily resolution DataArray inputs for the pmodel, splash,
and rothc models. All DataArrays share a configurable lat-lon grid with
configurable daily data, representing reasonable UK conditions for
integration testing.

Configuration Parameters (pass via driver config):
    n_days: Number of days (default: 731)
    n_lat: Number of latitude points (default: 2)
    n_lon: Number of longitude points (default: 2)
    lat_min: Minimum latitude (default: 50.0)
    lat_max: Maximum latitude (default: 54.0)
    lon_min: Minimum longitude (default: -4.0)
    lon_max: Maximum longitude (default: 2.0)
    random_seed: Random seed for reproducibility (default: 42)
"""

import numpy as np
from numpy.typing import NDArray
from xarray import DataArray


def time_coord(
    n_days: int | None = None,
    start_date: str | None = None,
) -> NDArray[np.datetime64]:
    """Create time coordinate - configurable via n_days config.

    Parameters
    ----------
    n_days
        Number of days for the time series.
    start_date
        Start date for the time series (ISO format string).

    Returns
    -------
    NDArray[np.datetime64]
        Array of datetime64[D] values.
    """
    if n_days is None:
        n_days = 731
    if start_date is None:
        start_date = "2020-01-01"
    start = np.datetime64(start_date)
    return start + np.arange(n_days)


def spatial_grid(
    n_lat: int | None = None,
    n_lon: int | None = None,
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create lat-lon grid - configurable via n_lat, n_lon, and bounds.

    Parameters
    ----------
    n_lat
        Number of latitude points.
    n_lon
        Number of longitude points.
    lat_min
        Minimum latitude (degrees north).
    lat_max
        Maximum latitude (degrees north).
    lon_min
        Minimum longitude (degrees east).
    lon_max
        Maximum longitude (degrees east).

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        Tuple of (lat, lon) coordinate arrays.
    """
    if n_lat is None:
        n_lat = 2
    if n_lon is None:
        n_lon = 2
    if lat_min is None:
        lat_min = 50.0
    if lat_max is None:
        lat_max = 54.0
    if lon_min is None:
        lon_min = -4.0
    if lon_max is None:
        lon_max = 2.0
    lat = np.linspace(lat_min, lat_max, n_lat)
    lon = np.linspace(lon_min, lon_max, n_lon)
    return lat, lon


def random_seed_config(random_seed: int | None = None) -> int:
    """Random seed for reproducible synthetic data generation.

    Parameters
    ----------
    random_seed
        Integer seed for numpy random number generator.

    Returns
    -------
    int
        The random seed value.
    """
    if random_seed is None:
        random_seed = 42
    np.random.seed(random_seed)
    return random_seed


def _generate_seasonal_cycle(
    n_days: int, amplitude: float, phase_shift: float, baseline: float
) -> NDArray[np.float64]:
    """Generate a sinusoidal seasonal cycle."""
    t = np.arange(n_days)
    return baseline + amplitude * np.sin(2 * np.pi * t / 365.25 + phase_shift)


def dates(time_coord: NDArray[np.datetime64]) -> DataArray:
    """Daily dates for the time series."""
    return DataArray(
        data=time_coord,
        dims=["time"],
        coords={"time": time_coord},
        name="dates",
    )


def latitude(
    spatial_grid: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> DataArray:
    """Latitude coordinates of the grid cells (2D for consistency with elevation)."""
    lat, lon = spatial_grid
    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing="ij")
    return DataArray(
        data=lat_2d,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={"units": "degrees_north", "long_name": "latitude"},
    )


def longitude(
    spatial_grid: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> DataArray:
    """Longitude coordinates of the grid cells (2D for consistency with elevation)."""
    lat, lon = spatial_grid
    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing="ij")
    return DataArray(
        data=lon_2d,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={"units": "degrees_east", "long_name": "longitude"},
    )


def elevation(
    spatial_grid: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> DataArray:
    """Elevation of the grid cells in meters."""
    lat, lon = spatial_grid
    n_lat, n_lon = len(lat), len(lon)
    elevation_data = np.full((n_lat, n_lon), 150.0)
    if n_lat > 1 and n_lon > 1:
        elevation_data[0, 0] = 100.0
    if n_lon > 1:
        elevation_data[0, min(1, n_lon - 1)] = 200.0
    if n_lat > 1:
        elevation_data[min(1, n_lat - 1), 0] = 250.0
    if n_lat > 1 and n_lon > 1:
        elevation_data[min(1, n_lat - 1), min(1, n_lon - 1)] = 300.0
    return DataArray(
        data=elevation_data,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={"units": "m", "long_name": "elevation"},
    )


def max_soil_moisture(
    spatial_grid: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> DataArray:
    """Maximum soil moisture capacity in mm."""
    lat, lon = spatial_grid
    n_lat, n_lon = len(lat), len(lon)
    data = np.full((n_lat, n_lon), 200.0)
    return DataArray(
        data=data,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={"units": "mm", "long_name": "maximum soil moisture"},
    )


def temperature_celcius_daily(
    time_coord: NDArray[np.datetime64],
    spatial_grid: tuple[NDArray[np.float64], NDArray[np.float64]],
    random_seed_config: int,
) -> DataArray:
    """Daily air temperature in degrees Celsius for UK conditions."""
    lat, lon = spatial_grid
    n_lat, n_lon = len(lat), len(lon)
    n_days = len(time_coord)

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

    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={"time": time_coord, "lat": lat, "lon": lon},
        attrs={"units": "degrees_C", "long_name": "air temperature"},
    )


def precipitation_mm_daily(
    time_coord: NDArray[np.datetime64],
    spatial_grid: tuple[NDArray[np.float64], NDArray[np.float64]],
    random_seed_config: int,
) -> DataArray:
    """Daily precipitation in mm for UK conditions."""
    lat, lon = spatial_grid
    n_lat, n_lon = len(lat), len(lon)
    n_days = len(time_coord)

    data = np.zeros((n_days, n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            base_precip = 2.5 + (54 - lat[i]) * 0.3
            seasonal = _generate_seasonal_cycle(n_days, 1.0, 0, 0)
            daily_precip = np.random.exponential(base_precip + seasonal, n_days)
            wet_days = np.random.random(n_days) < 0.6
            data[:, i, j] = np.where(wet_days, daily_precip, 0.0)

    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={"time": time_coord, "lat": lat, "lon": lon},
        attrs={"units": "mm", "long_name": "precipitation"},
    )


def sunshine_fraction_daily(
    time_coord: NDArray[np.datetime64],
    spatial_grid: tuple[NDArray[np.float64], NDArray[np.float64]],
    random_seed_config: int,
) -> DataArray:
    """Daily sunshine fraction (0-1) for UK conditions."""
    lat, lon = spatial_grid
    n_lat, n_lon = len(lat), len(lon)
    n_days = len(time_coord)

    data = np.zeros((n_days, n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            seasonal = _generate_seasonal_cycle(n_days, 0.3, 0, 0.5)
            noise = np.random.uniform(-0.15, 0.15, n_days)
            data[:, i, j] = np.clip(seasonal + noise, 0.0, 1.0)

    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={"time": time_coord, "lat": lat, "lon": lon},
        attrs={"units": "dimensionless", "long_name": "sunshine fraction"},
    )


def pressure_pa_daily(
    time_coord: NDArray[np.datetime64],
    spatial_grid: tuple[NDArray[np.float64], NDArray[np.float64]],
    elevation: DataArray,
    random_seed_config: int,
) -> DataArray:
    """Daily atmospheric pressure in Pascals."""
    lat, lon = spatial_grid
    n_lat, n_lon = len(lat), len(lon)
    n_days = len(time_coord)

    baseline_pressure = 101325.0
    data = np.zeros((n_days, n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            elevation_effect = -elevation.values[i, j] * 10.0
            seasonal = _generate_seasonal_cycle(n_days, 500, 0, 0)
            noise = np.random.normal(0, 300, n_days)
            data[:, i, j] = baseline_pressure + elevation_effect + seasonal + noise

    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={"time": time_coord, "lat": lat, "lon": lon},
        attrs={"units": "Pa", "long_name": "atmospheric pressure"},
    )


def vpd_pa_daily(
    temperature_celcius_daily: DataArray,
    random_seed_config: int,
) -> DataArray:
    """Daily vapor pressure deficit in Pascals."""
    temp_da = temperature_celcius_daily
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


def co2_ppm_daily(
    time_coord: NDArray[np.datetime64],
    spatial_grid: tuple[NDArray[np.float64], NDArray[np.float64]],
    random_seed_config: int,
) -> DataArray:
    """Daily atmospheric CO2 concentration in ppm."""
    lat, lon = spatial_grid
    n_days = len(time_coord)

    baseline = 412.0
    trend = np.linspace(0, 5, n_days)
    seasonal = _generate_seasonal_cycle(n_days, 3, 0, 0)
    noise = np.random.normal(0, 1, n_days)

    data_1d = baseline + trend + seasonal + noise
    data = np.broadcast_to(
        data_1d[:, np.newaxis, np.newaxis], (n_days, len(lat), len(lon))
    )

    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={"time": time_coord, "lat": lat, "lon": lon},
        attrs={"units": "ppm", "long_name": "atmospheric CO2 concentration"},
    )


def fapar_daily(
    time_coord: NDArray[np.datetime64],
    spatial_grid: tuple[NDArray[np.float64], NDArray[np.float64]],
    random_seed_config: int,
) -> DataArray:
    """Daily fraction of absorbed photosynthetically active radiation (0-1)."""
    lat, lon = spatial_grid
    n_lat, n_lon = len(lat), len(lon)
    n_days = len(time_coord)

    data = np.zeros((n_days, n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            seasonal = _generate_seasonal_cycle(n_days, 0.25, 0, 0.55)
            noise = np.random.uniform(-0.1, 0.1, n_days)
            data[:, i, j] = np.clip(seasonal + noise, 0.05, 0.95)

    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={"time": time_coord, "lat": lat, "lon": lon},
        attrs={"units": "dimensionless", "long_name": "fAPAR"},
    )


def ppfd_umol_m2_s1_daily(
    time_coord: NDArray[np.datetime64],
    spatial_grid: tuple[NDArray[np.float64], NDArray[np.float64]],
    random_seed_config: int,
) -> DataArray:
    """Daily photosynthetic photon flux density in µmol/m²/s."""
    lat, lon = spatial_grid
    n_lat, n_lon = len(lat), len(lon)
    n_days = len(time_coord)

    data = np.zeros((n_days, n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            day_of_year = np.arange(n_days) % 365.25
            max_ppfd = 1200 * np.sin(np.pi * day_of_year / 182.6)
            cloud_effect = 0.4 + np.random.uniform(0.2, 0.6, n_days)
            data[:, i, j] = max_ppfd * cloud_effect

    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={"time": time_coord, "lat": lat, "lon": lon},
        attrs={"units": "umol/m2/s", "long_name": "photosynthetic photon flux density"},
    )


def evaporation_daily(
    temperature_celcius_daily: DataArray,
    sunshine_fraction_daily: DataArray,
    random_seed_config: int,
) -> DataArray:
    """Daily evaporation in mm."""
    temp_da = temperature_celcius_daily
    sunshine_da = sunshine_fraction_daily
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


def aridity_index_daily(
    evaporation_daily: DataArray,
    precipitation_mm_daily: DataArray,
) -> DataArray:
    """Daily aridity index (AET/P ratio)."""
    evap_da = evaporation_daily
    precip_da = precipitation_mm_daily
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


def mean_growth_temperature_daily(
    temperature_celcius_daily: DataArray,
    random_seed_config: int,
) -> DataArray:
    """Daily mean growth temperature in degrees Celsius."""
    temp_da = temperature_celcius_daily
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


def plant_cover_daily(temperature_celcius_daily: DataArray) -> DataArray:
    """Daily plant cover as boolean (True = covered)."""
    temp_da = temperature_celcius_daily
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


def dpm_rpm_ratio_daily(
    temperature_celcius_daily: DataArray,
    spatial_grid: tuple[NDArray[np.float64], NDArray[np.float64]],
    random_seed_config: int,
) -> DataArray:
    """Daily ratio of decomposable to resistant plant material."""
    temp_da = temperature_celcius_daily
    n_days = temp_da.shape[0]
    lat, lon = spatial_grid

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


def carbon_input_daily(
    temperature_celcius_daily: DataArray,
    spatial_grid: tuple[NDArray[np.float64], NDArray[np.float64]],
    random_seed_config: int,
) -> DataArray:
    """Daily carbon input in tC/ha/day."""
    temp_da = temperature_celcius_daily
    n_days = temp_da.shape[0]
    lat, lon = spatial_grid

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


def farmyard_manure_input_daily(
    temperature_celcius_daily: DataArray,
    spatial_grid: tuple[NDArray[np.float64], NDArray[np.float64]],
    random_seed_config: int,
) -> DataArray:
    """Daily farmyard manure input in tC/ha/day."""
    temp_da = temperature_celcius_daily
    n_days = temp_da.shape[0]
    lat, lon = spatial_grid

    data = np.random.exponential(0.005, (n_days, len(lat), len(lon)))
    data = np.where(data > 0.02, 0.0, data)

    return DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={"time": temp_da.coords["time"], "lat": lat, "lon": lon},
        attrs={"units": "tC/ha", "long_name": "farmyard manure input"},
    )
