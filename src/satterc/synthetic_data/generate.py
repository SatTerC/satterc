"""
Generate synthetic input data for Hamilton DAG testing.

Creates daily, weekly, monthly, and static NetCDF files with physically plausible
data for a 2x2 lat-lon grid in the UK spanning 3 years (2020-2022).

Output files:
    - daily.nc: Daily resolution inputs (precipitation, sunshine, temperature, LAI, GPP)
    - weekly.nc: Weekly resolution inputs (CO2, FAPAR, PPFD, pressure, VPD)
    - monthly.nc: Monthly resolution inputs (dummy_variable)
    - static.nc: Static inputs (elevation, plant_type, soil properties, carbon pools)
"""

from pathlib import Path

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


def spatial_grid(
    n_lat: int = 2,
    n_lon: int = 2,
    lat_min: float = 50.0,
    lat_max: float = 54.0,
    lon_min: float = -4.0,
    lon_max: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create lat-lon grid."""
    lat = np.linspace(lat_min, lat_max, n_lat)
    lon = np.linspace(lon_min, lon_max, n_lon)
    return lat, lon


def random_seed_config(seed: int = 42) -> int:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    return seed


def temperature_celcius_daily(
    time_coord: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
) -> xr.DataArray:
    """Daily air temperature in degrees Celsius for UK conditions."""
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
    )


def precipitation_mm_daily(
    time_coord: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
) -> xr.DataArray:
    """Daily precipitation in mm for UK conditions."""
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
    )


def sunshine_fraction_daily(
    time_coord: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
) -> xr.DataArray:
    """Daily sunshine fraction (0-1) for UK conditions."""
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
    )


def lai_daily_func(
    time_coord: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
) -> xr.DataArray:
    """Daily Leaf Area Index (m²/m²) for UK conditions."""
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
    )


def gpp_daily_func(
    time_coord: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    temperature: xr.DataArray,
) -> xr.DataArray:
    """Daily Gross Primary Productivity (gC/m²/d)."""
    n_days = len(time_coord)
    n_lat, n_lon = len(lat), len(lon)

    data = np.zeros((n_days, n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            base_gpp = 8.0
            seasonal = _generate_seasonal_cycle(n_days, 5.0, -np.pi / 3, 0)
            temp_factor = np.maximum(temperature.values[:, i, j] - 5, 0) / 15.0
            noise = np.random.uniform(-1, 1, n_days)
            data[:, i, j] = np.maximum(base_gpp + seasonal * temp_factor + noise, 0.1)

    return xr.DataArray(
        data=data,
        dims=["time", "y", "x"],
        coords={"time": time_coord, "y": lat, "x": lon},
        attrs={"units": "gC/m2/d", "long_name": "gross primary productivity"},
    )


def co2_ppm_weekly_func(
    time_coord: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
) -> xr.DataArray:
    """Weekly atmospheric CO2 concentration in ppm."""
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
    )


def fapar_weekly_func(
    time_coord: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
) -> xr.DataArray:
    """Weekly fraction of absorbed photosynthetically active radiation (0-1)."""
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
    )


def ppfd_weekly_func(
    time_coord: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
) -> xr.DataArray:
    """Weekly photosynthetic photon flux density in µmol/m²/s."""
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
    )


def pressure_pa_weekly_func(
    time_coord: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    elevation: NDArray[np.float64],
) -> xr.DataArray:
    """Weekly atmospheric pressure in Pascals."""
    n_days = len(time_coord)
    n_lat, n_lon = len(lat), len(lon)

    baseline_pressure = 101325.0
    data = np.zeros((n_days, n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            elevation_effect = -elevation[i, j] * 10.0
            seasonal = _generate_seasonal_cycle(n_days, 500, 0, 0)
            noise = np.random.normal(0, 300, n_days)
            data[:, i, j] = baseline_pressure + elevation_effect + seasonal + noise

    return xr.DataArray(
        data=data,
        dims=["time", "y", "x"],
        coords={"time": time_coord, "y": lat, "x": lon},
        attrs={"units": "Pa", "long_name": "atmospheric pressure"},
    )


def vpd_pa_weekly_func(
    time_coord: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    temperature: xr.DataArray,
) -> xr.DataArray:
    """Weekly vapor pressure deficit in Pascals."""
    temp_da = temperature
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
    )


def elevation_static(
    lat: NDArray[np.float64], lon: NDArray[np.float64]
) -> xr.DataArray:
    """Static elevation in meters."""
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

    return xr.DataArray(
        data=elevation_data,
        dims=["y", "x"],
        coords={"y": lat, "x": lon},
        attrs={"units": "m", "long_name": "elevation"},
    )


def plant_type_static(
    lat: NDArray[np.float64], lon: NDArray[np.float64]
) -> xr.DataArray:
    """Static plant type (1 = grassland)."""
    n_lat, n_lon = len(lat), len(lon)
    data = np.full((n_lat, n_lon), 1)

    return xr.DataArray(
        data=data,
        dims=["y", "x"],
        coords={"y": lat, "x": lon},
        attrs={"units": "dimensionless", "long_name": "plant type"},
    )


def max_soil_moisture_static(
    lat: NDArray[np.float64], lon: NDArray[np.float64]
) -> xr.DataArray:
    """Static maximum soil moisture capacity in mm."""
    n_lat, n_lon = len(lat), len(lon)
    data = np.full((n_lat, n_lon), 200.0)

    return xr.DataArray(
        data=data,
        dims=["y", "x"],
        coords={"y": lat, "x": lon},
        attrs={"units": "mm", "long_name": "maximum soil moisture"},
    )


def clay_content_static(
    lat: NDArray[np.float64], lon: NDArray[np.float64]
) -> xr.DataArray:
    """Static clay content fraction (0-1)."""
    n_lat, n_lon = len(lat), len(lon)
    np.random.seed(42)
    data = np.random.uniform(0.1, 0.4, (n_lat, n_lon))

    return xr.DataArray(
        data=data,
        dims=["y", "x"],
        coords={"y": lat, "x": lon},
        attrs={"units": "fraction", "long_name": "clay content"},
    )


def soil_depth_static(
    lat: NDArray[np.float64], lon: NDArray[np.float64]
) -> xr.DataArray:
    """Static soil depth in mm."""
    n_lat, n_lon = len(lat), len(lon)
    data = np.full((n_lat, n_lon), 1000.0)

    return xr.DataArray(
        data=data,
        dims=["y", "x"],
        coords={"y": lat, "x": lon},
        attrs={"units": "mm", "long_name": "soil depth"},
    )


def organic_carbon_stocks_static(
    lat: NDArray[np.float64], lon: NDArray[np.float64]
) -> xr.DataArray:
    """Static soil organic carbon stocks in tC/ha."""
    n_lat, n_lon = len(lat), len(lon)
    np.random.seed(43)
    data = np.random.uniform(100, 150, (n_lat, n_lon))

    return xr.DataArray(
        data=data,
        dims=["y", "x"],
        coords={"y": lat, "x": lon},
        attrs={"units": "tC/ha", "long_name": "soil organic carbon stocks"},
    )


def root_pool_init_static(
    lat: NDArray[np.float64], lon: NDArray[np.float64]
) -> xr.DataArray:
    """Initial root carbon pool in tC/ha."""
    n_lat, n_lon = len(lat), len(lon)
    data = np.full((n_lat, n_lon), 5.0)

    return xr.DataArray(
        data=data,
        dims=["y", "x"],
        coords={"y": lat, "x": lon},
        attrs={"units": "tC/ha", "long_name": "initial root carbon pool"},
    )


def leaf_pool_init_static(
    lat: NDArray[np.float64], lon: NDArray[np.float64]
) -> xr.DataArray:
    """Initial leaf carbon pool in tC/ha."""
    n_lat, n_lon = len(lat), len(lon)
    data = np.full((n_lat, n_lon), 1.0)

    return xr.DataArray(
        data=data,
        dims=["y", "x"],
        coords={"y": lat, "x": lon},
        attrs={"units": "tC/ha", "long_name": "initial leaf carbon pool"},
    )


def stem_pool_init_static(
    lat: NDArray[np.float64], lon: NDArray[np.float64]
) -> xr.DataArray:
    """Initial stem carbon pool in tC/ha."""
    n_lat, n_lon = len(lat), len(lon)
    data = np.full((n_lat, n_lon), 10.0)

    return xr.DataArray(
        data=data,
        dims=["y", "x"],
        coords={"y": lat, "x": lon},
        attrs={"units": "tC/ha", "long_name": "initial stem carbon pool"},
    )


def add_crs_metadata(ds: xr.Dataset) -> xr.Dataset:
    """Add CRS metadata to dataset for rioxarray compatibility."""
    ds = ds.assign_attrs({"crs": "EPSG:4326"})
    for coord in ["x", "y"]:
        if coord in ds.coords:
            ds[coord].attrs["crs"] = "EPSG:4326"
    return ds


def generate_all_synthetic_data(output_dir: str | None = None):
    """Generate all synthetic input files."""
    import os

    if output_dir is None:
        output_dir = str(Path(__file__).parent / "data")

    os.makedirs(output_dir, exist_ok=True)

    random_seed_config(42)

    lat, lon = spatial_grid()
    n_days = 3 * 365 + 1
    daily_time = time_coord(n_days)

    print("Generating daily variables...")
    temp_daily = temperature_celcius_daily(daily_time, lat, lon)
    precip_daily = precipitation_mm_daily(daily_time, lat, lon)
    sunshine_daily = sunshine_fraction_daily(daily_time, lat, lon)
    lai_vals = lai_daily_func(daily_time, lat, lon)
    gpp_vals = gpp_daily_func(daily_time, lat, lon, temp_daily)

    daily_ds = xr.Dataset(
        data_vars={
            "temperature_celcius": temp_daily,
            "precipitation_mm": precip_daily,
            "sunshine_fraction": sunshine_daily,
            "lai": lai_vals,
            "gpp": gpp_vals,
        }
    )
    daily_ds = add_crs_metadata(daily_ds)

    print("Aggregating to weekly...")
    weekly_ds = daily_ds.resample(time="1W").mean()
    weekly_ds = weekly_ds.drop_vars(["lai", "gpp"])

    print("Generating weekly variables...")
    co2_daily = co2_ppm_weekly_func(daily_time, lat, lon)
    fapar_daily = fapar_weekly_func(daily_time, lat, lon)
    ppfd_daily = ppfd_weekly_func(daily_time, lat, lon)

    elevation_arr = elevation_static(lat, lon)
    pressure_daily = pressure_pa_weekly_func(daily_time, lat, lon, elevation_arr.values)
    vpd_daily = vpd_pa_weekly_func(daily_time, lat, lon, temp_daily)

    weekly_time = weekly_ds.time

    weekly_ds = weekly_ds.assign(
        {
            "co2_ppm": co2_daily.resample(time="1W")
            .mean()
            .assign_coords(time=weekly_time),
            "fapar": fapar_daily.resample(time="1W")
            .mean()
            .assign_coords(time=weekly_time),
            "ppfd_umol_m2_s1": ppfd_daily.resample(time="1W")
            .mean()
            .assign_coords(time=weekly_time),
            "pressure_pa": pressure_daily.resample(time="1W")
            .mean()
            .assign_coords(time=weekly_time),
            "vpd_pa": vpd_daily.resample(time="1W")
            .mean()
            .assign_coords(time=weekly_time),
        }
    )
    weekly_ds = add_crs_metadata(weekly_ds)

    print("Aggregating to monthly...")
    monthly_time = pd.date_range(start="2020-01-01", end="2022-12-31", freq="ME")
    monthly_ds = daily_ds.resample(time="1ME").mean()
    monthly_ds = monthly_ds.assign_coords(time=monthly_time)
    monthly_ds = monthly_ds.drop_vars(
        ["temperature_celcius", "precipitation_mm", "lai", "gpp"]
    )

    dummy_var = xr.DataArray(
        data=np.ones((len(monthly_time), len(lat), len(lon))),
        dims=["time", "y", "x"],
        coords={"time": monthly_time, "y": lat, "x": lon},
        attrs={"units": "dimensionless", "long_name": "dummy variable"},
    )
    monthly_ds = monthly_ds.assign(dummy_variable=dummy_var)
    monthly_ds = add_crs_metadata(monthly_ds)

    print("Generating static variables...")
    static_ds = xr.Dataset(
        data_vars={
            "elevation": elevation_static(lat, lon),
            "plant_type": plant_type_static(lat, lon),
            "max_soil_moisture": max_soil_moisture_static(lat, lon),
            "clay_content": clay_content_static(lat, lon),
            "soil_depth": soil_depth_static(lat, lon),
            "organic_carbon_stocks": organic_carbon_stocks_static(lat, lon),
            "root_pool_init": root_pool_init_static(lat, lon),
            "leaf_pool_init": leaf_pool_init_static(lat, lon),
            "stem_pool_init": stem_pool_init_static(lat, lon),
        }
    )
    static_ds = add_crs_metadata(static_ds)

    print(f"Writing daily.nc ({len(daily_ds.time)} days)...")
    daily_ds.to_netcdf(f"{output_dir}/daily.nc", format="NETCDF3_CLASSIC")
    print(f"Writing weekly.nc ({len(weekly_ds.time)} weeks)...")
    weekly_ds.to_netcdf(f"{output_dir}/weekly.nc", format="NETCDF3_CLASSIC")
    print(f"Writing monthly.nc ({len(monthly_ds.time)} months)...")
    monthly_ds.to_netcdf(f"{output_dir}/monthly.nc", format="NETCDF3_CLASSIC")
    print("Writing static.nc...")
    static_ds.to_netcdf(f"{output_dir}/static.nc", format="NETCDF3_CLASSIC")

    print("Done!")


if __name__ == "__main__":
    generate_all_synthetic_data()
