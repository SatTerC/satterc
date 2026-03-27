"""Generate synthetic static input data."""

from os import PathLike
from pathlib import Path

import numpy as np
import xarray as xr
from hamilton.function_modifiers import group, inject, source
from hamilton.function_modifiers.delayed import ResolveAt

from ..pipeline._hamilton_fixes import FixedResolve, NoOpDecorator


def _add_crs_metadata(ds: xr.Dataset) -> xr.Dataset:
    """Add CRS metadata to dataset for rioxarray compatibility."""
    ds = ds.assign_attrs({"crs": "EPSG:4326"})
    for coord in ["x", "y"]:
        if coord in ds.coords:
            ds[coord].attrs["crs"] = "EPSG:4326"
    return ds


def elevation(lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """Static elevation in meters.

    Parameters
    ----------
    lat : np.ndarray
        Latitude array.
    lon : np.ndarray
        Longitude array.

    Returns
    -------
    xr.DataArray
        Elevation data array.
    """
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
        name="elevation",
    )


def plant_type(lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """Static plant type (1 = grassland).

    Parameters
    ----------
    lat : np.ndarray
        Latitude array.
    lon : np.ndarray
        Longitude array.

    Returns
    -------
    xr.DataArray
        Plant type data array.
    """
    n_lat, n_lon = len(lat), len(lon)
    data = np.full((n_lat, n_lon), 1)

    return xr.DataArray(
        data=data,
        dims=["y", "x"],
        coords={"y": lat, "x": lon},
        attrs={"units": "dimensionless", "long_name": "plant type"},
        name="plant_type",
    )


def max_soil_moisture(lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """Static maximum soil moisture capacity in mm.

    Parameters
    ----------
    lat : np.ndarray
        Latitude array.
    lon : np.ndarray
        Longitude array.

    Returns
    -------
    xr.DataArray
        Max soil moisture data array.
    """
    n_lat, n_lon = len(lat), len(lon)
    data = np.full((n_lat, n_lon), 200.0)

    return xr.DataArray(
        data=data,
        dims=["y", "x"],
        coords={"y": lat, "x": lon},
        attrs={"units": "mm", "long_name": "maximum soil moisture"},
        name="max_soil_moisture",
    )


def clay_content(lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """Static clay content fraction (0-1).

    Parameters
    ----------
    lat : np.ndarray
        Latitude array.
    lon : np.ndarray
        Longitude array.

    Returns
    -------
    xr.DataArray
        Clay content data array.
    """
    n_lat, n_lon = len(lat), len(lon)
    data = np.random.uniform(0.1, 0.4, (n_lat, n_lon))

    return xr.DataArray(
        data=data,
        dims=["y", "x"],
        coords={"y": lat, "x": lon},
        attrs={"units": "fraction", "long_name": "clay content"},
        name="clay_content",
    )


def soil_depth(lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """Static soil depth in mm.

    Parameters
    ----------
    lat : np.ndarray
        Latitude array.
    lon : np.ndarray
        Longitude array.

    Returns
    -------
    xr.DataArray
        Soil depth data array.
    """
    n_lat, n_lon = len(lat), len(lon)
    data = np.full((n_lat, n_lon), 1000.0)

    return xr.DataArray(
        data=data,
        dims=["y", "x"],
        coords={"y": lat, "x": lon},
        attrs={"units": "mm", "long_name": "soil depth"},
        name="soil_depth",
    )


def organic_carbon_stocks(lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """Static soil organic carbon stocks in tC/ha.

    Parameters
    ----------
    lat : np.ndarray
        Latitude array.
    lon : np.ndarray
        Longitude array.

    Returns
    -------
    xr.DataArray
        Organic carbon stocks data array.
    """
    n_lat, n_lon = len(lat), len(lon)
    data = np.random.uniform(100, 150, (n_lat, n_lon))

    return xr.DataArray(
        data=data,
        dims=["y", "x"],
        coords={"y": lat, "x": lon},
        attrs={"units": "tC/ha", "long_name": "soil organic carbon stocks"},
        name="organic_carbon_stocks",
    )


def root_pool_init(lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """Initial root carbon pool in tC/ha.

    Parameters
    ----------
    lat : np.ndarray
        Latitude array.
    lon : np.ndarray
        Longitude array.

    Returns
    -------
    xr.DataArray
        Root pool init data array.
    """
    n_lat, n_lon = len(lat), len(lon)
    data = np.full((n_lat, n_lon), 5.0)

    return xr.DataArray(
        data=data,
        dims=["y", "x"],
        coords={"y": lat, "x": lon},
        attrs={"units": "tC/ha", "long_name": "initial root carbon pool"},
        name="root_pool_init",
    )


def leaf_pool_init(lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """Initial leaf carbon pool in tC/ha.

    Parameters
    ----------
    lat : np.ndarray
        Latitude array.
    lon : np.ndarray
        Longitude array.

    Returns
    -------
    xr.DataArray
        Leaf pool init data array.
    """
    n_lat, n_lon = len(lat), len(lon)
    data = np.full((n_lat, n_lon), 1.0)

    return xr.DataArray(
        data=data,
        dims=["y", "x"],
        coords={"y": lat, "x": lon},
        attrs={"units": "tC/ha", "long_name": "initial leaf carbon pool"},
        name="leaf_pool_init",
    )


def stem_pool_init(lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """Initial stem carbon pool in tC/ha.

    Parameters
    ----------
    lat : np.ndarray
        Latitude array.
    lon : np.ndarray
        Longitude array.

    Returns
    -------
    xr.DataArray
        Stem pool init data array.
    """
    n_lat, n_lon = len(lat), len(lon)
    data = np.full((n_lat, n_lon), 10.0)

    return xr.DataArray(
        data=data,
        dims=["y", "x"],
        coords={"y": lat, "x": lon},
        attrs={"units": "tC/ha", "long_name": "initial stem carbon pool"},
        name="stem_pool_init",
    )


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda static_outputs_vars: (
        inject(static_outputs_list=group(*[source(var) for var in static_outputs_vars]))
        if static_outputs_vars
        else NoOpDecorator()
    ),
)
def merged_static_outputs(
    static_outputs_list: list[xr.DataArray],
    static_outputs_vars: list[str],
) -> xr.Dataset:
    """Merge static output data arrays into a single dataset.

    Parameters
    ----------
    static_outputs_list : list[xr.DataArray]
        List of static output data arrays.
    static_outputs_vars : list[str]
        List of variable names (resolved from config).

    Returns
    -------
    xr.Dataset
        Merged dataset.
    """
    return xr.merge(static_outputs_list)


def save_static_outputs(
    merged_static_outputs: xr.Dataset, static_outputs_path: str | PathLike
) -> None:
    """Save static outputs to file.

    Parameters
    ----------
    merged_static_outputs : xr.Dataset
        Static outputs dataset.
    static_outputs_path : str | PathLike
        Path to save the dataset.
    """
    p = Path(static_outputs_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _add_crs_metadata(merged_static_outputs).to_netcdf(
        static_outputs_path, format="NETCDF3_CLASSIC"
    )
