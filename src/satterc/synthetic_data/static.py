"""Generate synthetic static input data."""

import numpy as np
import pandas as pd
import xarray as xr


def lat(n_lat: int) -> xr.DataArray:
    """Latitude coordinates.

    Parameters
    ----------
    n_lat : int
        Number of latitude points.

    Returns
    -------
    xr.DataArray
        Latitude DataArray with dim "y".
    """
    return xr.DataArray(
        data=np.linspace(50.0, 54.0, n_lat),
        dims=["y"],
        coords={"y": np.arange(n_lat)},
        attrs={"units": "degrees_north", "long_name": "latitude"},
        name="lat",
    )


def lon(n_lon: int) -> xr.DataArray:
    """Longitude coordinates.

    Parameters
    ----------
    n_lon : int
        Number of longitude points.

    Returns
    -------
    xr.DataArray
        Longitude DataArray with dim "x".
    """
    return xr.DataArray(
        data=np.linspace(-4.0, 2.0, n_lon),
        dims=["x"],
        coords={"x": np.arange(n_lon)},
        attrs={"units": "degrees_east", "long_name": "longitude"},
        name="lon",
    )


def pixel_coords(lat: xr.DataArray, lon: xr.DataArray) -> pd.MultiIndex:
    """Create MultiIndex for pixel dimension from lat/lon arrays.

    Creates a cartesian product of all lat/lon combinations.

    Parameters
    ----------
    lat : xr.DataArray
        Latitude DataArray with dim "y".
    lon : xr.DataArray
        Longitude DataArray with dim "x".

    Returns
    -------
    pd.MultiIndex
        MultiIndex with 'y' and 'x' levels for each pixel.
    """
    lat_vals = lat.data
    lon_vals = lon.data

    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()

    index = pd.MultiIndex.from_arrays([lat_flat, lon_flat], names=["y", "x"])
    return index


def elevation(n_lat: int, n_lon: int, pixel_coords: pd.MultiIndex) -> xr.DataArray:
    """Static elevation in meters.

    Parameters
    ----------
    n_lat : int
        Number of latitude points.
    n_lon : int
        Number of longitude points.
    pixel_coords : pd.MultiIndex
        MultiIndex with 'y' and 'x' levels.

    Returns
    -------
    xr.DataArray
        Elevation data array with dims=["pixel"].
    """
    n_pixels = n_lat * n_lon
    elevation_data = np.full(n_pixels, 150.0)
    if n_pixels > 0:
        elevation_data[0] = 100.0
    if n_pixels > 1:
        elevation_data[1] = 200.0
    if n_pixels > 2:
        elevation_data[2] = 250.0
    if n_pixels > 3:
        elevation_data[3] = 300.0

    return xr.DataArray(
        data=elevation_data,
        dims=["pixel"],
        coords={"pixel": pixel_coords},
        attrs={"units": "m", "long_name": "elevation"},
        name="elevation",
    )


def plant_type(n_lat: int, n_lon: int, pixel_coords: pd.MultiIndex) -> xr.DataArray:
    """Static plant type (1 = grassland).

    Parameters
    ----------
    n_lat : int
        Number of latitude points.
    n_lon : int
        Number of longitude points.
    pixel_coords : pd.MultiIndex
        MultiIndex with 'y' and 'x' levels.

    Returns
    -------
    xr.DataArray
        Plant type data array with dims=["pixel"].
    """
    n_pixels = n_lat * n_lon
    data = np.full(n_pixels, 1)

    return xr.DataArray(
        data=data,
        dims=["pixel"],
        coords={"pixel": pixel_coords},
        attrs={"units": "dimensionless", "long_name": "plant type"},
        name="plant_type",
    )


def max_soil_moisture(
    n_lat: int, n_lon: int, pixel_coords: pd.MultiIndex
) -> xr.DataArray:
    """Static maximum soil moisture capacity in mm.

    Parameters
    ----------
    n_lat : int
        Number of latitude points.
    n_lon : int
        Number of longitude points.
    pixel_coords : pd.MultiIndex
        MultiIndex with 'y' and 'x' levels.

    Returns
    -------
    xr.DataArray
        Max soil moisture data array with dims=["pixel"].
    """
    n_pixels = n_lat * n_lon
    data = np.full(n_pixels, 200.0)

    return xr.DataArray(
        data=data,
        dims=["pixel"],
        coords={"pixel": pixel_coords},
        attrs={"units": "mm", "long_name": "maximum soil moisture"},
        name="max_soil_moisture",
    )


def clay_content(n_lat: int, n_lon: int, pixel_coords: pd.MultiIndex) -> xr.DataArray:
    """Static clay content fraction (0-1).

    Parameters
    ----------
    n_lat : int
        Number of latitude points.
    n_lon : int
        Number of longitude points.
    pixel_coords : pd.MultiIndex
        MultiIndex with 'y' and 'x' levels.

    Returns
    -------
    xr.DataArray
        Clay content data array with dims=["pixel"].
    """
    n_pixels = n_lat * n_lon
    data = np.random.uniform(0.1, 0.4, n_pixels)

    return xr.DataArray(
        data=data,
        dims=["pixel"],
        coords={"pixel": pixel_coords},
        attrs={"units": "fraction", "long_name": "clay content"},
        name="clay_content",
    )


def soil_depth(n_lat: int, n_lon: int, pixel_coords: pd.MultiIndex) -> xr.DataArray:
    """Static soil depth in mm.

    Parameters
    ----------
    n_lat : int
        Number of latitude points.
    n_lon : int
        Number of longitude points.
    pixel_coords : pd.MultiIndex
        MultiIndex with 'y' and 'x' levels.

    Returns
    -------
    xr.DataArray
        Soil depth data array with dims=["pixel"].
    """
    n_pixels = n_lat * n_lon
    data = np.full(n_pixels, 1000.0)

    return xr.DataArray(
        data=data,
        dims=["pixel"],
        coords={"pixel": pixel_coords},
        attrs={"units": "mm", "long_name": "soil depth"},
        name="soil_depth",
    )


def organic_carbon_stocks(
    n_lat: int, n_lon: int, pixel_coords: pd.MultiIndex
) -> xr.DataArray:
    """Static soil organic carbon stocks in tC/ha.

    Parameters
    ----------
    n_lat : int
        Number of latitude points.
    n_lon : int
        Number of longitude points.
    pixel_coords : pd.MultiIndex
        MultiIndex with 'y' and 'x' levels.

    Returns
    -------
    xr.DataArray
        Organic carbon stocks data array with dims=["pixel"].
    """
    n_pixels = n_lat * n_lon
    data = np.random.uniform(100, 150, n_pixels)

    return xr.DataArray(
        data=data,
        dims=["pixel"],
        coords={"pixel": pixel_coords},
        attrs={"units": "tC/ha", "long_name": "soil organic carbon stocks"},
        name="organic_carbon_stocks",
    )


def root_pool_init(n_lat: int, n_lon: int, pixel_coords: pd.MultiIndex) -> xr.DataArray:
    """Initial root carbon pool in tC/ha.

    Parameters
    ----------
    n_lat : int
        Number of latitude points.
    n_lon : int
        Number of longitude points.
    pixel_coords : pd.MultiIndex
        MultiIndex with 'y' and 'x' levels.

    Returns
    -------
    xr.DataArray
        Root pool init data array with dims=["pixel"].
    """
    n_pixels = n_lat * n_lon
    data = np.full(n_pixels, 5.0)

    return xr.DataArray(
        data=data,
        dims=["pixel"],
        coords={"pixel": pixel_coords},
        attrs={"units": "tC/ha", "long_name": "initial root carbon pool"},
        name="root_pool_init",
    )


def leaf_pool_init(n_lat: int, n_lon: int, pixel_coords: pd.MultiIndex) -> xr.DataArray:
    """Initial leaf carbon pool in tC/ha.

    Parameters
    ----------
    n_lat : int
        Number of latitude points.
    n_lon : int
        Number of longitude points.
    pixel_coords : pd.MultiIndex
        MultiIndex with 'y' and 'x' levels.

    Returns
    -------
    xr.DataArray
        Leaf pool init data array with dims=["pixel"].
    """
    n_pixels = n_lat * n_lon
    data = np.full(n_pixels, 1.0)

    return xr.DataArray(
        data=data,
        dims=["pixel"],
        coords={"pixel": pixel_coords},
        attrs={"units": "tC/ha", "long_name": "initial leaf carbon pool"},
        name="leaf_pool_init",
    )


def stem_pool_init(n_lat: int, n_lon: int, pixel_coords: pd.MultiIndex) -> xr.DataArray:
    """Initial stem carbon pool in tC/ha.

    Parameters
    ----------
    n_lat : int
        Number of latitude points.
    n_lon : int
        Number of longitude points.
    pixel_coords : pd.MultiIndex
        MultiIndex with 'y' and 'x' levels.

    Returns
    -------
    xr.DataArray
        Stem pool init data array with dims=["pixel"].
    """
    n_pixels = n_lat * n_lon
    data = np.full(n_pixels, 10.0)

    return xr.DataArray(
        data=data,
        dims=["pixel"],
        coords={"pixel": pixel_coords},
        attrs={"units": "tC/ha", "long_name": "initial stem carbon pool"},
        name="stem_pool_init",
    )
