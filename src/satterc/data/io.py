"""
I/O module for loading geospatial data using rioxarray.

This module provides Hamilton nodes for loading NetCDF/Zarr datasets
and extracting latitude/longitude grids with spatial coordinates stacked
into a single "pixel" dimension.
"""

from pathlib import Path

from hamilton.function_modifiers import unpack_fields
import numpy as np
import xarray as xr

import rioxarray as rioxarray  # only used indirectly via dataset.rio; suppress linter errors
from pyproj import Transformer


def raw_dataset(path: str) -> xr.Dataset:
    """
    Loads data. decode_coords="all" is critical for rioxarray
    to find the CRS in NetCDF/Zarr metadata.

    Parameters
    ----------
    path : str
        Path to the NetCDF or Zarr dataset.

    Returns
    -------
    xr.Dataset
        The loaded dataset with coordinates decoded.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in [".nc", ".netcdf"]:
        engine = "netcdf4"
    elif suffix == ".zarr" or p.is_dir():
        engine = "zarr"
    else:
        raise ValueError(f"Unsupported file extension: {p.suffix}.")

    return xr.open_dataset(path, engine=engine, decode_coords="all")


@unpack_fields("latitude", "longitude")
def latitude_longitude(raw_dataset: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Computes both latitude and longitude grids and unpacks them
    into individual nodes in the Hamilton DAG.

    Spatial coordinates are stacked into a single "pixel" dimension.

    Parameters
    ----------
    raw_dataset : xr.Dataset
        The loaded dataset with coordinate reference system information.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        Dictionary with "latitude" and "longitude" DataArrays,
        each with (pixel,) dimensions.
    """
    crs = raw_dataset.rio.crs
    if crs is None:
        raise ValueError("No CRS found. Ensure decode_coords='all' in xr.open_dataset.")

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    x_dim = raw_dataset.rio.x_dim
    y_dim = raw_dataset.rio.y_dim
    x_values = raw_dataset[x_dim].values
    y_values = raw_dataset[y_dim].values

    xx, yy = np.meshgrid(x_values, y_values)
    lons, lats = transformer.transform(xx, yy)

    n_pixel = lats.size
    pixel_coords = np.arange(n_pixel)

    lat_da = xr.DataArray(
        lats.ravel(),
        coords={"pixel": pixel_coords},
        dims=("pixel",),
        name="latitude",
    )
    lon_da = xr.DataArray(
        lons.ravel(),
        coords={"pixel": pixel_coords},
        dims=("pixel",),
        name="longitude",
    )

    return lat_da, lon_da
