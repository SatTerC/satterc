from hamilton.function_modifiers import unpack_fields
import numpy as np
import xarray as xr
import rioxarray as rioxarray  # only used indirectly via dataset.rio; suppress linter errors
from pyproj import Transformer

from ._utils import stack_spatial_dims


class MisalignedGridError(Exception):
    pass


def _check_common_grid(ds1: xr.Dataset, ds2: xr.Dataset, atol: float = 1e-6) -> None:
    """Check that two Datasets share a common grid.

    Raises
    ------
    MisalignedGridError:
        If the grids are misaligned.
    """
    # Check CRS metadata
    if not (ds1.rio.crs == ds2.rio.crs):
        raise MisalignedGridError(f"Mismatched CRS! {ds1.rio.crs} ≠ {ds2.rio.crs}")

    # Attempt to access dim names
    try:
        x1, y1 = ds1.rio.x_dim, ds1.rio.y_dim
        x2, y2 = ds2.rio.x_dim, ds2.rio.y_dim
    except AttributeError as e:
        raise MisalignedGridError("Could not access (x, y) dims") from e

    # Check dim names agree
    if not (x1 == x2 and y1 == y2):
        raise MisalignedGridError(
            f"Mismatched dimension names: ({x1}, {y1}) ≠ ({x2}, {y2})"
        )

    # Check coord values
    try:
        np.testing.assert_allclose(ds1[x1].values, ds2[x2].values, atol=atol)
        np.testing.assert_allclose(ds1[y1].values, ds2[y2].values, atol=atol)
    except AssertionError as e:
        raise MisalignedGridError("Mismatched coordinate values!") from e


def common_grid(
    daily_inputs: xr.Dataset,
    weekly_inputs: xr.Dataset,
    monthly_inputs: xr.Dataset,
    static_inputs: xr.Dataset,
) -> xr.Dataset:
    _check_common_grid(daily_inputs, static_inputs)
    _check_common_grid(weekly_inputs, static_inputs)
    _check_common_grid(monthly_inputs, static_inputs)

    # Since all grids agree, just take static_inputs as the reference Dataset
    ds = static_inputs

    # Extract coordinates
    x_dim, y_dim = ds.rio.x_dim, ds.rio.y_dim
    x = ds[x_dim].values
    y = ds[y_dim].values

    # Create meshgrid for transformation
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")

    # Transform to lat/lon
    transformer = Transformer.from_crs(ds.rio.crs, "EPSG:4326", always_xy=True)
    lon_grid, lat_grid = transformer.transform(x_grid, y_grid)

    return xr.Dataset(
        data_vars={
            "latitude": (["x", "y"], lat_grid),
            "longitude": (["x", "y"], lon_grid),
        },
        coords={"x": x, "y": y},
        attrs={"crs": ds.rio.crs},
    )


def common_grid_stacked(common_grid: xr.Dataset) -> xr.Dataset:
    return stack_spatial_dims(common_grid)


@unpack_fields("latitude", "longitude")
def unpack_common_grid(
    common_grid_stacked: xr.Dataset,
) -> tuple[xr.DataArray, xr.DataArray]:
    # NOTE: not sure if there's any point in even including x, y as nodes.
    return (
        common_grid_stacked.latitude,
        common_grid_stacked.longitude,
    )
