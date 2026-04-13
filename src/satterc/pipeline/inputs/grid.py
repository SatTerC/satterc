from hamilton.function_modifiers import unpack_fields
import numpy as np
import xarray as xr
import rioxarray as rioxarray  # only used indirectly via dataset.rio; suppress linter errors
from pyproj import Transformer

from ._utils import stack_spatial_dims


class MisalignedGridError(Exception):
    pass


def _check_common_grid(
    ds1: xr.Dataset,
    ds2: xr.Dataset,
    label1: str = "ds1",
    label2: str = "ds2",
    atol: float = 1e-6,
) -> None:
    """Check that two Datasets share a common grid.

    Raises
    ------
    MisalignedGridError:
        If the grids are misaligned.
    """
    # Check CRS metadata
    if not (ds1.rio.crs == ds2.rio.crs):
        raise MisalignedGridError(
            f"Mismatched CRS! {label1}={ds1.rio.crs} ≠ {label2}={ds2.rio.crs}"
        )

    # Attempt to access dim names
    try:
        x1, y1 = ds1.rio.x_dim, ds1.rio.y_dim
        x2, y2 = ds2.rio.x_dim, ds2.rio.y_dim
    except AttributeError as e:
        raise MisalignedGridError(
            f"Could not access (x, y) dims for {label1} or {label2}"
        ) from e

    # Check dim names agree
    if not (x1 == x2 and y1 == y2):
        raise MisalignedGridError(
            f"Mismatched dimension names: {label1}=({x1}, {y1}) ≠ {label2}=({x2}, {y2})"
        )

    # Check coord values
    try:
        np.testing.assert_allclose(ds1[x1].values, ds2[x2].values, atol=atol)
        np.testing.assert_allclose(ds1[y1].values, ds2[y2].values, atol=atol)
    except AssertionError as e:
        raise MisalignedGridError(
            f"Mismatched coordinate values between {label1} and {label2}!"
        ) from e


def common_grid(
    loaded_daily_inputs: xr.Dataset | None = None,
    loaded_weekly_inputs: xr.Dataset | None = None,
    loaded_monthly_inputs: xr.Dataset | None = None,
    loaded_static_inputs: xr.Dataset | None = None,
) -> xr.Dataset:
    present = [
        (name, ds)
        for name, ds in [
            ("daily", loaded_daily_inputs),
            ("weekly", loaded_weekly_inputs),
            ("monthly", loaded_monthly_inputs),
            ("static", loaded_static_inputs),
        ]
        if ds is not None
    ]
    if not present:
        raise ValueError("At least one input dataset must be provided to common_grid")

    ref_name, ref_ds = present[0]
    for name, ds in present[1:]:
        _check_common_grid(ref_ds, ds, label1=ref_name, label2=name)

    # All inputs share a common grid; use the first available as spatial reference
    x_dim, y_dim = ref_ds.rio.x_dim, ref_ds.rio.y_dim
    x = ref_ds[x_dim].values
    y = ref_ds[y_dim].values

    # Create meshgrid for transformation
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")

    # Transform to lat/lon
    transformer = Transformer.from_crs(ref_ds.rio.crs, "EPSG:4326", always_xy=True)
    lon_grid, lat_grid = transformer.transform(x_grid, y_grid)

    return xr.Dataset(
        data_vars={
            "latitude": (["x", "y"], lat_grid),
            "longitude": (["x", "y"], lon_grid),
        },
        coords={"x": x, "y": y},
        attrs={"crs": ref_ds.rio.crs},
    )


def stacked_grid(common_grid: xr.Dataset) -> xr.Dataset:
    return stack_spatial_dims(common_grid)


@unpack_fields("latitude", "longitude")
def split_grid(
    stacked_grid: xr.Dataset,
) -> tuple[xr.DataArray, xr.DataArray]:
    # NOTE: not sure if there's any point in even including x, y as nodes.
    return (
        stacked_grid.latitude,
        stacked_grid.longitude,
    )
