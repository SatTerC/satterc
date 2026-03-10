from pathlib import Path
from os import PathLike

from hamilton.function_modifiers import unpack_fields
import xarray as xr


def _load_dataset(path: str | PathLike) -> xr.Dataset:
    """
    Loads a dataset from a netcdf file or zarr store.

    Parameters
    ----------
    path : str
        Path to the NetCDF or Zarr dataset.

    Returns
    -------
    xr.Dataset
        The loaded dataset with coordinates decoded.

    Notes
    -----
    Uses `decode_coords="all"` so that rioxarray can find the CRS
    in NetCDF/Zarr metadata.
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


def _check_datasets_aligned(
    daily_ds: xr.Dataset, weekly_ds: xr.Dataset, monthly_ds: xr.Dataset
) -> None:
    # TODO: improve this check, possibly using xindexes.
    # Do we need to know what the x,y dims are called??
    xr.testing.assert_allclose(daily_ds.coords[1], weekly_ds.coords[1])
    xr.testing.assert_allclose(daily_ds.coords[2], weekly_ds.coords[2])
    xr.testing.assert_allclose(daily_ds.coords[1], monthly_ds.coords[1])
    xr.testing.assert_allclose(daily_ds.coords[2], monthly_ds.coords[2])


def _stack_spatial_dims(ds: xr.Dataset) -> xr.Dataset:
    """Stack spatial dimensions into a single pixel dimension.

    Parameters:
    -----------
    ds : xr.Dataset
        Input Dataset with spatial dimensions at positions 1 and 2 in the
        dimensions list (with 'time' being the first dimension).

    Returns:
    --------
    xr.Dataset
        Dataset with two spatial dimensions stacked into a single 'pixel' dimension.
    """
    # Use rioxarray to identify spatial dimensions (y, x)
    spatial_dims = ds.rio.y_dim, ds.rio.x_dim

    # Stack (y, x) -> pixel
    ds_stacked = ds.stack(pixel=spatial_dims)

    # 4. "Clean up" the MultiIndex
    # This turns 'pixel' into a simple 0, 1, 2... index and
    # transforms y and x into standard 1D arrays aligned with 'pixel'
    # ds_lexi = ds_stacked.reset_index("pixel", drop=True)
    # NOTE: This may be a good idea, but need to change how the lat-lon grid is computed if so.

    return ds_stacked


@unpack_fields(
    "daily_inputs_dataset",
    "weekly_inputs_dataset",
    "monthly_inputs_dataset",
    "static_inputs_dataset",
)
def input_datasets(
    inputs_dir: str | PathLike,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
    inputs_path = Path(inputs_dir)

    daily_ds = _load_dataset(inputs_path / "daily.nc")
    weekly_ds = _load_dataset(inputs_path / "weekly.nc")
    monthly_ds = _load_dataset(inputs_path / "monthly.nc")
    static_ds = _load_dataset(inputs_path / "static.nc")

    _check_datasets_aligned(daily_ds, weekly_ds, monthly_ds)

    daily_ds_stacked = _stack_spatial_dims(daily_ds)
    weekly_ds_stacked = _stack_spatial_dims(weekly_ds)
    monthly_ds_stacked = _stack_spatial_dims(monthly_ds)
    static_ds_stacked = _stack_spatial_dims(static_ds)

    return daily_ds_stacked, weekly_ds_stacked, monthly_ds_stacked, static_ds_stacked
