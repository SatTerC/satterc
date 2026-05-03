from pathlib import Path
from os import PathLike

import pandas as pd
import xarray as xr


def unstack_if_grid(ds: xr.Dataset) -> xr.Dataset:
    """Unstack pixel → (y, x) if pixel is a (y, x) MultiIndex; pass through otherwise.

    Three cases:
    - No 'pixel' dim (single-point): return unchanged.
    - 'pixel' is a plain integer index (multi-point): return unchanged.
    - 'pixel' is a (y, x) MultiIndex (2D grid after stacking): unstack to grid.
    """
    if "pixel" in ds.dims and isinstance(ds.indexes.get("pixel"), pd.MultiIndex):
        return ds.unstack("pixel")
    return ds


def _save_dataset(ds: xr.Dataset, path: str | PathLike) -> None:
    """Saves a dataset to a NetCDF file or Zarr store based on the file extension.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to save.
    path : str | PathLike
        The destination path.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in [".nc", ".netcdf"]:
        ds.to_netcdf(path, engine="netcdf4")

    elif suffix == ".zarr" or (not suffix and p.is_dir()):
        ds.to_zarr(path)

    else:
        raise ValueError(
            f"Unsupported file extension: '{suffix}'. Use '.nc', '.netcdf', or '.zarr'."
        )


def dataset_to_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """Convert an output Dataset to a pandas DataFrame.

    Squeezes the size-1 'pixel' dimension if present, then calls to_dataframe().
    Any JAX-backed arrays are materialised to numpy in that call.
    """
    if "pixel" in ds.dims:
        ds = ds.squeeze("pixel", drop=True)
    return ds.to_dataframe()


def save_timeseries(df: pd.DataFrame, path: str | PathLike) -> None:
    """Save a DataFrame to CSV or Parquet, auto-detected by extension."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        df.to_csv(path)
    elif suffix in (".parquet", ".pq"):
        df.to_parquet(path)
    else:
        raise ValueError(
            f"Unsupported format: '{suffix}'. Use '.csv', '.parquet', or '.pq'."
        )
