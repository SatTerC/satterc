from pathlib import Path
from os import PathLike

import xarray as xr


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
