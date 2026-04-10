from os import PathLike
from pathlib import Path
from typing import Any

from hamilton.data_quality.base import DataValidator, ValidationResult
import pandas as pd
import xarray as xr


def load_dataset(path: str | PathLike) -> xr.Dataset:
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


def stack_spatial_dims(ds: xr.Dataset) -> xr.Dataset:
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


class DatetimeIndexValidator(DataValidator):
    def __init__(self, freq: str) -> None:
        super().__init__(importance="fail")

        if freq not in ("D", "W", "ME"):
            raise ValueError("`freq` must be one of 'D', 'W', or 'ME'")

        self.freq = freq

    def applies_to(self, datatype) -> bool:
        return issubclass(datatype, pd.Index)

    def description(self) -> str:
        return "Ensures the supplied index is a pandas DatetimeIndex with the expected frequency."

    @classmethod
    def name(cls) -> str:
        return "datetimeindex_validator"

    def validate(self, dataset: Any) -> ValidationResult:
        index = dataset

        if not isinstance(index, pd.DatetimeIndex):
            return ValidationResult(
                passes=False, message=f"Expected pd.DatetimeIndex, got {type(index)}"
            )

        inferred_freq = index.freqstr or pd.infer_freq(index)

        if inferred_freq is None:
            return ValidationResult(
                passes=False,
                message="Could not determine frequency from index",
            )

        if self.freq == "W":
            weekly_prefixes = ("W", "7D")
            passes = any(inferred_freq.startswith(p) for p in weekly_prefixes)
        elif self.freq == "ME":
            monthly_prefixes = ("ME", "MS")
            passes = any(inferred_freq.startswith(p) for p in monthly_prefixes)
        else:
            passes = inferred_freq == self.freq

        if not passes:
            return ValidationResult(
                passes=False,
                message=f"Expected frequency {self.freq}, got {inferred_freq}",
            )

        return ValidationResult(passes=True, message="")
