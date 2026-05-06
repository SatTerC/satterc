from os import PathLike
from pathlib import Path
from typing import Any

from hamilton.data_quality.base import DataValidator, ValidationResult
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rioxarray  # noqa: F401 — registers the .rio accessor

from .._utils import stack_spatial_dims


def load_timeseries(path: str | PathLike) -> xr.Dataset:
    """Load a single-point time series from CSV or Parquet, detected by extension.

    Returns a Dataset with dims (time, pixel) where pixel has a single coordinate
    value 0, matching the shape expected by the rest of the pipeline.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported format: '{suffix}'. Use '.csv', '.parquet', or '.pq'."
        )

    if "time" in df.columns:
        df = df.set_index("time")
    if df.index.name != "time":
        df.index.name = "time"
    df.index = pd.to_datetime(df.index)

    ds = df.to_xarray()
    ds = ds.expand_dims({"pixel": [0]})
    return ds.transpose("time", "pixel")


def load_static(path: str | PathLike) -> xr.Dataset:
    """Load single-point static inputs from JSON, YAML, or TOML, detected by extension.

    Returns a Dataset with dim (pixel,) where pixel has a single coordinate value 0.
    """
    import json
    import tomllib

    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".json":
        with open(p) as f:
            data: dict = json.load(f)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required for YAML static inputs. "
                "Install it with: uv add pyyaml"
            ) from e
        with open(p) as f:
            data = yaml.safe_load(f)
    elif suffix == ".toml":
        with open(p, "rb") as f:
            data = tomllib.load(f)
    else:
        raise ValueError(
            f"Unsupported format: '{suffix}'. Use '.json', '.yaml', '.yml', or '.toml'."
        )

    return xr.Dataset(
        {
            k: xr.DataArray(np.asarray([v], dtype=float), dims=["pixel"])
            for k, v in data.items()
        },
        coords={"pixel": [0]},
    )


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


def stack_if_spatial(ds: xr.Dataset) -> xr.Dataset:
    """Stack spatial dims if the dataset is a CRS-bearing 2D grid; pass through otherwise.

    Three cases:
    - Already has a 'pixel' dim (pre-stacked multi-point): return unchanged.
    - Has a CRS (2D grid): stack (y, x) → pixel via stack_spatial_dims.
    - No CRS, no pixel (single-point or non-spatial): return unchanged.
    """
    if "pixel" in ds.dims:
        return ds
    if ds.rio.crs is not None:
        return stack_spatial_dims(ds)
    return ds


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
