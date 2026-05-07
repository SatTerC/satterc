"""I/O functions for loading inputs and saving outputs outside the Hamilton DAG."""

from os import PathLike
from pathlib import Path
from typing import Any, cast

from .config import IOSpec

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rioxarray  # noqa: F401 — registers the .rio accessor
from pyproj import Transformer

from .pipeline._utils import stack_spatial_dims


class MisalignedGridError(Exception):
    pass


# ---------------------------------------------------------------------------
# Internal helpers: opening datasets
# ---------------------------------------------------------------------------


def load_dataset(path: str | PathLike) -> xr.Dataset:
    """Open a NetCDF or Zarr dataset with coordinates decoded."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".nc", ".netcdf"):
        engine = "netcdf4"
    elif suffix == ".zarr":
        engine = "zarr"
    else:
        raise ValueError(f"Unsupported file extension: {p.suffix}.")
    return xr.open_dataset(path, engine=engine, decode_coords="all")


def load_timeseries(path: str | PathLike) -> xr.Dataset:
    """Load a single-point time series from CSV or Parquet.

    Returns a Dataset with dims (time, pixel) where pixel has coordinate value 0.
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
    """Load single-point static inputs from JSON, YAML, or TOML.

    Returns a Dataset with dim (pixel,) where pixel has coordinate value 0.
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


def _load_raw(path: str) -> xr.Dataset:
    """Dispatch to the right loader based on file extension."""
    suffix = Path(path).suffix.lower()
    if suffix in (".nc", ".netcdf", ".zarr"):
        return load_dataset(path)
    if suffix in (".json", ".yaml", ".yml", ".toml"):
        return load_static(path)
    return load_timeseries(path)  # raises ValueError for unsupported extensions


# ---------------------------------------------------------------------------
# Internal helpers: stacking / unstacking
# ---------------------------------------------------------------------------


def stack_if_gridded(ds: xr.Dataset) -> xr.Dataset:
    """Stack (y, x) → pixel if the dataset is a CRS-bearing 2D grid; pass through otherwise."""
    if "pixel" in ds.dims:
        return ds
    if ds.rio.crs is not None:
        return stack_spatial_dims(ds)
    return ds


def unstack_if_gridded(ds: xr.Dataset) -> xr.Dataset:
    """Unstack pixel → (y, x) if pixel is a (y, x) MultiIndex; pass through otherwise."""
    if "pixel" in ds.dims and isinstance(ds.indexes.get("pixel"), pd.MultiIndex):
        return ds.unstack("pixel")
    return ds


# ---------------------------------------------------------------------------
# Internal helpers: datetime validation
# ---------------------------------------------------------------------------

_FREQ_CODES: dict[str, str] = {"daily": "D", "weekly": "W", "monthly": "ME"}


def _validate_dates(ds: xr.Dataset, freq: str) -> pd.DatetimeIndex:
    """Extract and validate the time index from a dataset."""
    idx = cast(pd.DatetimeIndex, ds.get_index("time"))
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError(
            f"Expected a DatetimeIndex for '{freq}' inputs, got {type(idx)}"
        )

    expected = _FREQ_CODES[freq]
    inferred = pd.infer_freq(idx)

    if inferred is None:
        raise ValueError(f"Could not determine frequency from '{freq}' time index")

    if expected == "W":
        passes = any(inferred.startswith(p) for p in ("W", "7D"))
    elif expected == "ME":
        passes = any(inferred.startswith(p) for p in ("ME", "MS"))
    else:
        passes = inferred == expected

    if not passes:
        raise ValueError(
            f"Expected '{freq}' time index with frequency '{expected}', got '{inferred}'"
        )

    return idx


# ---------------------------------------------------------------------------
# Internal helpers: grid computation
# ---------------------------------------------------------------------------


def _check_common_grid(
    ds1: xr.Dataset,
    ds2: xr.Dataset,
    label1: str = "ds1",
    label2: str = "ds2",
    atol: float = 1e-6,
) -> None:
    """Raise MisalignedGridError if two datasets do not share a common CRS and coordinates."""
    if ds1.rio.crs != ds2.rio.crs:
        raise MisalignedGridError(
            f"Mismatched CRS! {label1}={ds1.rio.crs} ≠ {label2}={ds2.rio.crs}"
        )
    x1, y1 = ds1.rio.x_dim, ds1.rio.y_dim
    x2, y2 = ds2.rio.x_dim, ds2.rio.y_dim
    if not (x1 == x2 and y1 == y2):
        raise MisalignedGridError(
            f"Mismatched dimension names: {label1}=({x1}, {y1}) ≠ {label2}=({x2}, {y2})"
        )
    try:
        np.testing.assert_allclose(ds1[x1].values, ds2[x2].values, atol=atol)
        np.testing.assert_allclose(ds1[y1].values, ds2[y2].values, atol=atol)
    except AssertionError as e:
        raise MisalignedGridError(
            f"Mismatched coordinate values between {label1} and {label2}!"
        ) from e


def _compute_lat_lon(
    spatial_datasets: dict[str, xr.Dataset],
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute stacked latitude and longitude DataArrays from CRS-bearing datasets."""
    items = list(spatial_datasets.items())
    ref_name, ref_ds = items[0]
    for name, ds in items[1:]:
        _check_common_grid(ref_ds, ds, label1=ref_name, label2=name)

    x_dim, y_dim = ref_ds.rio.x_dim, ref_ds.rio.y_dim
    x = ref_ds[x_dim].values
    y = ref_ds[y_dim].values

    # indexing="ij": x varies along axis 0, y along axis 1 — matches (x, y) DataArray dims
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    transformer = Transformer.from_crs(ref_ds.rio.crs, "EPSG:4326", always_xy=True)
    lon_grid, lat_grid = transformer.transform(x_grid, y_grid)

    grid_ds = xr.Dataset(
        data_vars={
            "latitude": (["x", "y"], lat_grid),
            "longitude": (["x", "y"], lon_grid),
        },
        coords={"x": x, "y": y},
    )
    stacked = stack_spatial_dims(grid_ds)
    return stacked.latitude, stacked.longitude


# ---------------------------------------------------------------------------
# Internal helpers: saving datasets
# ---------------------------------------------------------------------------


def dataset_to_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """Convert an output Dataset to a DataFrame, squeezing the size-1 pixel dim if present."""
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


def _save_netcdf(ds: xr.Dataset, path: str | PathLike) -> None:
    """Save a dataset to NetCDF or Zarr based on extension."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".nc", ".netcdf"):
        ds.to_netcdf(path, engine="netcdf4")
    elif suffix == ".zarr" or (not suffix and p.is_dir()):
        ds.to_zarr(path)
    else:
        raise ValueError(
            f"Unsupported file extension: '{suffix}'. Use '.nc', '.netcdf', or '.zarr'."
        )


def _save(ds: xr.Dataset, path: str) -> None:
    suffix = Path(path).suffix.lower()
    if suffix in (".nc", ".netcdf", ".zarr"):
        _save_netcdf(ds, path)
    else:
        save_timeseries(dataset_to_dataframe(ds), path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_inputs(input_specs: dict[str, IOSpec]) -> dict[str, Any]:
    """Load all configured inputs and return them as a flat dict of named DataArrays.

    Keys follow Hamilton naming conventions:
    - Temporal variables: ``{var}_{freq}`` (e.g. ``temperature_celcius_daily``)
    - Static variables: ``{var}`` (no suffix)
    - Time indices: ``dates_{freq}``
    - Grid: ``latitude``, ``longitude`` (only when CRS-bearing inputs are present)

    Parameters
    ----------
    input_specs:
        Mapping from frequency string to ``IOSpec`` (path, vars, format).
        Typically ``parsed_config.input_specs``.
    """
    inputs: dict[str, Any] = {}
    raw_datasets: dict[str, xr.Dataset] = {}

    for freq, spec in input_specs.items():
        ds_raw = _load_raw(spec.path)
        raw_datasets[freq] = ds_raw

        ds = stack_if_gridded(ds_raw)
        suffix = "" if freq == "static" else f"_{freq}"
        vars_to_load = list(ds.data_vars) if spec.vars is None else spec.vars
        for var in vars_to_load:
            inputs[f"{var}{suffix}"] = ds[var]

        if freq != "static":
            inputs[f"dates_{freq}"] = _validate_dates(ds_raw, freq)

    spatial = {f: ds for f, ds in raw_datasets.items() if ds.rio.crs is not None}
    if spatial:
        lat, lon = _compute_lat_lon(spatial)
        inputs["latitude"] = lat
        inputs["longitude"] = lon

    return inputs


def get_outputs(
    results: dict[str, xr.DataArray],
    output_specs: dict[str, IOSpec],
) -> dict[str, xr.Dataset]:
    """Merge and unstack model results into per-frequency Datasets.

    Parameters
    ----------
    results:
        Dict returned by ``driver.execute()``, keyed by Hamilton node name.
    output_specs:
        Mapping from frequency string to ``IOSpec``.
        Typically ``parsed_config.output_specs``.
    """
    out: dict[str, xr.Dataset] = {}
    for freq, spec in output_specs.items():
        suffix = "" if freq == "static" else f"_{freq}"
        # (Re-)assign names to all arrays to ensure merging succeeds
        arrays = [results[f"{var}{suffix}"].rename(var) for var in spec.vars]
        out[freq] = unstack_if_gridded(xr.merge(arrays))
    return out


def save_outputs(
    output_datasets: dict[str, xr.Dataset],
    output_specs: dict[str, IOSpec],
) -> None:
    """Write per-frequency Datasets to disk.

    Parameters
    ----------
    output_datasets:
        Dict returned by ``get_outputs()``.
    output_specs:
        Mapping from frequency string to ``IOSpec``.
        Typically ``parsed_config.output_specs``.
    """
    for freq, ds in output_datasets.items():
        _save(ds, output_specs[freq].path)
