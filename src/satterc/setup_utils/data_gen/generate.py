"""Generate synthetic input data using Hamilton DAG."""

import inspect
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from hamilton import driver
from hamilton.settings import ENABLE_POWER_USER_MODE

from . import daily, static
from .fallback import build_fallback_module
from ...pipeline import outputs, resample
from ...pipeline.outputs._utils import dataset_to_dataframe, save_timeseries
from ...config import ParsedConfig

_FLAT_SUFFIXES = {".csv", ".parquet", ".pq"}


def _set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def _save_dataset_with_crs(ds: xr.Dataset, path: str | PathLike) -> None:
    """Save dataset to NetCDF, Zarr, CSV, or Parquet.

    CSV and Parquet are written as flat time-indexed tables (CRS not stored).
    NetCDF and Zarr receive a crs='EPSG:4326' global attribute.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to save.
    path : str | PathLike
        The destination path. Format is inferred from the file extension.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in _FLAT_SUFFIXES:
        save_timeseries(dataset_to_dataframe(ds), path)
        return

    ds.attrs["crs"] = "EPSG:4326"

    if suffix in (".nc", ".netcdf"):
        ds.to_netcdf(path, engine="netcdf4")
    elif suffix == ".zarr" or (not suffix and p.is_dir()):
        ds.to_zarr(path)
    else:
        raise ValueError(
            f"Unsupported file extension: '{suffix}'. "
            "Use '.nc', '.netcdf', '.zarr', '.csv', or '.parquet'."
        )


def _known_daily_fns() -> set[str]:
    """Names of daily generator functions available in the daily module."""
    return {
        name
        for name, obj in inspect.getmembers(daily, inspect.isfunction)
        if not name.startswith("_")
    }


def _known_static_fns() -> set[str]:
    """Names of static generator functions available in the static module."""
    return {
        name
        for name, obj in inspect.getmembers(static, inspect.isfunction)
        if not name.startswith("_")
    }


def generate_synthetic_data(
    config: ParsedConfig,
    grid: tuple[int, int],
    n_days: int,
    seed: int = 42,
) -> None:
    """Generate synthetic input data using Hamilton DAG.

    Parameters
    ----------
    config : ParsedConfig
        Parsed configuration from load_config().
    grid : tuple[int, int]
        Grid dimensions as (n_lat, n_lon).
    n_days : int
        Number of days to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    None

    Notes
    -----
    This function builds a Hamilton DAG with:
    - synthetic_data modules for generating variables
    - pipeline.outputs modules for merging and unstacking temporal data
    - pipeline.resample for temporal resampling

    The config's input paths are mapped to output paths, since we're
    generating the input data files.

    After the DAG runs, CRS metadata (EPSG:4326) is added to all output
    netCDF files. Variables not found in the built-in generators fall back
    to Gaussian noise with a logged warning.
    """
    _set_random_seed(seed)

    n_lat, n_lon = grid
    daily_vars = set(config.driver_config.get("daily_inputs_vars", []))
    weekly_vars = set(config.driver_config.get("weekly_inputs_vars", []))
    monthly_vars = set(config.driver_config.get("monthly_inputs_vars", []))
    static_vars = list(config.driver_config.get("static_inputs_vars", []))

    daily_to_weekly = list(weekly_vars)
    daily_to_monthly = list(daily_vars | weekly_vars | monthly_vars)
    weekly_to_monthly: list[str] = []

    weekly_outputs_vars = list(weekly_vars | set(daily_to_weekly))
    monthly_outputs_vars = list(set(daily_to_monthly) | monthly_vars)

    driver_config: dict[str, Any] = {
        ENABLE_POWER_USER_MODE: True,
        "n_lat": n_lat,
        "n_lon": n_lon,
        "n_days": n_days,
        "start_date": "2020-01-01",
        "seed": seed,
        "daily_outputs_path": config.driver_config.get("daily_inputs_path"),
        "daily_outputs_vars": list(daily_vars),
        "weekly_outputs_path": config.driver_config.get("weekly_inputs_path"),
        "weekly_outputs_vars": weekly_outputs_vars,
        "monthly_outputs_path": config.driver_config.get("monthly_inputs_path"),
        "monthly_outputs_vars": monthly_outputs_vars,
        "static_outputs_path": config.driver_config.get("static_inputs_path"),
        "static_outputs_vars": static_vars,
        "daily_to_weekly": daily_to_weekly,
        "daily_to_monthly": daily_to_monthly,
        "weekly_to_monthly": weekly_to_monthly,
    }

    # Detect variables that have no explicit generator and inject fallbacks.
    all_temporal_vars = daily_vars | weekly_vars | monthly_vars
    known_daily = _known_daily_fns()
    known_static = _known_static_fns()
    unknown_daily = [v for v in all_temporal_vars if f"{v}_daily" not in known_daily]
    unknown_static = [v for v in static_vars if v not in known_static]

    modules = [daily, static, resample]

    if unknown_daily or unknown_static:
        modules.append(build_fallback_module(unknown_daily, unknown_static))

    targets = []

    if daily_vars:
        modules.append(outputs.daily)
        targets.append("unstacked_daily_outputs")
    if weekly_outputs_vars:
        modules.append(outputs.weekly)
        targets.append("unstacked_weekly_outputs")
    if monthly_outputs_vars:
        modules.append(outputs.monthly)
        targets.append("unstacked_monthly_outputs")
    modules.append(outputs.static)
    targets.append("unstacked_static_outputs")

    dr = (
        driver.Builder()
        .with_modules(*modules)
        .with_config(driver_config)
        .allow_module_overrides()
        .build()
    )

    results = dr.execute(targets)

    if daily_vars and driver_config["daily_outputs_path"]:
        _save_dataset_with_crs(
            results["unstacked_daily_outputs"], driver_config["daily_outputs_path"]
        )
    if weekly_outputs_vars and driver_config["weekly_outputs_path"]:
        _save_dataset_with_crs(
            results["unstacked_weekly_outputs"], driver_config["weekly_outputs_path"]
        )
    if monthly_outputs_vars and driver_config["monthly_outputs_path"]:
        _save_dataset_with_crs(
            results["unstacked_monthly_outputs"], driver_config["monthly_outputs_path"]
        )
    _save_dataset_with_crs(
        results["unstacked_static_outputs"], driver_config["static_outputs_path"]
    )
