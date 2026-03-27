"""Generate synthetic input data using Hamilton DAG."""

from typing import Any

import numpy as np
import rioxarray  # noqa: F401 - needed to enable .rio accessor
import xarray as xr
from hamilton import driver
from hamilton.settings import ENABLE_POWER_USER_MODE

from . import daily, static
from ..pipeline import outputs, resample


def _set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def _add_crs_to_netcdf(path: str) -> None:
    """Add CRS metadata to a netcdf file using rioxarray."""
    with xr.open_dataset(path, decode_coords="all") as ds:
        ds.rio.write_crs("EPSG:4326", inplace=True)
        # ds.to_netcdf(path)


def generate_synthetic_data(
    config: dict[str, Any],
    grid: tuple[int, int],
    n_days: int,
    seed: int = 42,
) -> None:
    """Generate synthetic input data using Hamilton DAG.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dict from load_config(). Should contain:
        - inputs: dict with daily, weekly, monthly, static sections
          each having 'path' and 'vars' keys.
        - resample: optional dict with daily_to_weekly, daily_to_monthly, etc.
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
    netCDF files using rioxarray.
    """
    _set_random_seed(seed)

    n_lat, n_lon = grid
    daily_vars = set(config["driver_config"].get("daily_inputs_vars", []))
    weekly_vars = set(config["driver_config"].get("weekly_inputs_vars", []))
    monthly_vars = set(config["driver_config"].get("monthly_inputs_vars", []))

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
        "daily_outputs_path": config["driver_config"].get("daily_inputs_path"),
        "daily_outputs_vars": list(daily_vars),
        "weekly_outputs_path": config["driver_config"].get("weekly_inputs_path"),
        "weekly_outputs_vars": weekly_outputs_vars,
        "monthly_outputs_path": config["driver_config"].get("monthly_inputs_path"),
        "monthly_outputs_vars": monthly_outputs_vars,
        "static_outputs_path": config["driver_config"].get("static_inputs_path"),
        "static_outputs_vars": config["driver_config"].get("static_inputs_vars", []),
        "daily_to_weekly": daily_to_weekly,
        "daily_to_monthly": daily_to_monthly,
        "weekly_to_monthly": weekly_to_monthly,
    }

    modules = [
        daily,
        static,
        resample,
        outputs.daily,
        outputs.weekly,
        outputs.monthly,
        outputs.static,
    ]

    dr = (
        driver.Builder()
        .with_modules(*modules)
        .with_config(driver_config)
        .allow_module_overrides()
        .build()
    )

    targets = [
        "save_daily_outputs",
        "save_weekly_outputs",
        "save_monthly_outputs",
        "save_static_outputs",
    ]

    dr.execute(targets)

    # Add CRS data to the netcdf files. Easier to do it as an extra step
    # rather than modifying / reimplementing existing outputs nodes.
    _add_crs_to_netcdf(driver_config["daily_outputs_path"])
    _add_crs_to_netcdf(driver_config["weekly_outputs_path"])
    _add_crs_to_netcdf(driver_config["monthly_outputs_path"])
    _add_crs_to_netcdf(driver_config["static_outputs_path"])
