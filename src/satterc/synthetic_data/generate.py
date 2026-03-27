"""Generate synthetic input data using Hamilton DAG."""

from pathlib import Path
from typing import Any

import numpy as np
from hamilton import driver
from hamilton.settings import ENABLE_POWER_USER_MODE

from . import daily, static, merged
from .. import pipeline


def _set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    np.random.seed(seed)


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
    - pipeline.outputs modules for merging and saving
    - pipeline.resample for temporal resampling

    The config's input paths are mapped to output paths, since we're
    generating the input data files.
    """
    _set_random_seed(seed)

    n_lat, n_lon = grid
    daily_vars = set(config["driver_config"].get("daily_inputs_vars", []))
    weekly_vars = set(config["driver_config"].get("weekly_inputs_vars", []))
    monthly_vars = set(config["driver_config"].get("monthly_inputs_vars", []))

    daily_to_weekly = list(weekly_vars)
    daily_to_monthly = list(daily_vars | weekly_vars | monthly_vars)
    weekly_to_monthly: list[str] = []

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
        "weekly_outputs_vars": list(weekly_vars),
        "monthly_outputs_path": config["driver_config"].get("monthly_inputs_path"),
        "monthly_outputs_vars": list(monthly_vars),
        "static_outputs_path": config["driver_config"].get("static_inputs_path"),
        "static_outputs_vars": config["driver_config"].get("static_inputs_vars", []),
        "daily_to_weekly": daily_to_weekly,
        "daily_to_monthly": daily_to_monthly,
        "weekly_to_monthly": weekly_to_monthly,
    }

    modules = [
        daily,
        static,
        merged,
        pipeline.resample,
    ]

    dr = driver.Builder().with_modules(*modules).with_config(driver_config).build()

    targets = [
        "save_daily_outputs",
        "save_weekly_outputs",
        "save_monthly_outputs",
        "save_static_outputs",
    ]

    dr.execute(targets)
