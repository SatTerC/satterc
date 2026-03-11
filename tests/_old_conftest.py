"""
Shared pytest fixtures for satterc integration tests.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from satterc import driver


@pytest.fixture
def synthetic_inputs_defaults():
    """Default configuration for synthetic data generation."""
    return dict(
        n_days=731,
        n_lat=2,
        n_lon=2,
        lat_min=50.0,
        lat_max=54.0,
        lon_min=-4.0,
        lon_max=2.0,
        random_seed=42,
        start_date="2020-01-01",
    )


@pytest.fixture
def synthetic_driver(synthetic_inputs_defaults):
    """Build driver with default synthetic inputs config."""
    return driver.build_driver(
        modules=["synthetic_inputs", "resample"],
        config=synthetic_inputs_defaults.copy(),
    )


@pytest.fixture
def synthetic_driver_3x3(synthetic_inputs_defaults):
    """Build driver with 3x3 grid."""
    config = synthetic_inputs_defaults.copy()
    config["n_lat"] = 3
    config["n_lon"] = 3
    return driver.build_driver(
        modules=["synthetic_inputs", "resample"],
        config=config,
    )


@pytest.fixture
def synthetic_driver_4x4(synthetic_inputs_defaults):
    """Build driver with 4x4 grid."""
    config = synthetic_inputs_defaults.copy()
    config["n_lat"] = 4
    config["n_lon"] = 4
    return driver.build_driver(
        modules=["synthetic_inputs", "resample"],
        config=config,
    )


@pytest.fixture
def synthetic_driver_splash(synthetic_inputs_defaults):
    """Build driver with synthetic inputs, resample, and splash modules."""
    return driver.build_driver(
        modules=["synthetic_inputs", "resample", "splash"],
        config=synthetic_inputs_defaults.copy(),
    )


@pytest.fixture
def synthetic_driver_splash_1x1(synthetic_inputs_defaults):
    """Build driver with 1x1 grid for splash tests."""
    config = synthetic_inputs_defaults.copy()
    config["n_lat"] = 1
    config["n_lon"] = 1
    return driver.build_driver(
        modules=["synthetic_inputs", "resample", "splash"],
        config=config,
    )


@pytest.fixture
def synthetic_driver_4x5(synthetic_inputs_defaults):
    """Build driver with 4x5 grid."""
    config = synthetic_inputs_defaults.copy()
    config["n_lat"] = 4
    config["n_lon"] = 5
    return driver.build_driver(
        modules=["synthetic_inputs"],
        config=config,
    )


@pytest.fixture
def synthetic_driver_leap_year(synthetic_inputs_defaults):
    """Build driver with leap year (366 days)."""
    config = synthetic_inputs_defaults.copy()
    config["n_days"] = 366
    config["start_date"] = "2020-01-01"
    return driver.build_driver(
        modules=["synthetic_inputs"],
        config=config,
    )
