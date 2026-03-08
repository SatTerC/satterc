"""
Shared pytest fixtures for satterc integration tests.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def ref_datarray_2d():
    """Create a valid 2D DataArray with (time, pixel) dims and DatetimeIndex."""
    n_time = 10
    n_pixel = 4
    time_index = pd.date_range("2020-01-01", periods=n_time, freq="D")
    data = np.arange(n_time * n_pixel).reshape(n_time, n_pixel).astype(float)
    return xr.DataArray(
        data,
        dims=("time", "pixel"),
        coords={"time": time_index, "pixel": np.arange(n_pixel)},
        attrs={"units": "test_units", "long_name": "test_variable"},
    )


@pytest.fixture
def ref_datarray_1d():
    """Create a 1D DataArray with (pixel,) dims only."""
    n_pixel = 4
    data = np.arange(n_pixel).astype(float)
    return xr.DataArray(
        data,
        dims=("pixel",),
        coords={"pixel": np.arange(n_pixel)},
        attrs={"units": "test_units"},
    )


@pytest.fixture
def sample_numpy_array():
    """Create a simple numpy array for passthrough tests."""
    return np.array([1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def sample_numpy_array_2d():
    """Create a simple 2D numpy array."""
    return np.arange(12).reshape(3, 4).astype(float)
