"""Unit tests for the resample pipeline module.

The resample functions are wrapped by @FixedResolve (a Hamilton class-based
decorator) and cannot be called as plain Python functions. Tests exercise them
via a minimal Hamilton driver with DataArrays injected directly as inputs.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hamilton import driver
from hamilton.settings import ENABLE_POWER_USER_MODE

from satterc.pipeline import resample


def _make_daily_da(n_days: int = 365, n_pixel: int = 4, seed: int = 0) -> xr.DataArray:
    """Create a synthetic daily DataArray with (time, pixel) dims."""
    rng = np.random.default_rng(seed)
    time_index = pd.date_range("2020-01-01", periods=n_days, freq="D")
    data = rng.standard_normal((n_days, n_pixel))
    return xr.DataArray(
        data,
        dims=("time", "pixel"),
        coords={"time": time_index, "pixel": np.arange(n_pixel)},
    )


def _make_weekly_da(n_weeks: int = 52, n_pixel: int = 4, seed: int = 1) -> xr.DataArray:
    """Create a synthetic weekly DataArray with (time, pixel) dims."""
    rng = np.random.default_rng(seed)
    time_index = pd.date_range("2020-01-01", periods=n_weeks, freq="7D")
    data = rng.standard_normal((n_weeks, n_pixel))
    return xr.DataArray(
        data,
        dims=("time", "pixel"),
        coords={"time": time_index, "pixel": np.arange(n_pixel)},
    )


def _resample_config(**kwargs) -> dict:
    base = {
        "daily_to_weekly": [],
        "daily_to_monthly": [],
        "weekly_to_monthly": [],
        ENABLE_POWER_USER_MODE: True,
    }
    base.update(kwargs)
    return base


@pytest.fixture(scope="module")
def daily_to_weekly_driver():
    return (
        driver.Builder()
        .with_modules(resample)
        .with_config(_resample_config(daily_to_weekly=["temperature"]))
        .build()
    )


@pytest.fixture(scope="module")
def daily_to_monthly_driver():
    return (
        driver.Builder()
        .with_modules(resample)
        .with_config(_resample_config(daily_to_monthly=["temperature"]))
        .build()
    )


@pytest.fixture(scope="module")
def weekly_to_monthly_driver():
    return (
        driver.Builder()
        .with_modules(resample)
        .with_config(_resample_config(weekly_to_monthly=["temperature"]))
        .build()
    )


class TestDailyToWeekly:
    """Tests for daily → weekly resampling."""

    @pytest.fixture(scope="class")
    def result(self, daily_to_weekly_driver):
        da = _make_daily_da(n_days=365)
        return daily_to_weekly_driver.execute(
            ["temperature_weekly"],
            inputs={"temperature_daily": da},
        )["temperature_weekly"]

    def test_output_is_dataarray(self, result):
        assert isinstance(result, xr.DataArray)

    def test_pixel_dimension_preserved(self, result):
        assert "pixel" in result.dims
        assert result.sizes["pixel"] == 4

    def test_weekly_output_shape(self, result):
        n_weeks = result.sizes["time"]
        assert 50 <= n_weeks <= 54

    def test_output_frequency_is_weekly(self, result):
        inferred = pd.infer_freq(result.coords["time"].values)
        assert inferred is not None
        assert inferred.startswith(("W", "7D"))

    def test_values_are_means(self):
        """Verify resampled values match manually computed weekly means."""
        da = _make_daily_da(n_days=14, n_pixel=1)
        result = (
            driver.Builder()
            .with_modules(resample)
            .with_config(_resample_config(daily_to_weekly=["temperature"]))
            .build()
            .execute(
                ["temperature_weekly"],
                inputs={"temperature_daily": da},
            )["temperature_weekly"]
        )
        expected_week1 = da.isel(time=slice(0, 7)).mean("time").values
        np.testing.assert_allclose(result.isel(time=0).values, expected_week1)


class TestDailyToMonthly:
    """Tests for daily → monthly resampling."""

    @pytest.fixture(scope="class")
    def result(self, daily_to_monthly_driver):
        da = _make_daily_da(n_days=365)
        return daily_to_monthly_driver.execute(
            ["temperature_monthly"],
            inputs={"temperature_daily": da},
        )["temperature_monthly"]

    def test_output_is_dataarray(self, result):
        assert isinstance(result, xr.DataArray)

    def test_pixel_dimension_preserved(self, result):
        assert "pixel" in result.dims
        assert result.sizes["pixel"] == 4

    def test_monthly_output_shape(self, result):
        assert result.sizes["time"] == 12

    def test_output_frequency_is_monthly(self, result):
        inferred = pd.infer_freq(result.coords["time"].values)
        assert inferred is not None
        assert inferred.startswith(("ME", "MS"))


class TestWeeklyToMonthly:
    """Tests for weekly → monthly resampling."""

    @pytest.fixture(scope="class")
    def result(self, weekly_to_monthly_driver):
        da = _make_weekly_da(n_weeks=52)
        return weekly_to_monthly_driver.execute(
            ["temperature_monthly"],
            inputs={"temperature_weekly": da},
        )["temperature_monthly"]

    def test_output_is_dataarray(self, result):
        assert isinstance(result, xr.DataArray)

    def test_pixel_dimension_preserved(self, result):
        assert "pixel" in result.dims
        assert result.sizes["pixel"] == 4

    def test_monthly_output_shape(self, result):
        assert result.sizes["time"] == 12
