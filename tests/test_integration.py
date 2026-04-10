"""End-to-end integration tests for the full satterc pipeline.

These tests build a Hamilton driver from the test config and synthetic data,
then execute pipeline nodes to verify the full stack — config → driver →
input loading → grid → (optionally) output writing — works end-to-end.
"""

import numpy as np
import pytest


N_PIXELS = 4  # 2x2 grid
N_DAYS = 365
N_MONTHS = 12


class TestInputNodes:
    """Test that input nodes load and return correctly shaped DataArrays."""

    def test_daily_input_shape(self, pipeline_driver):
        result = pipeline_driver.execute(["temperature_celcius_daily"])
        da = result["temperature_celcius_daily"]
        assert da.sizes["time"] == N_DAYS
        assert da.sizes["pixel"] == N_PIXELS

    def test_weekly_input_shape(self, pipeline_driver):
        result = pipeline_driver.execute(["co2_ppm_weekly"])
        da = result["co2_ppm_weekly"]
        assert 50 <= da.sizes["time"] <= 54
        assert da.sizes["pixel"] == N_PIXELS

    def test_monthly_input_shape(self, pipeline_driver):
        result = pipeline_driver.execute(["dummy_variable_monthly"])
        da = result["dummy_variable_monthly"]
        assert da.sizes["time"] == N_MONTHS
        assert da.sizes["pixel"] == N_PIXELS

    def test_static_input_shape(self, pipeline_driver):
        result = pipeline_driver.execute(["elevation"])
        da = result["elevation"]
        assert da.sizes["pixel"] == N_PIXELS
        assert "time" not in da.dims

    def test_no_nan_in_daily_inputs(self, pipeline_driver):
        result = pipeline_driver.execute(
            ["temperature_celcius_daily", "precipitation_mm_daily", "sunshine_fraction_daily"]
        )
        for name, da in result.items():
            assert not np.any(np.isnan(da.values)), f"{name} contains NaN"

    def test_no_nan_in_static_inputs(self, pipeline_driver):
        result = pipeline_driver.execute(["elevation", "max_soil_moisture", "clay_content"])
        for name, da in result.items():
            assert not np.any(np.isnan(da.values)), f"{name} contains NaN"


class TestGridNodes:
    """Test that the grid pipeline nodes execute and return sensible values."""

    def test_latitude_in_uk_bounds(self, pipeline_driver):
        result = pipeline_driver.execute(["latitude"])
        lat = result["latitude"].values
        assert np.all(lat >= 49.0)
        assert np.all(lat <= 55.0)

    def test_longitude_in_uk_bounds(self, pipeline_driver):
        result = pipeline_driver.execute(["longitude"])
        lon = result["longitude"].values
        assert np.all(lon >= -5.0)
        assert np.all(lon <= 3.0)

    def test_grid_pixel_count(self, pipeline_driver):
        result = pipeline_driver.execute(["latitude", "longitude"])
        assert result["latitude"].sizes["pixel"] == N_PIXELS
        assert result["longitude"].sizes["pixel"] == N_PIXELS


class TestFullPipeline:
    """Test the complete pipeline execution including the targets list."""

    def test_pipeline_executes_without_error(self, pipeline_driver, pipeline_config):
        """Pipeline with no output vars should execute without error."""
        pipeline_driver.execute(pipeline_config["targets"])

    def test_no_output_files_written_when_vars_empty(
        self, pipeline_driver, pipeline_config, synthetic_data_dir
    ):
        """With empty *_outputs_vars, output modules are not loaded — no files created."""
        pipeline_driver.execute(pipeline_config["targets"])
        assert not (synthetic_data_dir / "out_daily.nc").exists()
        assert not (synthetic_data_dir / "out_weekly.nc").exists()
        assert not (synthetic_data_dir / "out_monthly.nc").exists()
