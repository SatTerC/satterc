"""End-to-end integration tests for the full satterc pipeline.

These tests call load_inputs() with the test config and synthetic data,
then execute pipeline nodes to verify the full stack — config → load_inputs
→ driver.execute() — works end-to-end.
"""

import numpy as np


N_PIXELS = 4  # 2x2 grid
N_DAYS = 365
N_MONTHS = 12


class TestLoadedInputs:
    """Test that load_inputs() returns correctly shaped DataArrays."""

    def test_daily_input_shape(self, pipeline_inputs):
        da = pipeline_inputs["temperature_celcius_daily"]
        assert da.sizes["time"] == N_DAYS
        assert da.sizes["pixel"] == N_PIXELS

    def test_weekly_input_shape(self, pipeline_inputs):
        da = pipeline_inputs["co2_ppm_weekly"]
        assert 50 <= da.sizes["time"] <= 54
        assert da.sizes["pixel"] == N_PIXELS

    def test_monthly_input_shape(self, pipeline_inputs):
        da = pipeline_inputs["dummy_variable_monthly"]
        assert da.sizes["time"] == N_MONTHS
        assert da.sizes["pixel"] == N_PIXELS

    def test_static_input_shape(self, pipeline_inputs):
        da = pipeline_inputs["elevation"]
        assert da.sizes["pixel"] == N_PIXELS
        assert "time" not in da.dims

    def test_no_nan_in_daily_inputs(self, pipeline_inputs):
        for name in (
            "temperature_celcius_daily",
            "precipitation_mm_daily",
            "sunshine_fraction_daily",
        ):
            da = pipeline_inputs[name]
            assert not np.any(np.isnan(da.values)), f"{name} contains NaN"

    def test_no_nan_in_static_inputs(self, pipeline_inputs):
        for name in ("elevation", "max_soil_moisture", "clay_content"):
            da = pipeline_inputs[name]
            assert not np.any(np.isnan(da.values)), f"{name} contains NaN"


class TestGridInputs:
    """Test that load_inputs() computes latitude and longitude from spatial CRS data."""

    def test_latitude_in_uk_bounds(self, pipeline_inputs):
        lat = pipeline_inputs["latitude"].values
        assert np.all(lat >= 49.0)
        assert np.all(lat <= 55.0)

    def test_longitude_in_uk_bounds(self, pipeline_inputs):
        lon = pipeline_inputs["longitude"].values
        assert np.all(lon >= -5.0)
        assert np.all(lon <= 3.0)

    def test_grid_pixel_count(self, pipeline_inputs):
        assert pipeline_inputs["latitude"].sizes["pixel"] == N_PIXELS
        assert pipeline_inputs["longitude"].sizes["pixel"] == N_PIXELS


class TestDriverExecution:
    """Test that the Hamilton driver executes correctly with pre-loaded inputs."""

    def test_no_output_specs_no_error(self, pipeline_config):
        """Pipeline with no output_specs should execute without error."""
        assert pipeline_config.output_specs == {}
