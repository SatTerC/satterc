"""
Unit tests for the grid nodes in grid.py.

Tests the common_grid, stacked_grid, and split_grid nodes
using the synthetic NetCDF data files.
"""

import numpy as np
import pytest
import xarray as xr

from satterc.pipeline.inputs import grid


class TestCheckCommonGrid:
    """Tests for _check_common_grid function."""

    def test_identical_datasets_pass(self, daily_ds, static_ds):
        """Test that identical datasets pass the check."""
        grid._check_common_grid(daily_ds, static_ds)

    def test_mismatched_crs_raises(self, daily_ds):
        """Test that mismatched CRS raises MisalignedGridError."""
        ds2 = daily_ds.copy()
        ds2 = ds2.assign_attrs({"crs": "EPSG:4327"})

        with pytest.raises(grid.MisalignedGridError, match="Mismatched CRS"):
            grid._check_common_grid(daily_ds, ds2)


class TestCommonGrid:
    """Tests for common_grid function."""

    def test_common_grid_returns_dataset(self, common_grid_ds):
        """Test that common_grid returns a Dataset."""
        assert isinstance(common_grid_ds, xr.Dataset)

    def test_common_grid_has_two_data_vars(self, common_grid_ds):
        """Test that result has latitude, longitude as data variables."""
        assert set(common_grid_ds.data_vars) == {"latitude", "longitude"}

    def test_common_grid_has_crs(self, common_grid_ds):
        """Test that result has CRS metadata."""
        assert common_grid_ds.attrs.get("crs") == "EPSG:4326"

    def test_common_grid_has_xy_coords(self, common_grid_ds):
        """Test that x and y are coordinates in result."""
        assert "x" in common_grid_ds.coords
        assert "y" in common_grid_ds.coords

    def test_common_grid_lat_values(self, common_grid_ds):
        """Test latitude values are in expected UK range."""
        lat_values = common_grid_ds["latitude"].values
        assert np.all(lat_values >= 49.0)
        assert np.all(lat_values <= 55.0)

    def test_common_grid_lon_values(self, common_grid_ds):
        """Test longitude values are in expected UK range."""
        lon_values = common_grid_ds["longitude"].values
        assert np.all(lon_values >= -5.0)
        assert np.all(lon_values <= 3.0)

    def test_common_grid_xy_matches_input(self, common_grid_ds, daily_ds):
        """Test that x and y coordinate values match original input."""
        np.testing.assert_array_equal(
            common_grid_ds.coords["x"].values, daily_ds["x"].values
        )
        np.testing.assert_array_equal(
            common_grid_ds.coords["y"].values, daily_ds["y"].values
        )


class TestCommonGridStacked:
    """Tests for stacked_grid function."""

    def test_stacked_has_pixel_dimension(self, stacked_grid_ds):
        """Test that stacked result has pixel dimension."""
        assert "pixel" in stacked_grid_ds.dims

    def test_stacked_pixel_count(self, stacked_grid_ds):
        """Test that stacked result has correct number of pixels."""
        assert stacked_grid_ds.dims["pixel"] == 4  # 2x2 grid

    def test_stacked_preserves_lat_lon_vars(self, stacked_grid_ds):
        """Test that latitude and longitude variables are preserved after stacking."""
        assert set(stacked_grid_ds.data_vars) == {"latitude", "longitude"}

    def test_stacked_preserves_xy_coords(self, stacked_grid_ds):
        """Test that x and y coordinates are preserved after stacking."""
        assert "x" in stacked_grid_ds.coords
        assert "y" in stacked_grid_ds.coords


class TestUnpackCommonGrid:
    """Tests for split_grid function."""

    @pytest.fixture(scope="class")
    def split_result(self, stacked_grid_ds):
        return grid.split_grid(stacked_grid_ds)

    def test_unpack_returns_two_arrays(self, split_result):
        """Test that unpack returns a tuple of two DataArrays."""
        assert isinstance(split_result, tuple)
        assert len(split_result) == 2

    def test_unpack_latitude_values(self, split_result):
        """Test unpacked latitude values."""
        lat_da, _ = split_result
        assert lat_da.name == "latitude"
        assert np.all(lat_da.values >= 49.0)
        assert np.all(lat_da.values <= 55.0)

    def test_unpack_longitude_values(self, split_result):
        """Test unpacked longitude values."""
        _, lon_da = split_result
        assert lon_da.name == "longitude"
        assert np.all(lon_da.values >= -5.0)
        assert np.all(lon_da.values <= 3.0)

    def test_unpack_has_pixel_dimension(self, split_result):
        """Test unpacked arrays have pixel dimension."""
        lat_da, lon_da = split_result
        assert "pixel" in lat_da.dims
        assert "pixel" in lon_da.dims

    def test_unpack_pixel_count(self, split_result):
        """Test unpacked arrays have correct number of pixels."""
        lat_da, lon_da = split_result
        assert len(lat_da.pixel) == 4
        assert len(lon_da.pixel) == 4


class TestCommonGridPartialInputs:
    """Tests for common_grid with subsets of input datasets."""

    def test_daily_only(self, daily_ds):
        result = grid.common_grid(loaded_daily_inputs=daily_ds)
        assert isinstance(result, xr.Dataset)
        assert set(result.data_vars) == {"latitude", "longitude"}

    def test_static_only(self, static_ds):
        result = grid.common_grid(loaded_static_inputs=static_ds)
        assert isinstance(result, xr.Dataset)
        assert set(result.data_vars) == {"latitude", "longitude"}

    def test_daily_and_static(self, daily_ds, static_ds):
        result = grid.common_grid(
            loaded_daily_inputs=daily_ds, loaded_static_inputs=static_ds
        )
        assert isinstance(result, xr.Dataset)
        assert set(result.data_vars) == {"latitude", "longitude"}

    def test_no_inputs_raises(self):
        with pytest.raises(ValueError, match="At least one input"):
            grid.common_grid()

    def test_misaligned_labels_in_error(self, daily_ds):
        ds2 = daily_ds.copy()
        ds2 = ds2.assign_attrs({"crs": "EPSG:4327"})
        with pytest.raises(grid.MisalignedGridError, match="daily"):
            grid.common_grid(loaded_daily_inputs=daily_ds, loaded_weekly_inputs=ds2)


class TestIntegration:
    """Integration tests for the full grid pipeline."""

    def test_full_pipeline(self, common_grid_ds, stacked_grid_ds):
        """Test the full pipeline from inputs to unpacked coordinates."""
        assert isinstance(common_grid_ds, xr.Dataset)
        assert "pixel" in stacked_grid_ds.dims

        lat_da, lon_da = grid.split_grid(stacked_grid_ds)

        assert len(lat_da.pixel) == 4
        assert len(lon_da.pixel) == 4
        assert np.all(lat_da.values >= 49.0)
        assert np.all(lat_da.values <= 55.0)
        assert np.all(lon_da.values >= -5.0)
        assert np.all(lon_da.values <= 3.0)
        assert not np.any(np.isnan(lat_da.values))
        assert not np.any(np.isnan(lon_da.values))

    def test_xy_coords_survive_stacking(self, common_grid_ds, stacked_grid_ds):
        """Test that x and y coordinates are preserved after stacking."""
        x_stacked = stacked_grid_ds.coords["x"].values
        y_stacked = stacked_grid_ds.coords["y"].values

        assert len(x_stacked) == 4
        assert len(y_stacked) == 4

        assert set(common_grid_ds.coords["x"].values) == set(x_stacked)
        assert set(common_grid_ds.coords["y"].values) == set(y_stacked)
