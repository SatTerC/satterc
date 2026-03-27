"""
Unit tests for the grid nodes in grid.py.

Tests the common_grid, common_grid_stacked, and unpack_common_grid nodes
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

    def test_common_grid_returns_dataset(
        self, daily_ds, weekly_ds, monthly_ds, static_ds
    ):
        """Test that common_grid returns a Dataset."""
        result = grid.common_grid(daily_ds, weekly_ds, monthly_ds, static_ds)
        assert isinstance(result, xr.Dataset)

    def test_common_grid_has_two_data_vars(
        self, daily_ds, weekly_ds, monthly_ds, static_ds
    ):
        """Test that result has latitude, longitude as data variables."""
        result = grid.common_grid(daily_ds, weekly_ds, monthly_ds, static_ds)
        expected_vars = {"latitude", "longitude"}
        assert set(result.data_vars) == expected_vars

    def test_common_grid_has_crs(self, daily_ds, weekly_ds, monthly_ds, static_ds):
        """Test that result has CRS metadata."""
        result = grid.common_grid(daily_ds, weekly_ds, monthly_ds, static_ds)
        assert result.attrs.get("crs") == "EPSG:4326"

    def test_common_grid_has_xy_coords(
        self, daily_ds, weekly_ds, monthly_ds, static_ds
    ):
        """Test that x and y are coordinates in result."""
        result = grid.common_grid(daily_ds, weekly_ds, monthly_ds, static_ds)
        assert "x" in result.coords
        assert "y" in result.coords

    def test_common_grid_lat_values(self, daily_ds, weekly_ds, monthly_ds, static_ds):
        """Test latitude values are in expected UK range."""
        result = grid.common_grid(daily_ds, weekly_ds, monthly_ds, static_ds)
        lat_values = result["latitude"].values
        assert np.all(lat_values >= 49.0)
        assert np.all(lat_values <= 55.0)

    def test_common_grid_lon_values(self, daily_ds, weekly_ds, monthly_ds, static_ds):
        """Test longitude values are in expected UK range."""
        result = grid.common_grid(daily_ds, weekly_ds, monthly_ds, static_ds)
        lon_values = result["longitude"].values
        assert np.all(lon_values >= -5.0)
        assert np.all(lon_values <= 3.0)

    def test_common_grid_xy_matches_input(
        self, daily_ds, weekly_ds, monthly_ds, static_ds
    ):
        """Test that x and y coordinate values match original input."""
        result = grid.common_grid(daily_ds, weekly_ds, monthly_ds, static_ds)
        np.testing.assert_array_equal(result.coords["x"].values, daily_ds["x"].values)
        np.testing.assert_array_equal(result.coords["y"].values, daily_ds["y"].values)


class TestCommonGridStacked:
    """Tests for common_grid_stacked function."""

    def test_stacked_has_pixel_dimension(
        self, daily_ds, weekly_ds, monthly_ds, static_ds
    ):
        """Test that stacked result has pixel dimension."""
        common = grid.common_grid(daily_ds, weekly_ds, monthly_ds, static_ds)
        stacked = grid.common_grid_stacked(common)
        assert "pixel" in stacked.dims

    def test_stacked_pixel_count(self, daily_ds, weekly_ds, monthly_ds, static_ds):
        """Test that stacked result has correct number of pixels."""
        common = grid.common_grid(daily_ds, weekly_ds, monthly_ds, static_ds)
        stacked = grid.common_grid_stacked(common)
        assert stacked.dims["pixel"] == 4  # 2x2 grid

    def test_stacked_preserves_lat_lon_vars(
        self, daily_ds, weekly_ds, monthly_ds, static_ds
    ):
        """Test that latitude and longitude variables are preserved after stacking."""
        common = grid.common_grid(daily_ds, weekly_ds, monthly_ds, static_ds)
        stacked = grid.common_grid_stacked(common)
        expected_vars = {"latitude", "longitude"}
        assert set(stacked.data_vars) == expected_vars

    def test_stacked_preserves_xy_coords(
        self, daily_ds, weekly_ds, monthly_ds, static_ds
    ):
        """Test that x and y coordinates are preserved after stacking."""
        common = grid.common_grid(daily_ds, weekly_ds, monthly_ds, static_ds)
        stacked = grid.common_grid_stacked(common)
        # After stacking, x and y should still be accessible as coordinates
        assert "x" in stacked.coords
        assert "y" in stacked.coords


class TestUnpackCommonGrid:
    """Tests for unpack_common_grid function."""

    def test_unpack_returns_two_arrays(
        self, daily_ds, weekly_ds, monthly_ds, static_ds
    ):
        """Test that unpack returns a tuple of two DataArrays."""
        common = grid.common_grid(daily_ds, weekly_ds, monthly_ds, static_ds)
        stacked = grid.common_grid_stacked(common)
        result = grid.unpack_common_grid(stacked)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_unpack_latitude_values(self, daily_ds, weekly_ds, monthly_ds, static_ds):
        """Test unpacked latitude values."""
        common = grid.common_grid(daily_ds, weekly_ds, monthly_ds, static_ds)
        stacked = grid.common_grid_stacked(common)
        lat_da, lon_da = grid.unpack_common_grid(stacked)

        assert lat_da.name == "latitude"
        assert np.all(lat_da.values >= 49.0)
        assert np.all(lat_da.values <= 55.0)

    def test_unpack_longitude_values(self, daily_ds, weekly_ds, monthly_ds, static_ds):
        """Test unpacked longitude values."""
        common = grid.common_grid(daily_ds, weekly_ds, monthly_ds, static_ds)
        stacked = grid.common_grid_stacked(common)
        lat_da, lon_da = grid.unpack_common_grid(stacked)

        assert lon_da.name == "longitude"
        assert np.all(lon_da.values >= -5.0)
        assert np.all(lon_da.values <= 3.0)

    def test_unpack_has_pixel_dimension(
        self, daily_ds, weekly_ds, monthly_ds, static_ds
    ):
        """Test unpacked arrays have pixel dimension."""
        common = grid.common_grid(daily_ds, weekly_ds, monthly_ds, static_ds)
        stacked = grid.common_grid_stacked(common)
        lat_da, lon_da = grid.unpack_common_grid(stacked)

        assert "pixel" in lat_da.dims
        assert "pixel" in lon_da.dims

    def test_unpack_pixel_count(self, daily_ds, weekly_ds, monthly_ds, static_ds):
        """Test unpacked arrays have correct number of pixels."""
        common = grid.common_grid(daily_ds, weekly_ds, monthly_ds, static_ds)
        stacked = grid.common_grid_stacked(common)
        lat_da, lon_da = grid.unpack_common_grid(stacked)

        assert len(lat_da.pixel) == 4
        assert len(lon_da.pixel) == 4


class TestIntegration:
    """Integration tests for the full grid pipeline."""

    def test_full_pipeline(self, daily_ds, weekly_ds, monthly_ds, static_ds):
        """Test the full pipeline from inputs to unpacked coordinates."""
        # Step 1: Create common grid
        common = grid.common_grid(daily_ds, weekly_ds, monthly_ds, static_ds)
        assert isinstance(common, xr.Dataset)

        # Step 2: Stack
        stacked = grid.common_grid_stacked(common)
        assert "pixel" in stacked.dims

        # Step 3: Unpack
        lat_da, lon_da = grid.unpack_common_grid(stacked)

        # Verify outputs
        assert len(lat_da.pixel) == 4
        assert len(lon_da.pixel) == 4

        # Verify lat/lon are in UK bounds
        assert np.all(lat_da.values >= 49.0)
        assert np.all(lat_da.values <= 55.0)
        assert np.all(lon_da.values >= -5.0)
        assert np.all(lon_da.values <= 3.0)

        # Verify no NaN values
        assert not np.any(np.isnan(lat_da.values))
        assert not np.any(np.isnan(lon_da.values))

    def test_transform_is_invertible(self, daily_ds, weekly_ds, monthly_ds, static_ds):
        """Test that coordinate transform is mathematically consistent."""
        from pyproj import Transformer

        common = grid.common_grid(daily_ds, weekly_ds, monthly_ds, static_ds)

        # Get 2D arrays for lat/lon
        lat_2d = common["latitude"].values
        lon_2d = common["longitude"].values

        # Transform back to original CRS
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:4326", always_xy=True)
        x_back, y_back = transformer.transform(lon_2d, lat_2d)

        # Get original 2D meshgrid
        x_2d, y_2d = np.meshgrid(
            common.coords["x"].values, common.coords["y"].values, indexing="ij"
        )

        # Should be very close (within floating point tolerance)
        np.testing.assert_allclose(x_2d, x_back, rtol=1e-10)
        np.testing.assert_allclose(y_2d, y_back, rtol=1e-10)

    def test_xy_coords_survive_stacking(
        self, daily_ds, weekly_ds, monthly_ds, static_ds
    ):
        """Test that x and y coordinates are preserved after stacking."""
        common = grid.common_grid(daily_ds, weekly_ds, monthly_ds, static_ds)
        stacked = grid.common_grid_stacked(common)

        # After stacking, x and y become 1D arrays indexed by pixel
        # They should contain all combinations of the original x and y values
        x_stacked = stacked.coords["x"].values
        y_stacked = stacked.coords["y"].values

        # Should have 4 values (2x2 grid flattened)
        assert len(x_stacked) == 4
        assert len(y_stacked) == 4

        # Original unique values should be preserved
        orig_x = set(common.coords["x"].values)
        orig_y = set(common.coords["y"].values)
        stacked_x = set(x_stacked)
        stacked_y = set(y_stacked)

        assert orig_x == stacked_x
        assert orig_y == stacked_y
