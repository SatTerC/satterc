"""
Tests for input/output shape handling across spatial configurations.

The stack_spatial_dims utility — called by every temporal input module —
uses rioxarray to identify spatial dimensions and requires CRS metadata.
These tests probe three spatial configurations:

  grid_2d (time, y, x)      — current production case: fully supported
  single_point (time,)      — documents current behaviour (not yet supported)
  multi_point (time, pixel) — documents current behaviour (not yet supported)

The static (no-time) equivalents of each shape are not tested here.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from satterc.pipeline.inputs._utils import stack_spatial_dims


N_TIMES = 10
N_POINTS = 5
N_Y, N_X = 2, 3
TIMES = pd.date_range("2020-01-01", periods=N_TIMES, freq="D")
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def single_point_ds():
    """(time,) only — single location, no spatial dimensions."""
    return xr.Dataset(
        {"var_a": (["time"], RNG.random(N_TIMES))},
        coords={"time": TIMES},
    )


@pytest.fixture
def multi_point_ds():
    """(time, pixel) — multiple locations with an integer pixel index, no grid structure."""
    return xr.Dataset(
        {"var_a": (["time", "pixel"], RNG.random((N_TIMES, N_POINTS)))},
        coords={"time": TIMES, "pixel": np.arange(N_POINTS)},
    )


@pytest.fixture
def grid_2d_ds():
    """(time, y, x) — regular 2D grid with CRS metadata."""
    y = np.linspace(51.0, 52.0, N_Y)
    x = np.linspace(-1.0, 1.0, N_X)
    ds = xr.Dataset(
        {
            "var_a": (["time", "y", "x"], RNG.random((N_TIMES, N_Y, N_X))),
            "var_b": (["time", "y", "x"], RNG.random((N_TIMES, N_Y, N_X))),
        },
        coords={"time": TIMES, "y": y, "x": x},
    )
    return ds.rio.write_crs("EPSG:4326")


@pytest.fixture
def stacked_2d(grid_2d_ds):
    return stack_spatial_dims(grid_2d_ds)


# ---------------------------------------------------------------------------
# 2D grid (time, y, x) — the current production case
# ---------------------------------------------------------------------------


class TestGrid2DStack:
    """stack_spatial_dims collapses (y, x) into a single pixel dimension."""

    def test_produces_pixel_dim(self, stacked_2d):
        assert "pixel" in stacked_2d.dims

    def test_removes_spatial_dims(self, stacked_2d):
        assert "y" not in stacked_2d.dims
        assert "x" not in stacked_2d.dims

    def test_pixel_count(self, stacked_2d):
        assert stacked_2d.sizes["pixel"] == N_Y * N_X

    def test_time_preserved(self, stacked_2d):
        assert "time" in stacked_2d.dims
        assert stacked_2d.sizes["time"] == N_TIMES

    def test_data_vars_preserved(self, stacked_2d):
        assert set(stacked_2d.data_vars) == {"var_a", "var_b"}

    def test_xy_survive_as_pixel_coords(self, stacked_2d):
        """y and x coordinate arrays must be retained so unstack can reconstruct the grid."""
        assert "x" in stacked_2d.coords
        assert "y" in stacked_2d.coords


class TestGrid2DUnstack:
    """Unstacking the pixel MultiIndex restores the original (y, x) grid."""

    def test_restores_spatial_dims(self, stacked_2d):
        restored = stacked_2d.unstack("pixel")
        assert "y" in restored.dims
        assert "x" in restored.dims
        assert "pixel" not in restored.dims

    def test_correct_spatial_shape(self, stacked_2d):
        restored = stacked_2d.unstack("pixel")
        assert restored.sizes["y"] == N_Y
        assert restored.sizes["x"] == N_X

    def test_roundtrip_preserves_values(self, grid_2d_ds, stacked_2d):
        restored = stacked_2d.unstack("pixel")
        xr.testing.assert_allclose(
            restored["var_a"].sortby(["y", "x"]),
            grid_2d_ds["var_a"].sortby(["y", "x"]),
        )
        xr.testing.assert_allclose(
            restored["var_b"].sortby(["y", "x"]),
            grid_2d_ds["var_b"].sortby(["y", "x"]),
        )


class TestGrid2DPipelineSimulation:
    """Simulate the split_inputs → (processing) → merge_outputs → unstack chain."""

    def test_split_produces_named_daily_arrays(self, stacked_2d):
        """split_daily_inputs extracts one (time, pixel) DataArray per variable."""
        split = {f"{v}_daily": stacked_2d[v] for v in stacked_2d.data_vars}
        assert set(split.keys()) == {"var_a_daily", "var_b_daily"}
        for da in split.values():
            assert set(da.dims) == {"time", "pixel"}
            assert da.sizes["time"] == N_TIMES
            assert da.sizes["pixel"] == N_Y * N_X

    def test_merge_then_unstack_roundtrip(self, grid_2d_ds, stacked_2d):
        """xr.merge (merged_daily_outputs) → unstack (unstacked_daily_outputs) reproduces the original grid."""
        arrays = [stacked_2d[v] for v in stacked_2d.data_vars]
        merged = xr.merge(arrays)
        restored = merged.unstack("pixel")

        xr.testing.assert_allclose(
            restored["var_a"].sortby(["y", "x"]),
            grid_2d_ds["var_a"].sortby(["y", "x"]),
        )

    def test_merged_dataset_has_pixel_dim(self, stacked_2d):
        arrays = [stacked_2d[v] for v in stacked_2d.data_vars]
        merged = xr.merge(arrays)
        assert "pixel" in merged.dims
        assert "time" in merged.dims


# ---------------------------------------------------------------------------
# Single point (time,) — documents current limitation
# ---------------------------------------------------------------------------


class TestSinglePoint:
    """(time,) — single-location time series with no spatial dimensions.

    stack_spatial_dims delegates to rioxarray to discover spatial dims; a
    dataset with no spatial structure cannot be stacked. These tests document
    the current failure mode and will need to be revised when single-point
    support is added (likely via a separate input module).
    """

    def test_stack_raises(self, single_point_ds):
        with pytest.raises(Exception):
            stack_spatial_dims(single_point_ds)


# ---------------------------------------------------------------------------
# Multi-point (time, pixel) — documents current limitation
# ---------------------------------------------------------------------------


class TestMultiPoint:
    """(time, pixel) — unstructured multi-point data with an integer pixel index.

    rioxarray cannot identify spatial dims on a dataset whose non-time
    dimension is an ordinary integer index rather than a (y, x) MultiIndex.
    These tests document the current failure mode.

    Note: even if stacking were skipped, the output unstack("pixel") step
    also requires a (y, x) MultiIndex — a plain integer index cannot be
    unpacked into a 2D grid.
    """

    def test_stack_raises(self, multi_point_ds):
        with pytest.raises(Exception):
            stack_spatial_dims(multi_point_ds)

    def test_unstack_integer_pixel_raises(self, multi_point_ds):
        """Unstacking a plain integer pixel index raises because there is no (y, x) MultiIndex to expand."""
        with pytest.raises(Exception):
            multi_point_ds.unstack("pixel")
