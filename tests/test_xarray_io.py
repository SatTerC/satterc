"""
Tests for the xarray_io decorator in satterc.utils.

This module can be run via pytest:
    pytest tests/test_xarray_io.py -v
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from satterc.models._utils import xarray_io


class TestPassthroughBehavior:
    """Tests for decorator passthrough when no DataArray inputs are provided."""

    def test_numpy_array_input_passthrough(self, sample_numpy_array):
        @xarray_io()
        def func(arr):
            return arr * 2

        result = func(sample_numpy_array)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, sample_numpy_array * 2)

    def test_scalar_input_passthrough(self):
        @xarray_io()
        def func(x):
            return x + 1

        result = func(5)
        assert result == 6

    def test_numpy_scalar_input_passthrough(self):
        @xarray_io()
        def func(x):
            return x + 1

        result = func(np.float64(5.0))
        assert result == np.float64(6.0)

    def test_multiple_numpy_arrays_passthrough(self, sample_numpy_array):
        @xarray_io()
        def func(arr1, arr2):
            return arr1 + arr2

        result = func(sample_numpy_array, sample_numpy_array)
        np.testing.assert_array_equal(result, sample_numpy_array * 2)

    def test_mixed_args_kwargs_passthrough(self, sample_numpy_array):
        @xarray_io()
        def func(arr, scalar, multiplier=1):
            return arr * scalar * multiplier

        result = func(sample_numpy_array, 2, multiplier=3)
        np.testing.assert_array_equal(result, sample_numpy_array * 6)


class TestInputDataArrayDimensionVariations:
    """Tests for different DataArray input dimension configurations."""

    def test_datarray_2d_input_time_pixel(self, ref_datarray_2d):
        @xarray_io()
        def func(arr):
            return arr * 2

        result = func(ref_datarray_2d)
        assert isinstance(result, xr.DataArray)
        assert result.dims == ("time", "pixel")
        np.testing.assert_array_equal(result.values, ref_datarray_2d.values * 2)

    @pytest.mark.xfail(reason="decorator requires 2D DataArray with time dimension")
    def test_datarray_1d_input_pixel_only(self, ref_datarray_1d):
        @xarray_io()
        def func(arr):
            return arr * 2

        result = func(ref_datarray_1d)
        assert isinstance(result, xr.DataArray)
        assert result.dims == ("pixel",)
        np.testing.assert_array_equal(result.values, ref_datarray_1d.values * 2)

    @pytest.mark.xfail(reason="decorator requires 2D DataArray with time dimension")
    def test_datarray_1d_input_as_kwarg(self, ref_datarray_1d):
        @xarray_io()
        def func(arr=None):
            return arr * 2

        result = func(arr=ref_datarray_1d)
        assert isinstance(result, xr.DataArray)
        assert result.dims == ("pixel",)


class TestOutputDimensionTests:
    """Tests for different output array dimensions."""

    def test_output_0d_scalar(self, ref_datarray_2d):
        @xarray_io()
        def func(arr):
            return float(arr.sum())

        result = func(ref_datarray_2d)
        assert isinstance(result, (float, np.floating))
        assert result == float(ref_datarray_2d.values.sum())

    def test_output_1d(self, ref_datarray_2d):
        @xarray_io()
        def func(arr):
            return arr.mean(axis=0)

        result = func(ref_datarray_2d)
        assert isinstance(result, xr.DataArray)
        assert result.dims == ("pixel",)
        np.testing.assert_array_equal(
            result.values, ref_datarray_2d.values.mean(axis=0)
        )

    def test_output_2d(self, ref_datarray_2d):
        @xarray_io()
        def func(arr):
            return arr * 2

        result = func(ref_datarray_2d)
        assert isinstance(result, xr.DataArray)
        assert result.dims == ("time", "pixel")
        np.testing.assert_array_equal(result.values, ref_datarray_2d.values * 2)

    def test_output_3d_raises_error(self, ref_datarray_2d):
        @xarray_io()
        def func(arr):
            return np.expand_dims(arr, axis=0)

        with pytest.raises(Exception, match="no"):
            func(ref_datarray_2d)


class TestMultipleArrayInputs:
    """Tests for scenarios with multiple DataArray inputs."""

    def test_multiple_datarray_inputs(self, ref_datarray_2d):
        @xarray_io()
        def func(arr1, arr2):
            return arr1 + arr2

        result = func(ref_datarray_2d, ref_datarray_2d)
        assert isinstance(result, xr.DataArray)
        np.testing.assert_array_equal(result.values, ref_datarray_2d.values * 2)

    def test_multiple_datarray_inputs_uses_first_valid_reference(self):
        n_time = 10
        n_pixel = 4
        time_index = pd.date_range("2020-01-01", periods=n_time, freq="D")

        da1 = xr.DataArray(
            np.arange(n_time * n_pixel).reshape(n_time, n_pixel).astype(float),
            dims=("time", "pixel"),
            coords={"time": time_index, "pixel": np.arange(n_pixel)},
            attrs={"source": "da1"},
        )

        da2 = xr.DataArray(
            np.arange(n_time * n_pixel, n_time * n_pixel * 2)
            .reshape(n_time, n_pixel)
            .astype(float),
            dims=("time", "pixel"),
            coords={"time": time_index, "pixel": np.arange(n_pixel)},
            attrs={"source": "da2"},
        )

        @xarray_io()
        def func(arr1, arr2):
            return arr1 + arr2

        result = func(da1, da2)
        assert isinstance(result, xr.DataArray)
        assert result.attrs["source"] == "da1"

    def test_mixed_numpy_and_datarray_inputs(self, ref_datarray_2d, sample_numpy_array):
        @xarray_io()
        def func(da, arr):
            return da * arr

        result = func(ref_datarray_2d, sample_numpy_array)
        assert isinstance(result, xr.DataArray)
        assert result.dims == ("time", "pixel")


class TestReturnValueVariations:
    """Tests for different return value types from decorated functions."""

    def test_return_dict_of_arrays(self, ref_datarray_2d):
        @xarray_io()
        def func(arr):
            return {"output1": arr * 2, "output2": arr.sum(axis=0)}

        result = func(ref_datarray_2d)
        assert isinstance(result, dict)
        assert "output1" in result
        assert "output2" in result
        assert isinstance(result["output1"], xr.DataArray)
        assert isinstance(result["output2"], xr.DataArray)
        assert result["output1"].dims == ("time", "pixel")
        assert result["output2"].dims == ("pixel",)

    def test_return_list_of_arrays(self, ref_datarray_2d):
        @xarray_io()
        def func(arr):
            return [arr * 2, arr.sum(axis=0)]

        result = func(ref_datarray_2d)
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], xr.DataArray)
        assert isinstance(result[1], xr.DataArray)

    def test_return_tuple_of_arrays(self, ref_datarray_2d):
        @xarray_io()
        def func(arr):
            return (arr * 2, arr.sum(axis=0))

        result = func(ref_datarray_2d)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], xr.DataArray)
        assert isinstance(result[1], xr.DataArray)


class TestErrorCases:
    """Tests for error conditions in the decorator."""

    def test_no_valid_reference_raises(self):
        da_invalid = xr.DataArray(
            np.arange(12).reshape(3, 4),
            dims=("lat", "lon"),
            coords={"lat": [1, 2, 3], "lon": [1, 2, 3, 4]},
        )

        @xarray_io()
        def func(arr):
            return arr

        with pytest.raises(
            Exception, match="None of the xarray.DataArray inputs satisfy"
        ):
            func(da_invalid)

    def test_invalid_time_index_raises(self):
        da_invalid = xr.DataArray(
            np.arange(12).reshape(3, 4),
            dims=("time", "pixel"),
            coords={"time": [1, 2, 3], "pixel": [0, 1, 2, 3]},
        )

        @xarray_io()
        def func(arr):
            return arr

        with pytest.raises(
            Exception, match="None of the xarray.DataArray inputs satisfy"
        ):
            func(da_invalid)


class TestMetadataPreservation:
    """Tests that metadata (attrs, coords) is correctly preserved."""

    def test_attrs_preserved_in_output(self, ref_datarray_2d):
        @xarray_io()
        def func(arr):
            return arr * 2

        result = func(ref_datarray_2d)
        assert result.attrs == ref_datarray_2d.attrs

    def test_coords_preserved_in_output_2d(self, ref_datarray_2d):
        @xarray_io()
        def func(arr):
            return arr * 2

        result = func(ref_datarray_2d)
        assert "time" in result.coords
        assert "pixel" in result.coords

    def test_coords_preserved_in_output_1d(self, ref_datarray_2d):
        @xarray_io()
        def func(arr):
            return arr.mean(axis=0)

        result = func(ref_datarray_2d)
        assert "pixel" in result.coords
        assert "time" not in result.coords

    def test_name_not_preserved(self, ref_datarray_2d):
        ref_datarray_2d.name = "test_variable"

        @xarray_io()
        def func(arr):
            return arr * 2

        result = func(ref_datarray_2d)
        assert result.name is None
