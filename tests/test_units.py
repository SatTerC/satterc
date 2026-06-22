"""Tests for unit declarations and runtime unit validation."""

import warnings
from typing import Annotated, TypedDict

import numpy as np
import pint
import pytest
import xarray as xr

from satterc import units
from satterc.config import Config
from satterc.dag._utils import declare_units


def _da(values, unit=None):
    """Build a (time, pixel) DataArray, optionally with a units attribute."""
    arr = np.asarray(values, dtype=float)
    time = xr.date_range("2020-01-01", periods=arr.shape[0], freq="7D")
    da = xr.DataArray(
        arr,
        dims=("time", "pixel"),
        coords={"time": time, "pixel": np.arange(arr.shape[1])},
    )
    if unit is not None:
        da.attrs["units"] = unit
    return da


# ---------------------------------------------------------------------------
# Mode resolution
# ---------------------------------------------------------------------------


class TestMode:
    def test_default_mode_is_warn(self):
        with units.mode(None):
            assert units.get_mode() == "warn"

    def test_set_mode(self):
        with units.mode("strict"):
            assert units.get_mode() == "strict"

    def test_env_overrides_process_mode(self, monkeypatch):
        with units.mode("off"):
            monkeypatch.setenv(units.MODE_ENV_VAR, "strict")
            assert units.get_mode() == "strict"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid units mode"):
            units.set_mode("bogus")


# ---------------------------------------------------------------------------
# Unit resolution helpers
# ---------------------------------------------------------------------------


class TestResolution:
    def test_resolve_input_from_declaration(self):
        assert units.resolve_input_unit("x", {"x": "kg"}) == "kg"

    def test_resolve_input_no_declaration_is_none(self):
        # No central registry: an undeclared input resolves to None.
        assert units.resolve_input_unit("vpd_pa_weekly", None) is None
        assert units.resolve_input_unit("y", {"x": "kg"}) is None

    def test_resolve_output_bare_string(self):
        assert units.resolve_output_unit("anything", "g m-2 d-1") == "g m-2 d-1"

    def test_resolve_output_dict(self):
        assert units.resolve_output_unit("gpp", {"gpp": "g m-2 d-1"}) == "g m-2 d-1"

    def test_resolve_output_no_declaration_is_none(self):
        assert units.resolve_output_unit("gpp_weekly", None) is None
        assert units.resolve_output_unit("other", {"gpp": "g m-2 d-1"}) is None


# ---------------------------------------------------------------------------
# check_units: conversion, round-trip, incompatibility, missing
# ---------------------------------------------------------------------------


class TestCheckUnits:
    def test_round_trip_preserves_coords_and_stamps_declared(self):
        da = _da([[1.0, 2.0], [3.0, 4.0]], unit="Pa")
        out = units.check_units(da, "Pa", "vpd", "strict")
        assert out.attrs["units"] == "Pa"
        xr.testing.assert_equal(out["time"], da["time"])
        xr.testing.assert_equal(out["pixel"], da["pixel"])
        np.testing.assert_allclose(out.values, da.values)

    def test_conversion_hpa_to_pa(self):
        da = _da([[10.0, 20.0]], unit="hPa")
        out = units.check_units(da, "Pa", "vpd", "strict")
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[1000.0, 2000.0]])

    def test_incompatible_raises_dimensionality_error(self):
        da = _da([[1.0, 2.0]], unit="degC")
        with pytest.raises(pint.DimensionalityError):
            units.check_units(da, "kg", "x", "strict")

    def test_affine_kelvin_to_celsius(self):
        da = _da([[300.0, 273.15]], unit="K")
        out = units.check_units(da, "degC", "temperature", "strict")
        np.testing.assert_allclose(out.values, [[26.85, 0.0]])

    def test_missing_units_strict_raises(self):
        da = _da([[1.0, 2.0]])
        with pytest.raises(ValueError, match="no 'units' attribute"):
            units.check_units(da, "Pa", "vpd", "strict")

    def test_missing_units_warn_warns_and_passes_through(self):
        da = _da([[1.0, 2.0]])
        with pytest.warns(UserWarning, match="unvalidated"):
            out = units.check_units(da, "Pa", "vpd", "warn")
        assert "units" not in out.attrs
        np.testing.assert_array_equal(out.values, da.values)


# ---------------------------------------------------------------------------
# CF / UDUNITS string parsing
# ---------------------------------------------------------------------------


class TestCFParsing:
    @pytest.mark.parametrize(
        "unit",
        ["umol m-2 s-1", "g m-2 d-1", "t ha-1", "mm d-1", "ppm", "degC"],
    )
    def test_cf_unit_strings_parse_and_convert(self, unit):
        da = _da([[1.0, 2.0]], unit=unit)
        out = units.check_units(da, unit, "x", "strict")
        assert out.attrs["units"] == unit
        np.testing.assert_allclose(out.values, da.values)


# ---------------------------------------------------------------------------
# units_from_signature: reading declarations off a node's annotations
# ---------------------------------------------------------------------------


class TestUnitsFromSignature:
    def test_extracts_inputs_and_typeddict_outputs(self):
        class Out(TypedDict):
            gpp: Annotated[xr.DataArray, "g m-2 d-1"]
            lue: Annotated[xr.DataArray, "g MJ-1"]

        def node(
            temp: Annotated[xr.DataArray, "degC"],
            plain: xr.DataArray,
            scalar: int = 3,
        ) -> Out: ...

        inputs, outputs = units.units_from_signature(node)
        # Only Annotated params with a string unit contribute; others are ignored.
        assert inputs == {"temp": "degC"}
        assert outputs == {"gpp": "g m-2 d-1", "lue": "g MJ-1"}

    def test_bare_annotated_return(self):
        def node(x: Annotated[xr.DataArray, "1"]) -> Annotated[xr.DataArray, "1"]: ...

        inputs, outputs = units.units_from_signature(node)
        assert inputs == {"x": "1"}
        assert outputs == "1"

    def test_no_annotations(self):
        def node(x: xr.DataArray) -> xr.DataArray: ...

        inputs, outputs = units.units_from_signature(node)
        assert inputs == {}
        assert outputs is None


# ---------------------------------------------------------------------------
# declare_units: input validation + output stamping + edge propagation
# ---------------------------------------------------------------------------


class TestDeclareUnits:
    def test_input_converted_before_reaching_body(self):
        class Out(TypedDict):
            out: Annotated[xr.DataArray, "Pa"]

        @declare_units
        def f(vpd: Annotated[xr.DataArray, "Pa"]) -> Out:
            # vpd reaches the body as a DataArray already converted to declared units
            return {"out": vpd}

        with units.mode("warn"):
            result = f(vpd=_da([[10.0, 20.0]], unit="hPa"))
        np.testing.assert_allclose(result["out"].values, [[1000.0, 2000.0]])

    def test_output_stamped_with_declared_unit_not_inherited(self):
        class Out(TypedDict):
            gpp_weekly: Annotated[xr.DataArray, "g m-2 d-1"]

        @declare_units
        def f(temperature_celcius_weekly: Annotated[xr.DataArray, "degC"]) -> Out:
            return {"gpp_weekly": temperature_celcius_weekly * 2}

        out = f(temperature_celcius_weekly=_da([[1.0, 2.0]], unit="degC"))
        assert out["gpp_weekly"].attrs["units"] == "g m-2 d-1"

    def test_edge_propagation_two_node_chain(self):
        """An internal edge is validated using the upstream node's stamped output."""

        class ProducerOut(TypedDict):
            gpp_weekly: Annotated[xr.DataArray, "g m-2 d-1"]

        class ConsumerOut(TypedDict):
            npp: Annotated[xr.DataArray, "g m-2 d-1"]

        @declare_units
        def producer(
            temperature_celcius_weekly: Annotated[xr.DataArray, "degC"],
        ) -> ProducerOut:
            return {"gpp_weekly": temperature_celcius_weekly}

        @declare_units
        def consumer(gpp_weekly: Annotated[xr.DataArray, "g m-2 d-1"]) -> ConsumerOut:
            return {"npp": gpp_weekly}

        with units.mode("strict"):
            produced = producer(
                temperature_celcius_weekly=_da([[1.0, 2.0]], unit="degC")
            )
            # No exception: the stamped 'g m-2 d-1' output validates as consumer input.
            consumed = consumer(gpp_weekly=produced["gpp_weekly"])
        np.testing.assert_allclose(consumed["npp"].values, [[1.0, 2.0]])

    def test_off_mode_skips_validation(self):
        class Out(TypedDict):
            out: Annotated[xr.DataArray, "Pa"]

        @declare_units
        def f(vpd: Annotated[xr.DataArray, "Pa"]) -> Out:
            return {"out": vpd}

        with units.mode("off"), warnings.catch_warnings():
            # Input has no units; strict would raise, but off skips validation and
            # leaves the data unconverted.
            warnings.simplefilter("error")
            result = f(vpd=_da([[10.0, 20.0]]))
        np.testing.assert_allclose(result["out"].values, [[10.0, 20.0]])

    def test_off_mode_still_stamps_output(self):
        class Out(TypedDict):
            gpp_weekly: Annotated[xr.DataArray, "g m-2 d-1"]

        @declare_units
        def f(temperature_celcius_weekly: Annotated[xr.DataArray, "degC"]) -> Out:
            return {"gpp_weekly": temperature_celcius_weekly}

        with units.mode("off"):
            out = f(temperature_celcius_weekly=_da([[1.0, 2.0]], unit="degC"))
        # Stamping applies regardless of mode (it is labelling, not validation):
        # the inherited 'degC' must not leak onto the output.
        assert out["gpp_weekly"].attrs["units"] == "g m-2 d-1"

    def test_bare_annotated_single_output_stamped(self):
        @declare_units
        def f(x: Annotated[xr.DataArray, "degC"]) -> Annotated[xr.DataArray, "1"]:
            return x

        out = f(x=_da([[1.0, 2.0]], unit="degC"))
        assert out.attrs["units"] == "1"


# ---------------------------------------------------------------------------
# Config [units] section
# ---------------------------------------------------------------------------


class TestConfigUnits:
    def test_parse_units_mode(self):
        parsed = Config.loads('[units]\nmode = "strict"\n').parse()
        assert parsed.units_mode == "strict"

    def test_no_units_section_is_none(self):
        parsed = Config.loads("").parse()
        assert parsed.units_mode is None

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="must be one of"):
            Config.loads('[units]\nmode = "bogus"\n').parse()
