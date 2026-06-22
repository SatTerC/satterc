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
# Declared-unit validation (fail fast at decoration time)
# ---------------------------------------------------------------------------


class TestAssertValidUnit:
    @pytest.mark.parametrize(
        "unit", ["degC", "Pa", "1", "umol m-2 s-1", "g m-2 d-1", "t ha-1 month-1"]
    )
    def test_valid_units_pass(self, unit):
        units.assert_valid_unit(unit, "ctx")  # no raise

    @pytest.mark.parametrize("unit", ["degrees_C", "not_a_unit", "kg/"])
    def test_invalid_units_raise_with_context(self, unit):
        with pytest.raises(ValueError, match="not a recognised"):
            units.assert_valid_unit(unit, "myctx input 'x'")


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

    def test_bad_input_unit_rejected_at_decoration(self):
        with pytest.raises(ValueError, match="not a recognised"):

            @declare_units
            def f(
                x: Annotated[xr.DataArray, "not_a_unit"],
            ) -> Annotated[xr.DataArray, "1"]:
                return x

    def test_bad_output_unit_rejected_at_decoration(self):
        class Out(TypedDict):
            y: Annotated[xr.DataArray, "bogus_unit"]

        with pytest.raises(ValueError, match="not a recognised"):

            @declare_units
            def f(x: Annotated[xr.DataArray, "degC"]) -> Out:
                return {"y": x}


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


# ---------------------------------------------------------------------------
# End-to-end: units validation/conversion through a real model node
# ---------------------------------------------------------------------------


class TestModelNodeEndToEnd:
    """Units behaviour through the real ``pmodel`` node, in ``strict`` mode.

    Exercises the actual ``@declare_units`` + ``@xarray_io`` composition on the
    public node (not a synthetic decorator): every declared input is validated,
    a convertible wrong-unit input is converted, an incompatible one raises, and
    the output is stamped — all through the pyrealm-backed implementation.
    """

    @staticmethod
    def _pmodel_inputs(**overrides):
        # (value, unit) per declared input; overrides replace specific entries.
        spec = {
            "temperature_celcius_weekly": (15.0, "degC"),
            "vpd_pa_weekly": (1000.0, "Pa"),
            "co2_ppm_weekly": (400.0, "ppm"),
            "pressure_pa_weekly": (101325.0, "Pa"),
            "fapar_weekly": (0.5, "1"),
            "ppfd_umol_m2_s1_weekly": (500.0, "umol m-2 s-1"),
            "mean_growth_temperature_weekly": (15.0, "degC"),
            "aridity_index_weekly": (0.5, "1"),
            "soil_moisture_weekly": (100.0, "mm"),
        }
        spec.update(overrides)
        return {
            name: _da([[value]] * 4, unit=unit) for name, (value, unit) in spec.items()
        }

    def test_convertible_input_accepted_and_output_stamped(self):
        from satterc.dag.pmodel import pmodel

        # Pressure supplied in hPa where Pa is declared: must convert, not fail.
        inputs = self._pmodel_inputs(pressure_pa_weekly=(1013.25, "hPa"))
        with units.mode("strict"):
            out = pmodel(**inputs)
        assert out["gpp_weekly"].attrs["units"] == "g m-2 d-1"

    def test_incompatible_input_raises(self):
        import pint

        from satterc.dag.pmodel import pmodel

        # VPD supplied in kg where Pa is declared: dimensionally incompatible.
        inputs = self._pmodel_inputs(vpd_pa_weekly=(1000.0, "kg"))
        with units.mode("strict"), pytest.raises(pint.DimensionalityError):
            pmodel(**inputs)

    def test_missing_units_strict_raises(self):
        from satterc.dag.pmodel import pmodel

        inputs = self._pmodel_inputs()
        inputs["co2_ppm_weekly"].attrs.pop("units")
        with units.mode("strict"), pytest.raises(ValueError, match="no 'units'"):
            pmodel(**inputs)
