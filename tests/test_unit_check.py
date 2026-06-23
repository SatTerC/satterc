"""Tests for the build-time (static) DAG unit-consistency check."""

import sys
import types
import warnings
from typing import Annotated, TypedDict

import pytest
import xarray as xr
from hamilton import driver
from hamilton.function_modifiers import extract_fields
from hamilton.settings import ENABLE_POWER_USER_MODE

from satterc import units
from satterc.config import DeriveSpec, ResampleSpec
from satterc.dag._utils import declare_units
from satterc.dag.driver import build_driver
from satterc.dag.unit_check import check_dag_units


def _da():
    return xr.DataArray([1.0])


@pytest.fixture
def register():
    """Build Hamilton-scannable modules from functions and clean up afterwards.

    Hamilton only picks up a module's functions when they live in ``sys.modules``
    and their ``__module__`` matches the module name (mirrors ``derive.py``).
    """
    names: list[str] = []

    def _make(name: str, *funcs) -> types.ModuleType:
        mod = types.ModuleType(name)
        for fn in funcs:
            fn.__module__ = name
            setattr(mod, fn.__name__, fn)
        sys.modules[name] = mod
        names.append(name)
        return mod

    yield _make

    for name in names:
        sys.modules.pop(name, None)


def _build(*mods) -> driver.Driver:
    return (
        driver.Builder()
        .with_modules(*mods)
        .with_config({ENABLE_POWER_USER_MODE: True})
        .build()
    )


def _producer(unit: str):
    """A node producing ``gpp_weekly`` with the given declared output unit."""

    class Out(TypedDict):
        gpp_weekly: Annotated[xr.DataArray, unit]

    @extract_fields()
    @declare_units
    def producer() -> Out:  # type: ignore[valid-type]
        return {"gpp_weekly": _da()}

    return producer


def _consumer(unit: str, name: str = "consumer", in_name: str = "gpp_weekly"):
    """A node consuming ``in_name`` with the given declared input unit.

    Built via ``exec`` so the consumed parameter (= the upstream node name) and
    the function name are both dynamic. The output node is named ``f"{name}_out"``
    so multiple consumers can coexist in one graph without a name collision.
    """
    src = (
        "from typing import Annotated, TypedDict\n"
        "import xarray as xr\n"
        "from hamilton.function_modifiers import extract_fields\n"
        "from satterc.dag._utils import declare_units\n"
        f"class _Out(TypedDict):\n"
        f"    {name}_out: Annotated[xr.DataArray, 't ha-1']\n"
        "@extract_fields()\n"
        "@declare_units\n"
        f"def {name}({in_name}: Annotated[xr.DataArray, {unit!r}]) -> _Out:\n"
        f"    return {{{name + '_out'!r}: {in_name}}}\n"
    )
    ns: dict = {}
    exec(src, ns)
    return ns[name]


# ---------------------------------------------------------------------------
# Dimensional incompatibility (always a finding)
# ---------------------------------------------------------------------------


class TestDimensionalMismatch:
    def _dr(self, register):
        # producer emits 'g m-2 d-1'; consumer declares 'kg' (incompatible).
        prod = register("uc_prod", _producer("g m-2 d-1"))
        cons = register("uc_cons", _consumer("kg"))
        return _build(prod, cons)

    def test_strict_raises(self, register):
        dr = self._dr(register)
        with pytest.raises(ValueError, match="dimensionally incompatible"):
            check_dag_units(dr, mode="strict")

    def test_warn_warns(self, register):
        dr = self._dr(register)
        with pytest.warns(UserWarning, match="dimensionally incompatible"):
            check_dag_units(dr, mode="warn")

    def test_off_is_silent(self, register):
        dr = self._dr(register)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_dag_units(dr, mode="off")  # returns without raising/warning

    def test_message_names_node_and_units(self, register):
        dr = self._dr(register)
        with pytest.raises(ValueError, match="gpp_weekly") as exc:
            check_dag_units(dr, mode="strict")
        msg = str(exc.value)
        assert "'g m-2 d-1'" in msg
        assert "'kg'" in msg


# ---------------------------------------------------------------------------
# Exact-string mismatch (only when exact is enabled)
# ---------------------------------------------------------------------------


class TestExactMatch:
    def _dr(self, register):
        # Dimensionally compatible but not identical: 'Pa' produced, 'hPa' consumed.
        prod = register("ue_prod", _producer("Pa"))
        cons = register("ue_cons", _consumer("hPa"))
        return _build(prod, cons)

    def test_compatible_passes_when_exact_off(self, register):
        dr = self._dr(register)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_dag_units(dr, mode="strict", exact=False)

    def test_compatible_flagged_when_exact_on(self, register):
        dr = self._dr(register)
        with pytest.raises(ValueError, match="exact match required"):
            check_dag_units(dr, mode="strict", exact=True)


# ---------------------------------------------------------------------------
# Shared external input consumed with conflicting units (no producer)
# ---------------------------------------------------------------------------


class TestSharedInputConflict:
    def test_conflicting_consumers_flagged(self, register):
        a = register("us_a", _consumer("Pa", name="consumer_a"))
        b = register("us_b", _consumer("kg", name="consumer_b"))
        dr = _build(a, b)
        with pytest.raises(ValueError, match="dimensionally incompatible"):
            check_dag_units(dr, mode="strict")


# ---------------------------------------------------------------------------
# Consistent declarations pass (synthetic + real models)
# ---------------------------------------------------------------------------


class TestConsistent:
    def test_matching_units_pass_even_under_exact(self, register):
        prod = register("uk_prod", _producer("Pa"))
        cons = register("uk_cons", _consumer("Pa"))
        dr = _build(prod, cons)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_dag_units(dr, mode="strict", exact=True)

    def test_real_pmodel_sgam_pipeline_clean(self):
        # pmodel -> sgam declares gpp/lue/iwue identically; clean even under exact.
        dr = build_driver(["models.pmodel", "models.sgam"], {})
        check_dag_units(dr, mode="strict", exact=True)


# ---------------------------------------------------------------------------
# build_driver invokes the check (gated by the global mode)
# ---------------------------------------------------------------------------


class TestBuildDriverIntegration:
    def test_build_driver_runs_check_in_strict(self, register):
        register("ub_prod", _producer("g m-2 d-1"))
        register("ub_cons", _consumer("kg"))
        with (
            units.mode("strict"),
            pytest.raises(ValueError, match="dimensionally incompatible"),
        ):
            build_driver(["ub_prod", "ub_cons"], {})

    def test_build_driver_skips_in_off(self, register):
        register("ub2_prod", _producer("g m-2 d-1"))
        register("ub2_cons", _consumer("kg"))
        with units.mode("off"):
            build_driver(["ub2_prod", "ub2_cons"], {})  # no raise despite mismatch


# ---------------------------------------------------------------------------
# Resample propagation: a resampled var inherits its source's unit
# ---------------------------------------------------------------------------


class TestResamplePropagation:
    """The real ``pmodel`` emits ``gpp_weekly`` ('g m-2 d-1'); resampling to
    ``gpp_monthly`` should propagate that unit so a downstream consumer is
    checked against it."""

    def _build(self, register, consumer_unit):
        register("rs_cons", _consumer(consumer_unit, in_name="gpp_monthly"))
        specs = [
            ResampleSpec(vars=["gpp"], source_freq="weekly", target_freq="monthly")
        ]
        return build_driver(
            ["models.pmodel", "resample", "rs_cons"], {"resample_specs": specs}
        )

    def test_incompatible_consumer_of_resampled_var_raises(self, register):
        with (
            units.mode("strict"),
            pytest.raises(ValueError, match="gpp_monthly"),
        ):
            self._build(register, "kg")

    def test_compatible_consumer_of_resampled_var_passes(self, register):
        with units.mode("strict"):
            self._build(register, "g m-2 d-1")  # matches propagated unit


# ---------------------------------------------------------------------------
# Derive: a declared `units=` makes the node a checkable producer
# ---------------------------------------------------------------------------


class TestDerivePropagation:
    def _build(self, register, consumer_unit):
        register("dv_cons", _consumer(consumer_unit, in_name="flux"))
        specs = [
            DeriveSpec(
                output="flux",
                inputs=["a", "b"],
                expression="a + b",
                import_path=None,
                function=None,
                units="g m-2 d-1",
            )
        ]
        return build_driver(["derive", "dv_cons"], {"derive_specs": specs})

    def test_incompatible_consumer_of_derived_var_raises(self, register):
        with units.mode("strict"), pytest.raises(ValueError, match="flux"):
            self._build(register, "kg")

    def test_compatible_consumer_of_derived_var_passes(self, register):
        with units.mode("strict"):
            self._build(register, "g m-2 d-1")
