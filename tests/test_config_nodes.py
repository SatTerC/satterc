"""`[[node]]` entries must lower into a buildable graph with their contracts.

`tests/test_config.toml` declares no `[[node]]` or `[[resample]]` entries, so the
fast suite never exercised conduit's node lowering at all — the only test that
did was `test_contracts.py`, which generates two years of data and runs four
models. That left the cheapest, most load-bearing part of every satterc config
covered only by the slowest test in the suite.

These tests build the same *shapes* of node that `recipes/config.toml` relies on
— an expression carrying units and a frequency, a climatological one that reduces
the time axis away and declares no frequency, and one node consuming another —
and assert the declared contract survives onto the built function. They need no
data and no model, so they run in milliseconds.
"""

import textwrap
import typing
from pathlib import Path

import pytest
from conduit import build_driver, load_config

CONFIG = """
[inputs.daily]
path = "{path}"
suffix = ""
vars = ["rain", "demand"]

# Units and a frequency, the ordinary case.
[[node]]
name = "balance"
inputs = ["rain", "demand"]
expression = "rain - demand"
units = "mm"
freq = "D"

# Consumes another node rather than an input.
[[node]]
name = "balance_is_positive"
inputs = ["balance"]
expression = "balance > 0"
units = "1"
freq = "D"

# Climatological: reduces the time axis away, so it declares no frequency. This
# is the shape of `aridity_index` in recipes/config.toml.
[[node]]
name = "dryness"
inputs = ["rain", "demand"]
expression = 'demand.sum("time") / rain.sum("time")'
units = "1"
"""


@pytest.fixture(scope="module")
def node_config(tmp_path_factory):
    """A config whose only content is input specs and `[[node]]` entries."""
    path = tmp_path_factory.mktemp("nodes") / "config.toml"
    path.write_text(textwrap.dedent(CONFIG).format(path=path.parent / "daily.nc"))
    return load_config(path)


class TestNodeSpecParsing:
    """Parsing is pure config handling and works on every supported Python."""

    def test_parses_into_node_specs(self, node_config):
        assert {spec.name for spec in node_config.node_specs} == {
            "balance",
            "balance_is_positive",
            "dryness",
        }

    def test_climatological_node_declares_no_frequency(self, node_config):
        """A node that reduces the time axis away must not claim a frequency."""
        spec = next(s for s in node_config.node_specs if s.name == "dryness")
        assert spec.freq is None
        assert spec.units == "1"


class TestNodeLowering:
    """Building the graph, as distinct from parsing the config that describes it."""

    def test_driver_builds(self, node_config):
        """The regression guard: this is what fails when lowering breaks."""
        driver = build_driver(
            node_config.modules,
            node_config.driver_config,
            node_specs=node_config.node_specs,
        )
        assert {"balance", "balance_is_positive", "dryness"} <= set(driver.graph.nodes)

    @pytest.mark.parametrize(
        ("name", "unit"),
        [("balance", "mm"), ("balance_is_positive", "1"), ("dryness", "1")],
    )
    def test_declared_units_reach_the_built_function(self, node_config, name, unit):
        """The contract must survive `exec` *and* the decorators wrapping it.

        conduit injects the return annotation after building the function body,
        and the `declare_*` decorators then wrap it. Reading the hint back through
        `get_type_hints` is what catches the injection being lost in between —
        which is exactly what PEP 749 caused on Python 3.14, by swapping
        `__annotations__` for `__annotate__` in `functools.WRAPPER_ASSIGNMENTS`
        so that `functools.wraps` no longer carried the injected hint across.
        Fixed upstream in xarray-annotated 0.4.1; this is the regression guard.
        """
        driver = build_driver(
            node_config.modules,
            node_config.driver_config,
            node_specs=node_config.node_specs,
        )
        hints = typing.get_type_hints(
            driver.graph.nodes[name].callable, include_extras=True
        )
        assert "return" in hints, f"{name} lost its injected return annotation"
        assert unit in str(hints["return"])


def test_the_example_config_still_uses_nodes():
    """Guards the premise: if recipes/config.toml ever drops its `[[node]]`
    entries, the coverage above stops mirroring anything real."""
    config = Path(__file__).parent.parent / "recipes" / "config.toml"
    assert "[[node]]" in config.read_text()
