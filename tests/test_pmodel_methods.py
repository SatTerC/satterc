"""Every pyrealm P-Model method gets exactly the extra inputs it declares.

`satterc.models.pmodel` selects an ``<axis>_inputs`` variant per configured method,
from a hand-written grouping of methods by what they need. pyrealm declares the same
thing on each method class (``requires``, or ``required_env_variables`` for the
Arrhenius classes). These tests hold the grouping to those declarations, so a method
added or changed upstream fails here instead of in a pipeline run.
"""

import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyrealm.pmodel
import pytest
import xarray as xr
from conduit import build_driver
from pyrealm.pmodel.arrhenius import ARRHENIUS_METHOD_REGISTRY
from pyrealm.pmodel.jmax_limitation import JMAX_LIMITATION_CLASS_REGISTRY
from pyrealm.pmodel.optimal_chi import OPTIMAL_CHI_CLASS_REGISTRY
from pyrealm.pmodel.quantum_yield import QUANTUM_YIELD_CLASS_REGISTRY
from xarray_annotated import unwrap_annotated

from satterc.models.pmodel import _OUTPUT_ATTRIBUTES, pmodel_config

MODULE = "satterc.models.pmodel"

#: config key -> (pyrealm registry, attribute naming its requirements, node)
AXES = {
    "method_optchi": (OPTIMAL_CHI_CLASS_REGISTRY, "requires", "optchi_inputs"),
    "method_kphio": (QUANTUM_YIELD_CLASS_REGISTRY, "requires", "kphio_inputs"),
    "method_arrhenius": (
        ARRHENIUS_METHOD_REGISTRY,
        "required_env_variables",
        "arrhenius_inputs",
    ),
}

CASES = [
    pytest.param(key, method, id=f"{key}={method}")
    for key, (registry, _, _) in AXES.items()
    for method in registry
]

CORE_INPUTS = {
    "temperature_weekly",
    "vpd_weekly",
    "co2_weekly",
    "pressure_weekly",
    "fapar_weekly",
    "ppfd_weekly",
}

N_WEEKS = 8
N_PIXELS = 2
WEEKLY_DATES = pd.date_range("2020-01-01", periods=N_WEEKS, freq="7D")
PIXELS = np.arange(N_PIXELS)


def _weekly(value: float) -> xr.DataArray:
    return xr.DataArray(
        np.full((N_WEEKS, N_PIXELS), value),
        dims=["time", "pixel"],
        coords={"time": WEEKLY_DATES, "pixel": PIXELS},
    )


def _static(value: float) -> xr.DataArray:
    return xr.DataArray(
        np.full(N_PIXELS, value), dims=["pixel"], coords={"pixel": PIXELS}
    )


#: A plausible value for every input any configuration can ask for.
INPUTS = {
    "temperature_weekly": _weekly(15.0),
    "vpd_weekly": _weekly(1000.0),
    "co2_weekly": _weekly(400.0),
    "pressure_weekly": _weekly(101325.0),
    "fapar_weekly": _weekly(0.5),
    "ppfd_weekly": _weekly(500.0),
    "volumetric_water_content_weekly": _weekly(0.3),
    "rootzonestress_weekly": _weekly(0.7),
    "aridity_index": _static(1.0),
    # Feeds the `mean_growth_temperature` node.
    "temperature_daily": xr.DataArray(
        np.full((N_WEEKS * 7, N_PIXELS), 15.0),
        dims=["time", "pixel"],
        coords={
            "time": pd.date_range("2020-01-01", periods=N_WEEKS * 7, freq="D"),
            "pixel": PIXELS,
        },
    ),
}


def _required_inputs(dr) -> set[str]:
    """The data inputs the run needs, leaving out unset settings.

    A setting the config leaves unset is an external input too, defaulted by
    `pmodel_config`, so filter on type as `satterc.scaffold.config_gen` does.
    """
    return {
        v.name
        for v in dr.what_is_upstream_of("gpp_flux_weekly")
        if v.is_external_input and unwrap_annotated(v.type) is xr.DataArray
    }


def _requires(key: str, method: str) -> list[str]:
    registry, attribute, _ = AXES[key]
    # No getattr default: if pyrealm renames the attribute, fail rather than
    # read every method as needing nothing.
    return getattr(registry[method], attribute)


@pytest.mark.parametrize(("key", "method"), CASES)
def test_axis_node_passes_what_pyrealm_requires(key, method):
    """The selected variant hands pyrealm exactly the inputs the method requires.

    Runs the whole node too, so pyrealm itself confirms it got what it needs.
    """
    _, _, node = AXES[key]
    dr = build_driver([MODULE], {key: method})
    inputs = {name: INPUTS[name] for name in _required_inputs(dr)}
    result = dr.execute(  # type: ignore[reportArgumentType]
        [node, "gpp_flux_weekly"], inputs=inputs
    )

    assert set(result[node]) == set(_requires(key, method))
    assert np.isfinite(result["gpp_flux_weekly"].values).all()


@pytest.mark.parametrize(("key", "method"), CASES)
def test_only_required_inputs_are_asked_for(key, method):
    """No input outside the core drivers is required unless a method uses it."""
    dr = build_driver([MODULE], {key: method})
    extra = _required_inputs(dr) - CORE_INPUTS
    if _requires(key, method):
        assert extra
    else:
        assert extra == set()


def test_defaults_need_no_extra_inputs():
    """An unset method key selects the empty variant, so the default must need none.

    `config.when` cannot see `pmodel_config`'s defaults, only the keys a config
    actually sets.
    """
    defaults = pmodel_config()
    for key in AXES:
        assert _requires(key, defaults[key]) == []

    dr = build_driver([MODULE], {})
    assert _required_inputs(dr) == CORE_INPUTS


@pytest.mark.parametrize(
    ("config", "derived"),
    [
        ({}, False),
        ({"method_kphio": "sandoval"}, True),
        ({"method_arrhenius": "kattge_knorr"}, True),
        ({"method_kphio": "fixed", "method_arrhenius": "simple"}, False),
    ],
)
def test_mean_growth_temperature_exists_only_for_its_consumers(config, derived):
    """Without a consumer the node, and the daily input behind it, leave the DAG."""
    names = {v.name for v in build_driver([MODULE], config).list_available_variables()}
    assert ("mean_growth_temperature" in names) == derived
    assert ("temperature_daily" in names) == derived


def test_arrhenius_method_changes_only_the_standard_temperature_rates():
    """``method_arrhenius`` reaches `vcmax25_weekly` and `jmax25_weekly`, nothing else."""

    def run(method):
        dr = build_driver([MODULE], {"method_arrhenius": method})
        inputs = {name: INPUTS[name] for name in _required_inputs(dr)}
        return dr.execute(  # type: ignore[reportArgumentType]
            list(_OUTPUT_ATTRIBUTES), inputs=inputs
        )

    simple, kattge_knorr = run("simple"), run("kattge_knorr")
    scaled = {"vcmax25_weekly", "jmax25_weekly"}
    for name in _OUTPUT_ATTRIBUTES:
        same = np.allclose(simple[name].values, kattge_knorr[name].values)
        assert same == (name not in scaled), name


@pytest.mark.parametrize("method_kphio", ["temperature", "fixed", "sandoval"])
def test_reference_kphio_reaches_pyrealm(method_kphio):
    """The wrapper matches a direct pyrealm run given the same reference value."""
    reference_kphio = 0.05
    config = {"method_kphio": method_kphio, "reference_kphio": reference_kphio}
    dr = build_driver([MODULE], config)
    inputs = {name: INPUTS[name] for name in _required_inputs(dr)}
    wrapped = dr.execute(  # type: ignore[reportArgumentType]
        ["gpp_flux_weekly"], inputs=inputs
    )["gpp_flux_weekly"]

    extras: dict[str, Any] = (
        {
            "aridity_index": np.full((N_WEEKS, N_PIXELS), 1.0),
            "mean_growth_temperature": np.full((N_WEEKS, N_PIXELS), 15.0),
        }
        if method_kphio == "sandoval"
        else {}
    )
    env = pyrealm.pmodel.PModelEnvironment(
        tc=INPUTS["temperature_weekly"].values,
        vpd=INPUTS["vpd_weekly"].values,
        co2=INPUTS["co2_weekly"].values,
        patm=INPUTS["pressure_weekly"].values,
        fapar=INPUTS["fapar_weekly"].values,
        ppfd=INPUTS["ppfd_weekly"].values,
        **extras,
    )
    direct = pyrealm.pmodel.PModel(
        env=env, method_kphio=method_kphio, reference_kphio=reference_kphio
    )
    np.testing.assert_allclose(wrapped.values, direct.gpp)

    default = pyrealm.pmodel.PModel(env=env, method_kphio=method_kphio)
    assert not np.allclose(direct.gpp, default.gpp)


@pytest.mark.parametrize(
    ("kwargs", "env_extra", "expected"),
    [
        (
            'method_arrhenius="kattge_knorr"',
            "mean_growth_temperature=one * 15.0",
            "strongly recommend",
        ),
        (
            'method_optchi="prentice14_rootzonestress"',
            "rootzonestress=one * 0.7",
            "ExperimentalFeatureWarning",
        ),
    ],
)
def test_pyrealm_warnings_survive_importing_satterc(kwargs, env_extra, expected):
    """Run outside pytest, which replaces the filters `satterc` installs on import."""
    script = f"""
import numpy as np
import satterc
from pyrealm.pmodel import PModel, PModelEnvironment
one = np.ones(2)
env = PModelEnvironment(
    tc=one * 15, vpd=one * 1000, co2=one * 400, patm=one * 101325, {env_extra}
)
PModel(env, {kwargs})
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert expected in result.stderr
    assert "Pyrealm 2.0.0" not in result.stderr


@pytest.mark.parametrize("method", list(JMAX_LIMITATION_CLASS_REGISTRY))
def test_jmax_limitation_needs_no_extra_inputs(method):
    """`method_jmaxlim` has no `<axis>_inputs` node, which holds while no method
    declares requirements."""
    cls = JMAX_LIMITATION_CLASS_REGISTRY[method]
    assert not getattr(cls, "requires", None)
    assert not getattr(cls, "required_env_variables", None)
