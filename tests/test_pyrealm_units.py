"""Regression tests pinning pyrealm's unit conventions to satterc's annotations.

`tests/test_pmodel_seam.py` and `tests/test_splash_seam.py` already check that
the wrapped models agree with pyrealm called directly. They cannot catch the
failure this file exists for. Both sides call the same pyrealm, so if upstream
changes what a number *means*, the two move together and the comparison still
passes, while every `@declare_units` annotation downstream becomes confidently
wrong. `pyrealm` is pinned as ``>=2.0.0,<3``: the cap keeps a major release from
arriving unreviewed on a plain ``uv lock --upgrade``, but a 2.x release is still
free to change a convention, and nothing but this file would notice.

This is not hypothetical. When these tests were written all three P-Model output
annotations were wrong. GPP was labelled ``g m-2 d-1`` for what pyrealm reports
as ``ug m-2 s-1``, a factor of 0.0864. LUE was labelled per-megajoule for a
per-mole quantity. iWUE was labelled ``Pa`` for a ``umol mol-1`` mixing ratio.

Four layers, each failing differently on purpose:

1. `TestUpstreamUnitMetadata` compares satterc's declared units against pyrealm's
   own machine-readable tables. The cheapest check and the most direct. It would
   have caught the bug above for free.
2. `TestGoldenValues` pins absolute magnitudes in ``data/pyrealm_golden.json``,
   catching a convention change even when upstream forgets to relabel it.
3. `TestPModelInvariants` and `TestSplashInvariants` assert physical relations
   that hold in one unit system and not another. These say *what* moved when a
   golden value fails.
4. `TestAnnotationsMatchGolden` requires the annotations and the golden file to
   agree, so the two halves of layer 2 cannot drift apart independently.
"""

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyrealm
import pyrealm.core.bounds
import pyrealm.core.calendar
import pyrealm.pmodel
import pyrealm.splash.splash
import pytest
from pyrealm.constants import CoreConst
from pyrealm.splash.evap import DailyEvapFluxes
from pyrealm.splash.solar import DailySolarFluxes
from pyrealm_cases import (
    LATITUDE,
    MAX_SOIL_MOISTURE,
    N_PIXELS,
    PMODEL_METHODS,
    declared_input_units,
    declared_units,
    pmodel_inputs,
    splash_inputs,
    summarize,
)

from satterc.models.pmodel import _OUTPUT_ATTRIBUTES, PModelOut, _pmodel
from satterc.models.splash import _FLUX_ATTRIBUTES, SplashOut, _splash

GOLDEN_PATH = Path(__file__).parent / "data" / "pyrealm_golden.json"

REGENERATE_HINT = (
    "If pyrealm changed a unit convention, fix satterc's annotation. Do NOT "
    "just regenerate the golden file. Regenerate with "
    "`just regen-pyrealm-golden` only once the change is understood."
)

# pyrealm spells its units for humans; satterc declares them in CF/UDUNITS form.
# Normalising handles spelling only. It never reconciles a real difference.
# Substitutions are substring-wise and so must not touch exponents: "-" alone
# means dimensionless, but the "-" in "m-2" is a minus sign, which is why the
# bare-dimensionless case is handled as a whole-string match instead.
_SPELLING = {
    "°C": "degC",
    "°": "degrees",
    "µ": "u",
    " C ": " ",  # "g C mol-1" names the substance; CF has no slot for it
    "mm day-1": "mm d-1",
}
_DIMENSIONLESS = {"-", "unitless", ""}


def normalise(unit: str) -> str:
    """Render a pyrealm unit string in the spelling satterc's annotations use."""
    text = unit.strip()
    if text in _DIMENSIONLESS:
        return "1"
    for pyrealm_form, cf_form in _SPELLING.items():
        text = text.replace(pyrealm_form, cf_form)
    return " ".join(text.split())


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN_PATH.exists():  # pragma: no cover - only when the file is lost
        pytest.fail(
            f"Missing {GOLDEN_PATH}. Regenerate with `just regen-pyrealm-golden`."
        )
    return json.loads(GOLDEN_PATH.read_text())


@pytest.fixture(scope="module")
def bounds() -> dict:
    """pyrealm's variable -> (unit, range) table.

    ``_data`` is private, so guard it. If the hook moves, that is itself the
    news, and a clear failure beats an AttributeError from inside a test body.
    """
    checker = pyrealm.core.bounds.BoundsChecker()
    data = getattr(checker, "_data", None)
    if not isinstance(data, dict):  # pragma: no cover - upstream restructure
        pytest.fail(
            "pyrealm's BoundsChecker no longer exposes `_data`; the unit-drift "
            "check needs rewiring to whatever replaced it."
        )
    return data


def declared_pmodel_inputs() -> dict[str, str]:
    """Units declared for every P-Model input, across `pmodel` and its input nodes.

    The method-dependent inputs sit on the ``<axis>_inputs`` variants, not on
    `pmodel`. Two variants can declare the same input, and must agree on it.
    """
    from satterc.models import pmodel as pmodel_module

    nodes = [
        pmodel_module.pmodel,
        pmodel_module.optchi_inputs__theta,
        pmodel_module.optchi_inputs__rootzonestress,
        pmodel_module.kphio_inputs__sandoval,
        pmodel_module.arrhenius_inputs__kattge_knorr,
    ]
    declared: dict[str, str] = {}
    for node in nodes:
        for argument, unit in declared_input_units(node).items():
            assert declared.setdefault(argument, unit) == unit, (
                f"{node.__name__} declares {argument!r} as {unit!r}, but another "
                f"P-Model node declares it as {declared[argument]!r}."
            )
    return declared


# Each entry maps a satterc node argument to the pyrealm variable it becomes.
PMODEL_INPUT_UNITS = [
    ("temperature_weekly", "tc"),
    ("vpd_weekly", "vpd"),
    ("co2_weekly", "co2"),
    ("pressure_weekly", "patm"),
    ("fapar_weekly", "fapar"),
    ("ppfd_weekly", "ppfd"),
    ("volumetric_water_content_weekly", "theta"),
    ("rootzonestress_weekly", "rootzonestress"),
    ("aridity_index", "aridity_index"),
]

SPLASH_INPUT_UNITS = [
    ("sunshine_fraction_daily", "sf"),
    ("temperature_daily", "tc"),
    ("precipitation_daily", "pn"),
    ("max_soil_moisture", "kWm"),
    # ("latitude", "lat") is absent on purpose. See `satterc.latitude` and
    # `test_latitude_check_uses_pyrealms_unit` below.
]

# pyrealm's own two sources disagree here. The bounds table records
# `mean_growth_temperature` as "-" (dimensionless) while PModelEnvironment's
# docstring documents it as °C, which is plainly what it is, a temperature with
# a 0-50 range. satterc follows the docstring. This is recorded as a named
# exception rather than dropped from the mapping, so that if upstream ever fixes
# the bounds entry the test fails and tells us to delete the workaround.
BOUNDS_DISAGREES_WITH_DOCSTRING = {"mean_growth_temperature": ("1", "degC")}


class TestUpstreamUnitMetadata:
    """satterc's declared units still match pyrealm's own unit tables."""

    @pytest.mark.parametrize(("argument", "variable"), PMODEL_INPUT_UNITS)
    def test_pmodel_input_units(self, bounds, argument, variable):
        declared = declared_pmodel_inputs()[argument]
        expected = normalise(bounds[variable].unit)
        assert declared == expected, (
            f"pmodel declares {argument!r} as {declared!r}, but pyrealm "
            f"{pyrealm.__version__} documents {variable!r} as {expected!r}. "
            f"{REGENERATE_HINT}"
        )

    @pytest.mark.parametrize(("argument", "variable"), SPLASH_INPUT_UNITS)
    def test_splash_input_units(self, bounds, argument, variable):
        from satterc.models import splash as splash_module

        declared_inputs = declared_input_units(splash_module.splash)
        declared = declared_inputs[argument]
        expected = normalise(bounds[variable].unit)
        assert declared == expected, (
            f"splash declares {argument!r} as {declared!r}, but pyrealm "
            f"{pyrealm.__version__} documents {variable!r} as {expected!r}. "
            f"{REGENERATE_HINT}"
        )

    def test_known_upstream_disagreement_persists(self, bounds):
        """The one variable where pyrealm's two sources contradict each other.

        Fails when upstream fixes it, at which point delete the exception and
        add the variable to `PMODEL_INPUT_UNITS`.
        """
        for variable, (bounds_unit, _) in BOUNDS_DISAGREES_WITH_DOCSTRING.items():
            assert normalise(bounds[variable].unit) == bounds_unit, (
                f"pyrealm's bounds entry for {variable!r} changed; it may now "
                f"agree with the docstring. Re-check and simplify this test."
            )

    def test_mean_growth_temperature_follows_the_docstring(self):
        declared = declared_pmodel_inputs()["mean_growth_temperature"]
        assert declared == BOUNDS_DISAGREES_WITH_DOCSTRING["mean_growth_temperature"][1]

    def test_pmodel_returns_every_declared_output(self):
        """An output pyrealm adds upstream fails here until the wrapper returns it."""
        declared = {name for name, _ in pyrealm.pmodel.PModel._data_attributes}
        assert set(_OUTPUT_ATTRIBUTES.values()) == declared
        assert set(_OUTPUT_ATTRIBUTES) == set(declared_units(PModelOut))

    @pytest.mark.parametrize(("field", "variable"), _OUTPUT_ATTRIBUTES.items())
    def test_pmodel_output_units(self, field, variable):
        """The check that catches an upstream output-unit change outright."""
        attributes = getattr(pyrealm.pmodel.PModel, "_data_attributes", None)
        if attributes is None:  # pragma: no cover - upstream restructure
            pytest.fail(
                "pyrealm's PModel no longer exposes `_data_attributes`; the "
                "output unit-drift check needs rewiring."
            )
        upstream = normalise(dict(attributes)[variable])
        declared = declared_units(PModelOut)[field]
        assert declared == upstream, (
            f"PModelOut declares {field!r} as {declared!r}, but pyrealm "
            f"{pyrealm.__version__} reports {variable!r} in {upstream!r}. "
            f"{REGENERATE_HINT}"
        )

    @pytest.mark.parametrize(
        ("variable", "lower", "upper"),
        [
            ("ppfd", 0, 3000),
            ("vpd", 0, 10000),
            ("theta", 0, 0.8),
            ("patm", 30000, 110000),
            ("co2", 0, 1000),
            ("pn", 0, 1000),
        ],
    )
    def test_valid_ranges_unmoved(self, bounds, variable, lower, upper):
        """A shifted valid range often reveals a unit change the label lags."""
        entry = bounds[variable]
        assert (entry.lower, entry.upper) == (lower, upper), (
            f"pyrealm's valid range for {variable!r} moved from "
            f"[{lower}, {upper}] to [{entry.lower}, {entry.upper}]. That usually "
            f"means the expected units changed. {REGENERATE_HINT}"
        )


class TestUpstreamApiNames:
    """Every name satterc passes to or reads from pyrealm still exists.

    Turns a rename upstream into an immediate, legible failure instead of a
    TypeError raised deep inside a pipeline run.
    """

    def test_pmodel_environment_accepts_our_arguments(self):
        import inspect

        parameters = inspect.signature(pyrealm.pmodel.PModelEnvironment).parameters
        explicit = {"tc", "vpd", "co2", "patm", "fapar", "ppfd"}
        assert explicit <= set(parameters)
        # The method-dependent inputs arrive via **kwargs,
        # so the signature cannot vouch for them; the bounds table can.
        assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())

    def test_pmodel_accepts_our_method_arguments(self):
        import inspect

        parameters = inspect.signature(pyrealm.pmodel.PModel).parameters
        assert set(PMODEL_METHODS) <= set(parameters)

    @pytest.mark.parametrize("attribute", ["gpp", "lue", "iwue"])
    def test_pmodel_outputs_exist(self, pmodel_result_raw, attribute):
        assert hasattr(pmodel_result_raw, attribute)

    def test_splash_accepts_our_arguments(self):
        import inspect

        parameters = inspect.signature(pyrealm.splash.splash.SplashModel).parameters
        assert {"lat", "elv", "sf", "tc", "pn", "dates", "kWm"} <= set(parameters)

    @pytest.mark.parametrize(
        ("prefix", "cls"), [("evap", DailyEvapFluxes), ("solar", DailySolarFluxes)]
    )
    def test_splash_fluxes_are_all_classified(self, prefix, cls):
        """A flux pyrealm adds must be returned or named as an intermediate."""
        computed = {f"{prefix}.{f.name}" for f in dataclasses.fields(cls) if not f.init}
        returned = set(_FLUX_ATTRIBUTES.values())
        assert computed == {
            name for name in returned | SPLASH_INTERMEDIATES if name.startswith(prefix)
        }


#: pyrealm attributes `splash` deliberately does not return: orbital geometry, terms
#: of the equations, and physical properties of air and water at each temperature.
SPLASH_INTERMEDIATES = {
    "solar.nu",
    "solar.lambda_",
    "solar.distance_factor",
    "solar.declination",
    "solar.ru",
    "solar.rv",
    "solar.rw",
    "solar.sunset_hour_angle",
    "solar.crossover_hour_angle",
    "evap.sat",
    "evap.lv",
    "evap.pw",
    "evap.psy",
    "evap.econ",
    "evap.rx",
}


@pytest.fixture(scope="module")
def pmodel_result():
    """The wrapped P-Model over the fixed golden case."""
    return _pmodel(**pmodel_inputs())


@pytest.fixture(scope="module")
def splash_result():
    """The wrapped SPLASH model over the fixed golden case."""
    return _splash(**splash_inputs())


def _run_pyrealm_pmodel(**overrides):
    """Call pyrealm's P-Model directly on one cell, for the invariant probes."""
    settings = {
        "tc": 20.0,
        "vpd": 1000.0,
        "co2": 400.0,
        "patm": 101325.0,
        "fapar": 0.8,
        "ppfd": 800.0,
        "theta": 0.3,
        "mean_growth_temperature": 18.0,
        "aridity_index": 1.0,
    }
    settings.update(overrides)
    drivers: dict[str, Any] = {
        name: np.atleast_1d(float(value)) for name, value in settings.items()
    }
    environment = pyrealm.pmodel.PModelEnvironment(**drivers)
    methods: dict[str, Any] = dict(PMODEL_METHODS)
    return pyrealm.pmodel.PModel(env=environment, **methods), environment


@pytest.fixture(scope="module")
def pmodel_result_raw():
    model, _ = _run_pyrealm_pmodel()
    return model


class TestPModelInvariants:
    """Relations that hold in pyrealm's unit system and not in the old one."""

    def test_gpp_is_lue_times_absorbed_photons(self):
        """GPP = LUE x fAPAR x PPFD, exactly.

        Ties the three units together. GPP's per-second goes against PPFD's, its
        mass against LUE's, and LUE's per-mole against PPFD's micromoles. A GPP
        converted to a daily rate breaks this by a factor of 86400.
        """
        model, _ = _run_pyrealm_pmodel()
        np.testing.assert_allclose(model.gpp, model.lue * 0.8 * 800.0, rtol=1e-12)

    def test_iwue_is_a_mixing_ratio_not_a_pressure(self):
        """iWUE = (5/8)(ca - ci) / P with P in MPa, which lands in umol mol-1.

        The same quantity expressed in Pa is ~10x smaller, so this tells the two
        readings apart rather than restating pyrealm's formula.
        """
        model, environment = _run_pyrealm_pmodel()
        as_mixing_ratio = (
            (5 / 8) * (environment.ca - model.optchi.ci) / (1e-6 * environment.patm)
        )
        np.testing.assert_allclose(model.iwue, as_mixing_ratio, rtol=1e-12)
        as_pressure = (5 / 8) * (environment.ca - model.optchi.ci)
        assert not np.allclose(model.iwue, as_pressure, rtol=0.5)

    def test_gpp_magnitude_is_plausible_once_converted(self):
        """A well-watered temperate cell fixes 5-40 gC m-2 d-1.

        The raw value (~249) sits far outside that window, so a flip in either
        direction fails rather than passing a bracket wide enough to admit both.
        """
        model, _ = _run_pyrealm_pmodel()
        daily = float(model.gpp[0]) * 0.0864  # ug m-2 s-1 -> g m-2 d-1
        assert 5.0 < daily < 40.0
        assert not 5.0 < float(model.gpp[0]) < 40.0

    # Feeding a Kelvin value is the point of the test. pyrealm's downstream NaN
    # arithmetic warns on the way, which is expected rather than a defect.
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_temperature_is_celsius_not_kelvin(self):
        """288.15 is a fine air temperature in K and out of range in degC.

        pyrealm's own bounds checker says so, which is a stronger statement than
        the NaN alone. The warning names the unit it expected.
        """
        in_celsius, _ = _run_pyrealm_pmodel(tc=15.0)
        assert np.isfinite(in_celsius.gpp).all()

        with pytest.warns(UserWarning, match="Check units"):
            as_kelvin, _ = _run_pyrealm_pmodel(tc=288.15)
        assert not np.isfinite(as_kelvin.gpp).all()

    def test_vpd_is_pascals_not_kilopascals(self):
        """GPP falls monotonically across a realistic VPD range in Pa."""
        values = [
            float(_run_pyrealm_pmodel(vpd=v)[0].gpp[0]) for v in (500, 1500, 3000)
        ]
        assert values[0] > values[1] > values[2]
        # Over that span the response is real but not collapsing; read as kPa the
        # same numbers would sit at the very bottom of the range.
        assert 0.5 < values[2] / values[0] < 0.95

    def test_co2_is_ppm(self):
        """Doubling CO2 from 400 to 800 ppm fertilises GPP by a plausible amount."""
        base, _ = _run_pyrealm_pmodel(co2=400.0)
        doubled, _ = _run_pyrealm_pmodel(co2=800.0)
        increase = float(doubled.gpp[0] / base.gpp[0]) - 1.0
        assert 0.05 < increase < 0.40

    def test_gpp_is_linear_in_absorbed_light(self):
        """Doubling PPFD doubles GPP, pinning it as the photon-flux driver."""
        base, _ = _run_pyrealm_pmodel(ppfd=800.0)
        doubled, _ = _run_pyrealm_pmodel(ppfd=1600.0)
        np.testing.assert_allclose(doubled.gpp, 2.0 * base.gpp, rtol=1e-12)


class TestSplashInvariants:
    """SPLASH's outputs are a water budget, and the budget has to balance in mm."""

    def test_daily_water_balance_closes(self, splash_result):
        """dSM = P + condensation - AET - runoff, every day, every pixel.

        The strongest check here. It ties ``mm`` (soil moisture, runoff) to
        ``mm d-1`` (precipitation, AET) in a single identity, and breaks if
        runoff becomes a rate, if AET's interval changes, or if soil moisture
        becomes a volumetric fraction. It runs on the wrapper's outputs, which is
        why condensation has to be one of them. The first day is left out because
        its previous state is the spin-up result, which is not an output.
        """
        change = (
            splash_result["soil_moisture_daily"].values[1:]
            - (splash_result["soil_moisture_daily"].values[:-1])
        )
        supply = (
            splash_inputs()["precipitation_daily"].values[1:]
            + splash_result["condensation_daily"].values[1:]
            - splash_result["actual_evapotranspiration_daily"].values[1:]
            - splash_result["runoff_daily"].values[1:]
        )
        np.testing.assert_allclose(change, supply, atol=1e-9)

    def test_condensation_is_a_material_part_of_the_balance(self, splash_result):
        """The balance test above would prove little if condensation were tiny."""
        assert splash_result["condensation_daily"].values.mean() > 0.1

    def test_pet_is_scaled_equilibrium_evapotranspiration(self, splash_result):
        np.testing.assert_allclose(
            splash_result["potential_evapotranspiration_daily"].values,
            (1.0 + CoreConst().k_w)
            * splash_result["equilibrium_evapotranspiration_daily"].values,
        )

    def test_ppfd_is_a_daily_total_in_moles(self, splash_result):
        """Tens of mol m-2 d-1. As a rate in umol m-2 s-1 it would be hundreds."""
        assert 5.0 < float(splash_result["estimated_ppfd_daily"].max()) < 80.0

    def test_nighttime_net_radiation_is_negative(self, splash_result):
        assert float(splash_result["nighttime_net_radiation_daily"].max()) < 0.0
        assert float(splash_result["daytime_net_radiation_daily"].min()) > 0.0

    def test_soil_moisture_is_a_depth_bounded_by_capacity(self, splash_result):
        """0 <= SM <= kWm pins soil moisture to mm, on capacity's own scale."""
        moisture = (
            splash_result["soil_moisture_daily"].transpose("time", "pixel").values
        )
        assert moisture.min() >= 0.0
        for pixel in range(N_PIXELS):
            assert moisture[:, pixel].max() <= MAX_SOIL_MOISTURE[pixel] + 1e-9
        # A volumetric fraction would sit in [0, 1]. A real depth does not.
        assert moisture.max() > 1.0

    def test_actual_never_exceeds_potential_evapotranspiration(self, splash_result):
        aet = splash_result["actual_evapotranspiration_daily"].values
        pet = splash_result["potential_evapotranspiration_daily"].values
        assert np.all(aet <= pet + 1e-9)

    def test_pet_magnitude_is_millimetres_per_day(self, splash_result):
        """Daily PET sits in single-figure mm, not metres and not W m-2."""
        pet = splash_result["potential_evapotranspiration_daily"].values
        assert pet.min() >= 0.0
        assert 0.1 < float(np.nanmax(pet)) < 15.0

    def test_elevation_is_metres(self):
        """Transmissivity scales as ``1 + f * z``, with ``f`` per metre.

        Pins the scale, so elevation read as kilometres or feet fails, as well as
        elevation being ignored.
        """
        inputs = splash_inputs()
        sea_level = _splash(**{**inputs, "elevation": inputs["elevation"] * 0.0})
        high = _splash(**{**inputs, "elevation": inputs["elevation"] * 0.0 + 3000.0})
        _, _, per_metre = CoreConst().transmissivity_coef
        np.testing.assert_allclose(
            high["transmissivity_daily"].values
            / sea_level["transmissivity_daily"].values,
            1.0 + per_metre * 3000.0,
        )


class TestGoldenValues:
    """Absolute magnitudes, pinned against a reviewed pyrealm version.

    A pyrealm upgrade does not excuse these. The whole point of the anchor is to
    fire when pyrealm changes what its numbers mean, and a check that switches
    itself off on the version bump would go quiet at exactly the moment it is
    meant to speak. So none of this class skips: an upgrade that moves the
    numbers turns the suite red, and somebody reads the diff.
    """

    def test_recorded_pyrealm_version(self, golden):
        """Fails first, so the headline arrives before the numeric diffs.

        This says nothing the individual failures below do not also say — it
        just says it once, at the top, in terms of versions rather than in terms
        of a mean that moved in the sixth decimal place.
        """
        if golden["pyrealm_version"] != pyrealm.__version__:
            pytest.fail(
                f"Golden file was generated against pyrealm "
                f"{golden['pyrealm_version']}, installed is "
                f"{pyrealm.__version__}. Any failures below are most likely this "
                f"and not a bug in satterc. {REGENERATE_HINT}"
            )

    @pytest.mark.parametrize("field", _OUTPUT_ATTRIBUTES)
    @pytest.mark.parametrize("statistic", ["mean", "std", "min", "max"])
    def test_pmodel_outputs(self, golden, pmodel_result, field, statistic):
        expected = golden["pmodel"]["outputs"][field][statistic]
        actual = summarize(pmodel_result[field])[statistic]
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-6,
            err_msg=(
                f"pmodel {field!r} {statistic} moved against the value recorded "
                f"for pyrealm {golden['pyrealm_version']}. {REGENERATE_HINT}"
            ),
        )

    @pytest.mark.parametrize("field", SplashOut.__annotations__)
    @pytest.mark.parametrize("statistic", ["mean", "std", "min", "max"])
    def test_splash_outputs(self, golden, splash_result, field, statistic):
        expected = golden["splash"]["outputs"][field][statistic]
        actual = summarize(splash_result[field])[statistic]
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-6,
            err_msg=(
                f"splash {field!r} {statistic} moved against the value recorded "
                f"for pyrealm {golden['pyrealm_version']}. {REGENERATE_HINT}"
            ),
        )


class TestAnnotationsMatchGolden:
    """The pinned numbers and the declared units describe the same quantity."""

    @pytest.mark.parametrize(
        ("model", "typed_dict"), [("pmodel", PModelOut), ("splash", SplashOut)]
    )
    def test_declared_units_match_recorded(self, golden, model, typed_dict):
        assert declared_units(typed_dict) == golden[model]["units"], (
            f"{model}'s declared units no longer match those recorded alongside "
            f"the golden values, so the pinned numbers describe a different "
            f"quantity than the annotation claims. {REGENERATE_HINT}"
        )


def test_latitude_check_uses_pyrealms_unit():
    """`splash` checks `latitude` in its body, against the unit pyrealm wants.

    The check stands in for a declaration until catena labels the latitude it
    derives; see `satterc.latitude`. The undeclared parameter and the checked
    unit are pinned together, so removing the workaround means editing both.
    """
    from satterc.latitude import LATITUDE_UNITS
    from satterc.models import splash as splash_module

    assert "latitude" not in declared_input_units(splash_module.splash), (
        "`splash` now declares `latitude`, so remove `satterc.latitude` and "
        "restore the SPLASH_INPUT_UNITS entry."
    )
    checker = pyrealm.core.bounds.BoundsChecker()
    assert normalise(checker._data["lat"].unit) == LATITUDE_UNITS


def test_latitude_case_spans_hemispheres():
    """Guards the golden case itself. Latitude has to vary across the pixels, or
    the per-pixel statics in the SPLASH anchor would prove nothing."""
    assert min(LATITUDE) < 0 < max(LATITUDE)
