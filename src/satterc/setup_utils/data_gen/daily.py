"""Table of daily synthetic variables.

Each entry is a `satterc.setup_utils.data_gen.spec.Var` bound to a module-level
name: units, a long name, and one lambda producing values from the `DailyCtx` it
is handed. Adding an input for a new model means adding an entry here — see the
module docstring of `satterc.setup_utils.data_gen.spec` for what the context
offers. The entry's docstring is what explains *why* it is generated that way.

The values only need to be plausible enough that the models run and produce
sensible output: seasonal where the model responds to seasonality, correlated in
time where the model integrates state, and inside physical bounds throughout.
A variable with no entry here is not an error — `fallback` invents plausible
noise from its name — so only add one where the structure matters.
"""

import sys

import numpy as np
from numpy.typing import NDArray

from .spec import Var, collect_vars


def _saturation_vapour_pressure(
    temperature: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Tetens' formula for saturation vapour pressure over water, in Pa."""
    return 610.78 * np.exp(temperature / (temperature + 237.3) * 17.27)


temperature = Var(
    "degC",
    "air temperature",
    lambda g: (
        10.0
        + (g.lat - 52.0) * 0.5
        + (g.lon + 1.0) * 0.3
        + g.cycle(10.0, phase=-np.pi / 2)
        + g.ar1(sigma=2.0)
    ),
)
"""Cooler at higher latitudes, milder near the west coast, peaking in summer."""

precipitation = Var(
    "mm d-1",
    "precipitation",
    lambda g: np.where(
        g.uniform(0.0, 1.0) < 0.6,
        np.random.exponential(np.abs(2.5 + (54.0 - g.lat) * 0.3 + g.cycle(1.0)) + 0.1),
        0.0,
    ),
    bounds=(0.0, None),
)
"""Intermittent: 60% of days are wet, and wet-day totals are exponential.

Rain is the one climate input whose *distribution* matters rather than its mean —
a model integrating soil moisture responds quite differently to a steady drizzle
than to the same total falling on six days in ten.
"""

sunshine_fraction = Var(
    "1",
    "sunshine fraction",
    lambda g: g.cycle(0.3, baseline=0.5) + g.ar1(sigma=0.12, phi=0.6),
    bounds=(0.0, 1.0),
)
"""Seasonal, with day-to-day persistence: cloudy spells last more than a day."""

lai = Var(
    "1",
    "leaf area index",
    lambda g: g.cycle(2.5, phase=-np.pi / 3, baseline=3.0) + g.uniform(-0.3, 0.3),
    bounds=(0.1, 6.0),
)
"""Canopy development peaking somewhat before midsummer."""

gpp = Var(
    "g m-2 d-1",
    "gross primary productivity",
    lambda g: (
        8.0
        + g.cycle(5.0, phase=-np.pi / 3)
        * np.maximum(g.daily("temperature") - 5.0, 0.0)
        / 15.0
        + g.uniform(-1.0, 1.0)
    ),
    bounds=(0.1, None),
)
"""Seasonal amplitude gated on temperature above a 5 degC growth threshold."""

co2 = Var(
    "ppm",
    "atmospheric CO2 concentration",
    lambda g: (
        412.0
        + 5.0 * g.day / max(g.n_days - 1, 1)
        + g.cycle(3.0)
        + np.random.normal(0.0, 1.0, (g.n_days, 1))
    ),
)
"""A rising trend with a seasonal wobble, spatially uniform.

The ``(n_days, 1)`` column broadcasts across pixels, which is how a variable
declares that it does not vary in space.
"""

fapar = Var(
    "1",
    "fAPAR",
    lambda g: g.cycle(0.25, baseline=0.55) + g.uniform(-0.1, 0.1),
    bounds=(0.05, 0.95),
)
"""Fraction of absorbed photosynthetically active radiation, following the canopy."""

ppfd = Var(
    "umol m-2 s-1",
    "photosynthetic photon flux density",
    lambda g: (
        1200.0
        * np.abs(np.sin(np.pi * g.doy / (365.25 / 2)))
        * (0.4 + g.uniform(0.2, 0.6))
    ),
)
"""Clear-sky maximum following daylength, attenuated by random cloud cover."""

pressure = Var(
    "Pa",
    "atmospheric pressure",
    lambda g: (
        101325.0 - 10.0 * g.static("elevation") + g.cycle(500.0) + g.normal(0.0, 300.0)
    ),
)
"""Sea-level pressure less a lapse with elevation, plus synoptic variation."""

vpd = Var(
    "Pa",
    "vapour pressure deficit",
    lambda g: (
        _saturation_vapour_pressure(g.daily("temperature"))
        * (1.0 - np.clip(0.5 + g.uniform(-0.2, 0.2), 0.1, 0.95))
    ),
    bounds=(50.0, 3000.0),
)
"""Saturation deficit at the generated temperature and a random humidity.

Derived from `temperature` rather than drawn independently, so the two stay
physically consistent — `Resolver` memoises the temperature field, so both see
the same one.
"""

wind_speed = Var(
    "m s-1",
    "wind speed",
    lambda g: np.random.weibull(2.0, g.shape) * g.cycle(1.5, phase=np.pi, baseline=4.0),
    bounds=(0.0, None),
)
"""Weibull (shape 2, typical for mid-latitudes), windier in winter."""

dummy_variable = Var(
    "1",
    "dummy variable",
    lambda _: np.nan,
)
"""All-NaN placeholder, for exercising a config's plumbing without any physics."""


#: Daily variables, keyed by the name used in a config's ``[inputs]`` section.
DAILY_VARS: dict[str, Var] = collect_vars(sys.modules[__name__])
