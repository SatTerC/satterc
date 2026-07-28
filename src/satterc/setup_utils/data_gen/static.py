"""Table of static (time-invariant) synthetic variables.

Each entry is a `satterc.setup_utils.data_gen.spec.Var` bound to a module-level
name: units, a long name, and one lambda producing values from the `StaticCtx` it
is handed. Adding an input for a new model means adding an entry here — see the
module docstring of `satterc.setup_utils.data_gen.spec` for what the context
offers. The entry's docstring is what explains *why* it is generated that way.

Several entries derive from `elevation` rather than drawing independently, so
that soil properties covary the way they do in the field: deeper, wetter, more
carbon-rich soils in the lowlands. `Resolver` memoises `elevation`, so every
consumer sees the same field.
"""

import sys

import numpy as np
from numpy.typing import NDArray

from .spec import StaticCtx, Var, collect_vars

# Initial carbon pools (root t ha-1, leaf t ha-1, stem t ha-1) indexed by plant
# type, in `sgam.pft.PlantFunctionalType` order: 0=tree, 1=grass, 2=shrub, 3=crop.
# Magnitudes follow the standing biomass each type actually carries: a tree holds
# most of its carbon in a long-lived stem, a grass has no woody stem at all but a
# substantial root mat, a shrub sits between the two, and an annual crop carries
# little of anything because it is harvested each year.
_POOL_BY_TYPE: dict[int, tuple[float, float, float]] = {
    0: (8.0, 2.0, 25.0),
    1: (5.0, 1.5, 1.0),
    2: (4.0, 1.5, 8.0),
    3: (1.5, 1.0, 2.0),
}


def _pool(ctx: StaticCtx, index: int) -> NDArray[np.float64]:
    """Look up one carbon pool from `_POOL_BY_TYPE` for each pixel's plant type."""
    plant_type = ctx.static("plant_type")
    return np.array([_POOL_BY_TYPE[int(t)][index] for t in plant_type])


latitude = Var(
    "degrees_north",
    "latitude",
    lambda g: g.lat,
)
"""Each pixel's latitude, taken from the grid rather than invented."""

elevation = Var(
    "m",
    "elevation",
    lambda g: 100.0 + 200.0 * g.lat_norm + g.smooth_noise(35.0),
    bounds=(0.0, 1000.0),
)
"""South-north gradient (100 m to 300 m) plus spatially coherent noise.

Several soil properties derive from this, so it is the one static field whose
spatial structure is worth getting right.
"""

plant_type = Var(
    "1",
    "plant type",
    lambda g: np.arange(g.n_pixels) % 4,
    bounds=(0.0, 3.0),
    dtype=np.int32,
)
"""Plant functional type: 0=tree, 1=grass, 2=shrub, 3=crop.

The encoding is the declaration order of `sgam.pft.PlantFunctionalType`, which is
what both models index into — SGAM to select its PFT parameters, RothC to decide
crop bare seasons and DPM/RPM ratios — so it is not a synthetic-data convention
that could drift from them.

Cycled by pixel index so that any grid of four or more pixels holds all four
types, and the spatial layout is the same for a given grid shape.
"""

max_soil_moisture = Var(
    "mm",
    "maximum soil moisture",
    lambda g: 300.0 - 0.2 * g.static("elevation"),
    bounds=(100.0, 300.0),
)
"""Bucket capacity, falling with elevation."""

clay_content = Var(
    "percent",
    "clay content",
    lambda g: g.uniform(10.0, 40.0),
)
"""Typical topsoil clay fractions. RothC consumes percent, not a fraction."""

soil_depth = Var(
    "cm",
    "soil depth",
    lambda g: 120.0 - 0.08 * g.static("elevation"),
    bounds=(40.0, 120.0),
)
"""Deeper soils in lowland valleys, shallower on upland terrain."""

organic_carbon_stocks = Var(
    "t ha-1",
    "soil organic carbon stocks",
    lambda g: 130.0 - 0.06 * g.static("elevation") + g.uniform(-10.0, 10.0),
    bounds=(30.0, 200.0),
)
"""Upland mineral soils carry less SOC than lowland peats."""

root_pool_init = Var(
    "t ha-1",
    "initial root carbon pool",
    lambda g: _pool(g, 0),
)
"""Initial root carbon, looked up from the pixel's `plant_type`."""

leaf_pool_init = Var(
    "t ha-1",
    "initial leaf carbon pool",
    lambda g: _pool(g, 1),
)
"""Initial leaf carbon, looked up from the pixel's `plant_type`."""

stem_pool_init = Var(
    "t ha-1",
    "initial stem carbon pool",
    lambda g: _pool(g, 2),
)
"""Initial stem carbon, looked up from the pixel's `plant_type`."""


#: Static variables, keyed by the name used in a config's ``[inputs]`` section.
STATIC_VARS: dict[str, Var] = collect_vars(sys.modules[__name__])
