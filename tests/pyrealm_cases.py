"""The fixed pyrealm cases behind the unit anchor, shared by test and generator.

`tests/test_pyrealm_units.py` asserts against ``tests/data/pyrealm_golden.json``,
and `scripts/regen_pyrealm_golden.py` writes it. Both have to drive the models
with byte-identical inputs or the comparison means nothing, so the inputs are
defined once, here, and use no RNG.
"""

from typing import Any, get_args, get_type_hints

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

# Shapes mirror the seam-test fixtures, which are known to converge. SPLASH gets
# a full leap year because spin-up needs one; the P-Model gets 52 weeks.
N_WEEKS = 52
N_DAYS = 366
N_PIXELS = 3
WEEKLY_DATES = pd.date_range("2020-01-01", periods=N_WEEKS, freq="7D")
DAILY_DATES = pd.date_range("2020-01-01", periods=N_DAYS, freq="D")
PIXELS = np.arange(N_PIXELS)

PMODEL_METHODS = {
    "method_optchi": "prentice14",
    "method_jmaxlim": "wang17",
    "method_kphio": "temperature",
    "method_arrhenius": "simple",
}

# Per-pixel SPLASH statics, spanning northern, southern and near-equatorial
# latitudes at three elevations, so both statics change the result.
LATITUDE = [51.5, -33.0, 10.0]
ELEVATION = [50.0, 800.0, 2000.0]
MAX_SOIL_MOISTURE = [150.0, 120.0, 180.0]

SPLASH_MAX_ITER = 50
SPLASH_MAX_DIFF = 0.01


def _weekly(base: NDArray[np.float64]) -> xr.DataArray:
    return xr.DataArray(
        base, dims=["time", "pixel"], coords={"time": WEEKLY_DATES, "pixel": PIXELS}
    )


def _daily(base: NDArray[np.float64]) -> xr.DataArray:
    return xr.DataArray(
        base, dims=["time", "pixel"], coords={"time": DAILY_DATES, "pixel": PIXELS}
    )


def _static(values: list[float]) -> xr.DataArray:
    return xr.DataArray(
        np.asarray(values, dtype=float), dims=["pixel"], coords={"pixel": PIXELS}
    )


def _offsets(spread: float) -> NDArray[np.float64]:
    """Evenly spaced per-pixel offsets, so no two columns are identical."""
    return np.linspace(-spread, spread, N_PIXELS)


def _seasonal(n: int) -> NDArray[np.float64]:
    """One smooth annual cycle over ``n`` steps."""
    return np.sin(2 * np.pi * np.arange(n) / 365)


def pmodel_inputs() -> dict[str, Any]:
    """Fixed P-Model drivers, in the units `satterc.models.pmodel` declares."""
    season = _seasonal(N_WEEKS * 7)[::7][:N_WEEKS]

    def field(center: float, spread: float, amplitude: float = 0.0) -> xr.DataArray:
        return _weekly(center + amplitude * season[:, None] + _offsets(spread)[None, :])

    return {
        "temperature_weekly": field(15.0, 5.0, amplitude=8.0),
        "vpd_weekly": field(1000.0, 300.0, amplitude=200.0),
        "co2_weekly": _weekly(np.full((N_WEEKS, N_PIXELS), 400.0)),
        "pressure_weekly": _weekly(np.full((N_WEEKS, N_PIXELS), 101325.0)),
        "fapar_weekly": field(0.5, 0.2),
        "ppfd_weekly": field(500.0, 150.0, amplitude=100.0),
        "mean_growth_temperature": field(15.0, 5.0),
        "aridity_index": field(1.0, 0.3),
        "volumetric_water_content_weekly": field(0.3, 0.1),
        **PMODEL_METHODS,
    }


def splash_inputs() -> dict[str, Any]:
    """Fixed SPLASH drivers, in the units `satterc.models.splash` declares."""
    season = _seasonal(N_DAYS)
    return {
        "sunshine_fraction_daily": _daily(
            np.tile(np.clip(0.5 + 0.05 * season, 0.0, 1.0)[:, None], (1, N_PIXELS))
        ),
        "temperature_daily": _daily(
            10.0 + 8.0 * season[:, None] + _offsets(4.0)[None, :]
        ),
        "precipitation_daily": _daily(np.full((N_DAYS, N_PIXELS), 2.0)),
        "elevation": _static(ELEVATION),
        "latitude": _static(LATITUDE),
        "max_soil_moisture": _static(MAX_SOIL_MOISTURE),
        "dates_daily": DAILY_DATES,
        "soil_moisture_init_max_iter": SPLASH_MAX_ITER,
        "soil_moisture_init_max_diff": SPLASH_MAX_DIFF,
    }


def summarize(da: xr.DataArray) -> dict[str, list[float]]:
    """Per-pixel summary statistics for one output.

    Four moments per pixel rather than the full array. That pins any scale factor
    or offset, which is what a unit change looks like, and stays small enough to
    read in a diff.
    """
    values = da.transpose("time", "pixel").values
    return {
        "mean": np.nanmean(values, axis=0).tolist(),
        "std": np.nanstd(values, axis=0).tolist(),
        "min": np.nanmin(values, axis=0).tolist(),
        "max": np.nanmax(values, axis=0).tolist(),
    }


def declared_units(typed_dict: type) -> dict[str, str]:
    """The unit string each field of a model's output `TypedDict` declares."""
    hints = get_type_hints(typed_dict, include_extras=True)
    return {name: get_args(hint)[1] for name, hint in hints.items()}
