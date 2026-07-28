"""Tests for the disturbance-detection ``apply_ufunc`` block seam
(``satterc.models.sgam._disturbances_daily``).

``sgam.Disturbances.forward`` is a *whole-block* computation: it diffs GPP/LAI along the
time axis and is otherwise element-wise over pixels, so it vectorises over ``pixel``
exactly like SPLASH. The seam replaces a bespoke ``xarray_io`` ``.data`` pull with
``xr.apply_ufunc`` (``time`` the core dim, ``pixel`` the broadcast dim,
``vectorize=False`` so the kernel still vectorises the whole block), retiring the last
use of ``xarray_io``.

``disturbance_threshold`` is per-PFT and therefore per-pixel: it enters the block as a
``(pixel,)`` array that broadcasts inside ``Disturbances.forward``. The reference below
is deliberately built the *other* way -- one scalar-threshold ``Disturbances`` call per
pixel column -- so it is an independent check that the broadcast is doing what a
per-pixel loop would.

1. Regression -- the seam reproduces an independent per-pixel ``Disturbances.forward``
   reference (the anchor pins the apply_ufunc plumbing: moveaxis, core-dim handling,
   threshold broadcast, coord reattach, transpose).
2. Canonical dims/coords -- every output ``(time, pixel)`` with the ``time`` coord
   reattached.
3. Per-pixel inputs differ -- distinct disturbance days per column give distinct outputs.
4. Per-PFT thresholds bite -- a moderate decline that exceeds the crop threshold but not
   the grass/tree ones is flagged only on the crop pixel.
5. Strategy C stays open -- ``.chunk({"pixel": k})`` (dask-backed) inputs give identical
   results. Skipped when dask is absent.
6. Caching intact -- the seam is internal to the node, so the cached pipeline still
   matches the uncached one.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sgam import Disturbances
from sgam.pft import PlantFunctionalType, get_default_pft_params

from satterc.models.sgam import _disturbances_daily, pft_params

N_DAYS = 120
N_PIXELS = 3
DAILY_DATES = pd.date_range("2020-01-01", periods=N_DAYS, freq="D")
PIXELS = np.arange(N_PIXELS)

# One pixel per PFT of interest: tree (threshold 0.3), crop (0.1), grass (0.2).
PLANT_TYPES = np.array([0, 3, 1])
TREE_PX, CROP_PX, GRASS_PX = 0, 1, 2

# A disturbance fires when, on a warm day, GPP and LAI both drop by more than the
# pixel's PFT threshold. Put a large drop on a *different* day per pixel so columns
# differ, then one shared moderate drop that only the crop threshold catches.
DROP_DAYS = np.array([40, 60, 80])
BIG_DROP = 0.2  # 80% decline -- above every PFT threshold
MODERATE_DROP_DAY = 100
MODERATE_DROP = 0.85  # 15% decline -- above crop (0.1), below grass (0.2)/tree (0.3)
GROWING_SEASON_LIMIT = 10.0


def _build_climate() -> dict[str, np.ndarray]:
    """Warm, smooth climate with one sharp GPP+LAI crash per pixel (distinct days),
    plus a shared moderate decline that only a crop-threshold pixel should flag."""
    temperature = np.full((N_DAYS, N_PIXELS), 20.0)  # always above growing-season limit
    gpp = np.full((N_DAYS, N_PIXELS), 5.0)
    lai = np.full((N_DAYS, N_PIXELS), 2.0)
    for p, day in enumerate(DROP_DAYS):
        gpp[day:, p] *= BIG_DROP
        lai[day:, p] *= BIG_DROP
    gpp[MODERATE_DROP_DAY:, :] *= MODERATE_DROP
    lai[MODERATE_DROP_DAY:, :] *= MODERATE_DROP
    return {"temperature": temperature, "gpp": gpp, "lai": lai}


def _temporal(base: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        base,
        dims=["time", "pixel"],
        coords={"time": DAILY_DATES, "pixel": PIXELS},
    )


def _static(values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(values, dims=["pixel"], coords={"pixel": PIXELS})


@pytest.fixture(scope="module")
def disturbance_inputs() -> dict:
    climate = _build_climate()
    return dict(
        temperature_daily=_temporal(climate["temperature"]),
        gpp_daily=_temporal(climate["gpp"]),
        lai_daily=_temporal(climate["lai"]),
        disturbance_threshold=pft_params(_static(PLANT_TYPES))["disturbance_threshold"],
        growing_season_limit=GROWING_SEASON_LIMIT,
    )


def _reference_block(inputs: dict) -> np.ndarray:
    """Independent per-pixel reference: one scalar-threshold ``Disturbances`` call per
    column, with the threshold looked up straight from the sgam PFT defaults."""
    members = list(PlantFunctionalType)
    columns = []
    for p, pft_int in enumerate(PLANT_TYPES):
        threshold = get_default_pft_params(members[int(pft_int)]).disturbance_threshold
        detector = Disturbances(
            growing_season_limit=GROWING_SEASON_LIMIT,
            disturbance_threshold=threshold,
        )
        columns.append(
            detector.forward(
                inputs["temperature_daily"].values[:, p],
                inputs["gpp_daily"].values[:, p],
                inputs["lai_daily"].values[:, p],
                aggregate=False,
            )
        )
    return np.stack(columns, axis=1)


class TestRegression:
    """The block seam reproduces the per-pixel Disturbances reference."""

    @pytest.fixture(scope="class")
    def seam_result(self, disturbance_inputs):
        return _disturbances_daily(**disturbance_inputs)

    @pytest.fixture(scope="class")
    def reference(self, disturbance_inputs) -> np.ndarray:
        return _reference_block(disturbance_inputs)

    def test_matches_reference(self, seam_result, reference):
        np.testing.assert_allclose(
            seam_result.transpose("time", "pixel").values, reference
        )

    def test_canonical_dims_and_coords(self, seam_result):
        assert seam_result.dims == ("time", "pixel")
        assert seam_result.sizes == {"time": N_DAYS, "pixel": N_PIXELS}
        # apply_ufunc drops the time coord; the seam must reattach it.
        assert "time" in seam_result.coords
        assert "pixel" in seam_result.coords
        np.testing.assert_array_equal(
            seam_result.coords["time"].values, DAILY_DATES.values
        )

    def test_per_pixel_inputs_differ(self, seam_result):
        # The crash is on a different day per pixel, so the disturbance columns must
        # differ; identical columns would mean the block was collapsed/broadcast wrong.
        sev = seam_result.transpose("time", "pixel").values
        assert not np.allclose(sev[:, 0], sev[:, 1])
        assert not np.allclose(sev[:, 1], sev[:, 2])
        # And a disturbance is actually detected on each pixel's drop day.
        for p, day in enumerate(DROP_DAYS):
            assert sev[day, p] > 0.0

    def test_thresholds_are_per_pft(self, seam_result):
        # The shared 15% decline is above the crop threshold (0.1) but below the
        # grass (0.2) and tree (0.3) ones, so only the crop pixel flags it. A
        # single hardcoded threshold could not produce this pattern.
        sev = seam_result.transpose("time", "pixel").values
        assert sev[MODERATE_DROP_DAY, CROP_PX] > 0.0
        assert sev[MODERATE_DROP_DAY, GRASS_PX] == 0.0
        assert sev[MODERATE_DROP_DAY, TREE_PX] == 0.0
        # Over the whole record the crop pixel registers strictly more events.
        n_events = (sev > 0.0).sum(axis=0)
        assert n_events[CROP_PX] > n_events[GRASS_PX] == n_events[TREE_PX]


class TestStrategyCStaysOpen:
    """Chunked (dask-backed) ``pixel`` inputs give identical results to eager numpy."""

    def test_chunked_pixel_equivalence(self, disturbance_inputs):
        pytest.importorskip("dask")
        eager = _disturbances_daily(**disturbance_inputs)

        chunked_inputs = dict(disturbance_inputs)
        for key, val in chunked_inputs.items():
            if isinstance(val, xr.DataArray):
                chunked_inputs[key] = val.chunk({"pixel": 1})
        chunked = _disturbances_daily(
            temperature_daily=chunked_inputs["temperature_daily"],
            gpp_daily=chunked_inputs["gpp_daily"],
            lai_daily=chunked_inputs["lai_daily"],
            disturbance_threshold=chunked_inputs["disturbance_threshold"],
            growing_season_limit=GROWING_SEASON_LIMIT,
        )

        np.testing.assert_allclose(
            chunked.compute().transpose("time", "pixel").values,
            eager.transpose("time", "pixel").values,
        )


class TestCachingIntact:
    """The cached disturbances node matches the uncached one (seam is node-internal)."""

    def test_cached_run_matches_uncached(self, tmp_path, disturbance_inputs):
        from conduit import CacheSpec, build_driver

        # disturbances_daily depends on the pft_params node, which the driver builds
        # from plant_type.
        inputs = {
            key: val
            for key, val in disturbance_inputs.items()
            if key != "disturbance_threshold" and key != "growing_season_limit"
        }
        inputs["plant_type"] = _static(PLANT_TYPES)
        spec = CacheSpec(path=str(tmp_path / "cache"))

        def run(cache):
            dr = build_driver(["satterc.models.sgam"], {}, cache=cache)
            return dr.execute(["disturbances_daily"], inputs=inputs)  # type: ignore[reportArgumentType]

        uncached = run(None)
        run(spec)  # cold cache
        cached = run(spec)  # warm cache

        np.testing.assert_allclose(
            cached["disturbances_daily"].values,
            uncached["disturbances_daily"].values,
        )
        assert (tmp_path / "cache").exists()
