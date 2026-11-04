"""Tests for the SPLASH ``apply_ufunc`` block seam (``satterc.models.splash._splash``).

Like the P-Model, SPLASH is a *whole-block* pyrealm model — it vectorises over the
spatial axis internally and loops only over ``time`` (soil moisture carries state day to
day). The seam replaces a bespoke ``xarray_io`` ``.data`` pull with ``xr.apply_ufunc``
(``time`` as the core dim, ``pixel`` the broadcast dim, ``vectorize=False`` so pyrealm
still vectorises the block), which both keeps a future dask backend reachable *and* fixes
a live correctness bug: the old code passed ``latitude[0]``/``elevation[0]``/
``max_soil_moisture[0]`` to pyrealm, so every pixel silently received pixel 0's static
parameters.

1. Regression — the seam reproduces an independent *whole-block* pyrealm call with the
   statics correctly broadcast per pixel (the anchor pins the apply_ufunc plumbing:
   moveaxis, core-dim handling, coord reattach, transpose).
2. Bug fix — feeding identical climate to two pixels with *different* latitudes yields
   *different* outputs; the old ``[0]``-indexing would have tied them together.
3. Strategy C stays open — ``.chunk({"pixel": k})`` (dask-backed) inputs give identical
   results. Skipped when dask is absent.
4. Caching intact — the seam is internal to the node, so the cached pipeline still
   matches the uncached one.
"""

from operator import attrgetter

import numpy as np
import pandas as pd
import pyrealm.core.calendar
import pyrealm.splash.splash
import pytest
import xarray as xr

from satterc.models.splash import (
    _FLUX_ATTRIBUTES,
    _OUTPUT_FIELDS,
    _describe_pixels,
    _splash,
    splash_config,
)

N_DAYS = 366  # 2020 is a leap year; SPLASH needs a full year for spin-up
N_PIXELS = 3
DAILY_DATES = pd.date_range("2020-01-01", periods=N_DAYS, freq="D")
PIXELS = np.arange(N_PIXELS)
MAX_ITER = 50
MAX_DIFF = 0.01

# Per-pixel statics — deliberately distinct so the old [0]-indexing bug (every pixel
# getting pixel 0's lat/elv/kWm) would change the outputs and fail the regression check.
LATITUDE = np.array([51.5, -33.0, 10.0])  # NH, SH, near-equator
ELEVATION = np.array([50.0, 800.0, 2000.0])
MAX_SM = np.array([150.0, 120.0, 180.0])

_DAY = np.arange(N_DAYS)
_SEASON = np.sin(2 * np.pi * _DAY / 365)


def _temporal(base: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        base,
        dims=["time", "pixel"],
        coords={"time": DAILY_DATES, "pixel": PIXELS},
    )


def _static(values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(values, dims=["pixel"], coords={"pixel": PIXELS})


@pytest.fixture(scope="module")
def splash_inputs() -> dict:
    """Heterogeneous multi-pixel inputs; smooth seasonal climate so spin-up converges."""
    # Per-pixel temperature offsets; sunshine/precip seasonal but pixel-invariant.
    temp = 10.0 + 8.0 * _SEASON[:, None] + np.array([0.0, -5.0, 3.0])[None, :]
    sf = np.clip(0.5 + 0.05 * _SEASON, 0, 1)
    return dict(
        sunshine_fraction_daily=_temporal(np.tile(sf[:, None], (1, N_PIXELS))),
        temperature_daily=_temporal(temp),
        precipitation_daily=_temporal(np.full((N_DAYS, N_PIXELS), 2.0)),
        elevation=_static(ELEVATION),
        latitude=_static(LATITUDE),
        max_soil_moisture=_static(MAX_SM),
        dates_daily=DAILY_DATES,
        soil_moisture_init_max_iter=MAX_ITER,
        soil_moisture_init_max_diff=MAX_DIFF,
    )


def _reference_block(inputs: dict) -> dict[str, np.ndarray]:
    """Independent whole-block pyrealm reference with statics correctly broadcast.

    Returns a dict of (time, pixel) numpy arrays, the anchor for the regression check.
    ``lat``/``elv`` go in as ``(1, pixel)`` (broadcast over time); ``kWm`` as ``(pixel,)``
    — the layout pyrealm requires and the bug fix relies on.
    """
    calendar = pyrealm.core.calendar.Calendar(inputs["dates_daily"].values)
    model = pyrealm.splash.splash.SplashModel(
        lat=inputs["latitude"].values[None, :],
        elv=inputs["elevation"].values[None, :],
        sf=inputs["sunshine_fraction_daily"].values,
        tc=inputs["temperature_daily"].values,
        pn=inputs["precipitation_daily"].values,
        dates=calendar,
        kWm=inputs["max_soil_moisture"].values,
    )
    # The capacity pyrealm 2.0.0 leaves out; see `TestCapacityReachesAET`.
    model.evap.kWm = model.kWm
    init = model.estimate_initial_soil_moisture(
        max_iter=MAX_ITER, max_diff=MAX_DIFF, verbose=False
    )
    aet, moisture, runoff = model.calculate_soil_moisture(init)
    return {
        "actual_evapotranspiration_daily": aet,
        "soil_moisture_daily": moisture,
        "runoff_daily": runoff,
        **{name: attrgetter(a)(model) for name, a in _FLUX_ATTRIBUTES.items()},
    }


class TestRegression:
    """The block seam reproduces the whole-block pyrealm reference, for every output."""

    @pytest.fixture(scope="class")
    def seam_result(self, splash_inputs):
        return _splash(**splash_inputs)

    @pytest.fixture(scope="class")
    def reference(self, splash_inputs) -> dict:
        return _reference_block(splash_inputs)

    @pytest.mark.parametrize("name", _OUTPUT_FIELDS)
    def test_matches_reference(self, seam_result, reference, name):
        np.testing.assert_allclose(
            seam_result[name].transpose("time", "pixel").values, reference[name]
        )

    def test_canonical_dims_and_coords(self, seam_result):
        for da in seam_result.values():
            assert da.dims == ("time", "pixel")
            assert da.sizes == {"time": N_DAYS, "pixel": N_PIXELS}
            # apply_ufunc drops the time coord; the seam must reattach it.
            assert "time" in da.coords
            assert "pixel" in da.coords
            np.testing.assert_array_equal(da.coords["time"].values, DAILY_DATES.values)


class TestPerPixelStaticsThreaded:
    """The fix: each pixel uses its own latitude (old code used pixel 0's for all)."""

    def test_identical_climate_different_latitude_differs(self, splash_inputs):
        # Two pixels, identical climate, opposite-hemisphere latitudes. Solar fluxes
        # depend on latitude, so correct per-pixel threading must split the columns;
        # the old latitude[0] bug would have made them identical.
        n = 2
        px = np.arange(n)
        col_sf = np.clip(0.5 + 0.05 * _SEASON, 0, 1)
        col_tc = 10.0 + 8.0 * _SEASON

        def md(v):
            return xr.DataArray(
                v, dims=["time", "pixel"], coords={"time": DAILY_DATES, "pixel": px}
            )

        def sd(v):
            return xr.DataArray(
                np.asarray(v, dtype=float), dims=["pixel"], coords={"pixel": px}
            )

        result = _splash(
            sunshine_fraction_daily=md(np.tile(col_sf[:, None], (1, n))),
            temperature_daily=md(np.tile(col_tc[:, None], (1, n))),
            precipitation_daily=md(np.full((N_DAYS, n), 2.0)),
            elevation=sd([100.0, 100.0]),
            latitude=sd([51.5, -51.5]),
            max_soil_moisture=sd([150.0, 150.0]),
            dates_daily=DAILY_DATES,
            soil_moisture_init_max_iter=MAX_ITER,
            soil_moisture_init_max_diff=MAX_DIFF,
        )
        sm = result["soil_moisture_daily"].transpose("time", "pixel").values
        assert not np.allclose(sm[:, 0], sm[:, 1])


class TestCapacityReachesAET:
    """``max_soil_moisture`` sets the supply rate in AET, not only the cap and runoff."""

    def _pair(self, capacities: list[float]) -> dict:
        """Two pixels with identical sites and climate, differing only in capacity."""
        n = len(capacities)
        px = np.arange(n)

        def md(v):
            return xr.DataArray(
                v, dims=["time", "pixel"], coords={"time": DAILY_DATES, "pixel": px}
            )

        def sd(v):
            return xr.DataArray(
                np.asarray(v, dtype=float), dims=["pixel"], coords={"pixel": px}
            )

        # Dry half the year, so moisture never reaches either capacity at that point.
        rain = np.clip(3.0 * np.sin(2 * np.pi * _DAY / 365 + np.pi), 0.0, None)
        return dict(
            sunshine_fraction_daily=md(np.full((N_DAYS, n), 0.5)),
            temperature_daily=md(np.tile((15.0 + 8.0 * _SEASON)[:, None], (1, n))),
            precipitation_daily=md(np.tile(rain[:, None], (1, n))),
            elevation=sd([100.0] * n),
            latitude=sd([40.0] * n),
            max_soil_moisture=sd(capacities),
            dates_daily=DAILY_DATES,
            soil_moisture_init_max_iter=MAX_ITER,
            soil_moisture_init_max_diff=MAX_DIFF,
        )

    def test_unreached_capacity_still_changes_soil_moisture(self):
        # Neither pixel fills, so the cap and runoff play no part: only the supply
        # rate can separate them. With a fixed 150 mm in the supply rate they tie.
        result = _splash(**self._pair([800.0, 1600.0]))
        moisture = result["soil_moisture_daily"].values
        assert moisture.max() < 800.0
        assert result["runoff_daily"].values.max() == 0.0
        assert moisture[:, 1].mean() > 1.2 * moisture[:, 0].mean()

    def test_workaround_is_still_needed(self):
        """Fails once pyrealm passes kWm on, when the workaround should be deleted."""
        inputs = self._pair([50.0, 400.0])
        direct = pyrealm.splash.splash.SplashModel(
            lat=inputs["latitude"].values[None, :],
            elv=inputs["elevation"].values[None, :],
            sf=inputs["sunshine_fraction_daily"].values,
            tc=inputs["temperature_daily"].values,
            pn=inputs["precipitation_daily"].values,
            dates=pyrealm.core.calendar.Calendar(inputs["dates_daily"].values),
            kWm=inputs["max_soil_moisture"].values,
        )
        np.testing.assert_array_equal(direct.evap.kWm, [150.0])


class TestPrescribedInitialState:
    """``initial_soil_moisture`` replaces the spin-up as the run's starting state."""

    def test_first_day_starts_from_the_given_state(self, splash_inputs):
        initial = _static(np.array([10.0, 60.0, 170.0]))
        result = _splash(**splash_inputs, initial_soil_moisture=initial)
        first_day = result["soil_moisture_daily"].isel(time=0)
        balance = (
            initial
            + splash_inputs["precipitation_daily"].isel(time=0)
            + result["condensation_daily"].isel(time=0)
            - result["actual_evapotranspiration_daily"].isel(time=0)
            - result["runoff_daily"].isel(time=0)
        )
        np.testing.assert_allclose(first_day.values, balance.values)

    def test_equilibrated_state_reproduces_the_default_run(self, splash_inputs):
        """Given what the spin-up would find, prescribing it changes nothing."""
        model = pyrealm.splash.splash.SplashModel(
            lat=splash_inputs["latitude"].values[None, :],
            elv=splash_inputs["elevation"].values[None, :],
            sf=splash_inputs["sunshine_fraction_daily"].values,
            tc=splash_inputs["temperature_daily"].values,
            pn=splash_inputs["precipitation_daily"].values,
            dates=pyrealm.core.calendar.Calendar(DAILY_DATES.values),
            kWm=splash_inputs["max_soil_moisture"].values,
        )
        model.evap.kWm = model.kWm
        spun_up = model.estimate_initial_soil_moisture(
            max_iter=MAX_ITER, max_diff=MAX_DIFF
        )
        default = _splash(**splash_inputs)["soil_moisture_daily"]
        prescribed = _splash(**splash_inputs, initial_soil_moisture=_static(spun_up))[
            "soil_moisture_daily"
        ]
        np.testing.assert_allclose(prescribed.values, default.values)

    def test_shorter_than_a_year_runs_without_a_spin_up(self, splash_inputs):
        inputs: dict = {
            key: value.isel(time=slice(30)) if "time" in value.dims else value
            for key, value in splash_inputs.items()
            if isinstance(value, xr.DataArray)
        }
        result = _splash(
            **inputs,
            dates_daily=DAILY_DATES[:30],
            initial_soil_moisture=_static(np.full(N_PIXELS, 50.0)),
        )
        assert result["soil_moisture_daily"].sizes["time"] == 30

    def test_state_above_capacity_is_rejected(self, splash_inputs):
        initial = _static(MAX_SM + 1.0)
        with pytest.raises(ValueError, match="between 0 and kWm"):
            _splash(**splash_inputs, initial_soil_moisture=initial)

    def _driver(self, **settings):
        from conduit import build_driver

        return build_driver(["satterc.models.splash"], settings)

    def test_equilibrate_needs_no_initial_state(self):
        names = {v.name for v in self._driver().list_available_variables()}
        assert "initial_soil_moisture" not in names

    def test_prescribed_requires_the_input(self):
        variables = self._driver(
            soil_moisture_init_method="prescribed"
        ).list_available_variables()
        needed = {v.name for v in variables if v.is_external_input}
        assert "initial_soil_moisture" in needed

    def test_pipeline_uses_the_prescribed_state(self, splash_inputs):
        inputs = {
            key: value
            for key, value in splash_inputs.items()
            if isinstance(value, xr.DataArray)
        }
        initial = _static(np.array([10.0, 60.0, 170.0]))
        dr = self._driver(soil_moisture_init_method="prescribed")
        piped = dr.execute(
            ["soil_moisture_daily"],
            inputs={**inputs, "initial_soil_moisture": initial},
        )  # type: ignore[reportArgumentType]
        direct = _splash(**splash_inputs, initial_soil_moisture=initial)
        np.testing.assert_allclose(
            piped["soil_moisture_daily"].values, direct["soil_moisture_daily"].values
        )

    def test_misspelt_method_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown soil_moisture_init_method"):
            splash_config(soil_moisture_init_method="prescibed")


class TestInvalidPixels:
    """A pixel's outcome comes from its own inputs, whatever else is in its block."""

    @staticmethod
    def _with_nan(inputs: dict, name: str, *where) -> dict:
        changed = dict(inputs)
        changed[name] = inputs[name].copy()
        changed[name][where] = np.nan
        return changed

    @staticmethod
    def _masked(inputs: dict, pixel: int) -> dict:
        return {
            key: value.where(value.pixel != pixel)
            if key.endswith("_daily") and isinstance(value, xr.DataArray)
            else value
            for key, value in inputs.items()
        }

    @staticmethod
    def _pixels(inputs: dict, pixels: list[int]) -> dict:
        return {
            key: value.isel(pixel=pixels) if isinstance(value, xr.DataArray) else value
            for key, value in inputs.items()
        }

    def test_masked_pixel_is_nan_and_leaves_the_others_alone(self, splash_inputs):
        result = _splash(**self._masked(splash_inputs, 1))
        alone = _splash(**self._pixels(splash_inputs, [0, 2]))
        for name in _OUTPUT_FIELDS:
            assert result[name].sel(pixel=1).isnull().all()
            np.testing.assert_array_equal(
                result[name].sel(pixel=[0, 2]).values, alone[name].values
            )

    @pytest.mark.parametrize("static", ["elevation", "latitude", "max_soil_moisture"])
    def test_nan_static_masks_the_pixel(self, splash_inputs, static, recwarn):
        result = _splash(**self._with_nan(splash_inputs, static, 1))
        assert result["soil_moisture_daily"].sel(pixel=1).isnull().all()
        assert result["soil_moisture_daily"].sel(pixel=[0, 2]).notnull().all()
        assert not [w for w in recwarn if "SPLASH" in str(w.message)]

    def test_nan_initial_state_masks_the_pixel(self, splash_inputs):
        initial = _static(np.array([10.0, np.nan, 170.0]))
        result = _splash(**splash_inputs, initial_soil_moisture=initial)
        assert result["soil_moisture_daily"].sel(pixel=1).isnull().all()
        assert result["soil_moisture_daily"].sel(pixel=[0, 2]).notnull().all()

    @pytest.mark.parametrize("pixels", [[0, 1, 2], [1], [1, 2]])
    def test_masked_pixels_do_not_raise_whatever_the_block(self, splash_inputs, pixels):
        inputs = self._pixels(self._masked(splash_inputs, 1), pixels)
        result = _splash(**inputs)
        assert result["soil_moisture_daily"].sel(pixel=1).isnull().all()

    def test_fully_masked_block_keeps_the_output_shape(self, splash_inputs):
        inputs: dict = {
            key: value * np.nan if key.endswith("_daily") else value
            for key, value in splash_inputs.items()
            if key != "dates_daily"
        }
        result = xr.Dataset(_splash(**inputs, dates_daily=DAILY_DATES))
        assert set(result.data_vars) == set(_OUTPUT_FIELDS)
        assert dict(result.sizes) == {"time": N_DAYS, "pixel": N_PIXELS}
        assert result.to_array().isnull().all()

    @pytest.mark.parametrize("pixels", [[0, 1, 2], [1], [1, 2]])
    @pytest.mark.parametrize("day", [100, 365])
    def test_gap_raises_naming_the_pixel_whatever_the_block(
        self, splash_inputs, pixels, day
    ):
        inputs = self._with_nan(splash_inputs, "precipitation_daily", day, 1)
        with pytest.raises(ValueError, match=r"missing values in 1 pixel\(s\) \(1\)"):
            _splash(**self._pixels(inputs, pixels))

    def test_one_variable_with_no_data_is_a_gap_not_a_mask(self, splash_inputs):
        inputs = dict(splash_inputs)
        inputs["temperature_daily"] = inputs["temperature_daily"] * np.nan
        with pytest.raises(ValueError, match=r"missing values in 3 pixel\(s\)"):
            _splash(**inputs)

    def test_many_pixels_are_truncated_in_the_message(self):
        labels = np.arange(25)
        assert (
            _describe_pixels(labels)
            == "25 pixel(s) (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...)"
        )

    def test_gap_uses_the_pixel_label_not_the_position(self, splash_inputs):
        inputs = self._with_nan(splash_inputs, "precipitation_daily", 100, 2)
        relabelled: dict = {
            key: value.assign_coords(pixel=[70, 71, 72])
            if isinstance(value, xr.DataArray)
            else value
            for key, value in inputs.items()
        }
        with pytest.raises(ValueError, match=r"\(72\)"):
            _splash(**relabelled)

    @pytest.mark.parametrize("pixels", [[0, 1], [1]])
    def test_model_failure_warns_and_keeps_going(self, splash_inputs, pixels):
        """Fails once pyrealm clips ``hi_pre`` at -1, when this pixel stops being NaN."""
        inputs = dict(splash_inputs)
        inputs["latitude"] = _static(np.array([51.5, -80.0, 10.0]))
        with pytest.warns(
            UserWarning, match=r"NaN soil moisture for 1 pixel\(s\) \(1\)"
        ):
            result = _splash(**self._pixels(inputs, pixels))
        moisture = result["soil_moisture_daily"]
        assert moisture.sel(pixel=1).isnull().all()
        if 0 in pixels:
            assert moisture.sel(pixel=0).notnull().all()

    def test_real_non_convergence_keeps_pyrealm_error(self, splash_inputs):
        inputs = {
            **splash_inputs,
            "soil_moisture_init_max_iter": 1,
            "soil_moisture_init_max_diff": 1e-12,
        }
        with pytest.raises(RuntimeError, match="did not converge"):
            _splash(**inputs)


class TestConsecutiveDays:
    """pyrealm steps by position, so a gap in the calendar must not reach it."""

    def _run(self, inputs: dict, dates: pd.DatetimeIndex) -> xr.Dataset:
        kept: dict = {
            key: value.sel(time=dates)
            if isinstance(value, xr.DataArray) and "time" in value.dims
            else value
            for key, value in inputs.items()
            if key != "dates_daily"
        }
        return xr.Dataset(_splash(**kept, dates_daily=dates))

    def test_missing_day_raises_naming_it(self, splash_inputs):
        dates = DAILY_DATES.delete(100)
        with pytest.raises(ValueError, match=r"1 break.*2020-04-09 to 2020-04-11"):
            self._run(splash_inputs, dates)

    def test_consecutive_subset_runs(self, splash_inputs):
        initial = _static(np.full(N_PIXELS, 50.0))
        result = self._run(
            {**splash_inputs, "initial_soil_moisture": initial}, DAILY_DATES[10:40]
        )
        assert result.sizes["time"] == 30


class TestStrategyCStaysOpen:
    """Chunked (dask-backed) ``pixel`` inputs give identical results to eager numpy."""

    @staticmethod
    def _chunked(inputs: dict) -> dict:
        return {
            key: value.chunk({"pixel": 1})
            if isinstance(value, xr.DataArray) and "pixel" in value.dims
            else value
            for key, value in inputs.items()
        }

    def test_chunked_pixel_equivalence(self, splash_inputs):
        pytest.importorskip("dask")
        eager = _splash(**splash_inputs)
        chunked = _splash(**self._chunked(splash_inputs))

        for name in _OUTPUT_FIELDS:
            out = chunked[name].compute().transpose("time", "pixel").values
            np.testing.assert_allclose(
                out, eager[name].transpose("time", "pixel").values
            )

    def test_chunked_gap_names_the_pixel_label(self, splash_inputs):
        pytest.importorskip("dask")
        inputs = dict(splash_inputs)
        inputs["precipitation_daily"] = inputs["precipitation_daily"].copy()
        inputs["precipitation_daily"][100, 2] = np.nan
        with pytest.raises(ValueError, match=r"1 pixel\(s\) \(2\)"):
            _splash(**self._chunked(inputs))["soil_moisture_daily"].compute()


class TestCachingIntact:
    """The cached splash node matches the uncached one (seam is internal to the node)."""

    def test_cached_run_matches_uncached(self, tmp_path, splash_inputs):
        from conduit import CacheSpec, build_driver

        inputs = {
            k: v
            for k, v in splash_inputs.items()
            if k not in ("soil_moisture_init_max_iter", "soil_moisture_init_max_diff")
        }
        spec = CacheSpec(path=str(tmp_path / "cache"))

        def run(cache):
            dr = build_driver(
                ["satterc.models.splash"],
                {
                    "soil_moisture_init_max_iter": MAX_ITER,
                    "soil_moisture_init_max_diff": MAX_DIFF,
                },
                cache=cache,
            )
            return dr.execute(["soil_moisture_daily"], inputs=inputs)  # type: ignore[reportArgumentType]

        uncached = run(None)
        run(spec)  # cold cache
        cached = run(spec)  # warm cache

        np.testing.assert_allclose(
            cached["soil_moisture_daily"].values,
            uncached["soil_moisture_daily"].values,
        )
        assert (tmp_path / "cache").exists()
