"""Tests for the ACM-GPP-ET f2py seam (``satterc.models.acm_gpp_et``).

Skipped unless the extension has been built with ``just build-acm``.

The Fortran keeps module state sized by the first call's series length for the
rest of the process, so every test here uses `N_DAYS`, including the one that
checks a different length is refused.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("satterc.models._acm_gpp_et")

from satterc.models._acm_gpp_et import acm_pixel
from satterc.models.acm_gpp_et import (
    _ACM_OUTPUT_NAMES,
    _acm_gpp_et,
    _acm_gpp_et_1px,
    acm_gpp_et_config,
)

N_DAYS = 365
N_PIXELS = 3
DATES = pd.date_range("2020-01-01", periods=N_DAYS, freq="D")
PIXELS = np.arange(N_PIXELS)

LATITUDE = np.array([49.0, 55.0, 40.0])
CLAY = np.array([15.0, 30.0, 5.0])
SAND = np.array([40.0, 20.0, 70.0])


def _daily(values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        values, dims=["time", "pixel"], coords={"time": DATES, "pixel": PIXELS}
    )


def _static(values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(values, dims=["pixel"], coords={"pixel": PIXELS})


@pytest.fixture(scope="module")
def acm_inputs() -> dict:
    """Seasonal drivers that differ per pixel, plus distinct soils."""
    rng = np.random.default_rng(0)
    doy = DATES.day_of_year.to_numpy()  # type: ignore[reportAttributeAccessIssue]
    season = np.repeat(np.sin(2 * np.pi * (doy - 100) / 365)[:, None], N_PIXELS, 1)
    offset = np.linspace(-3.0, 3.0, N_PIXELS)[None, :]
    mean_t = 10.0 + 8.0 * season + offset + rng.normal(0, 1.5, (N_DAYS, N_PIXELS))
    lai = np.clip(3.0 + 2.0 * season + offset / 3, 0.2, None)
    return dict(
        temperature_min_daily=_daily(mean_t - 4.0),
        temperature_max_daily=_daily(mean_t + 4.0),
        temperature_daily=_daily(mean_t),
        shortwave_radiation_daily=_daily(np.clip(12.0 + 10.0 * season, 1.0, None)),
        co2_daily=_daily(np.full((N_DAYS, N_PIXELS), 400.0)),
        precipitation_daily=_daily(
            np.where(rng.uniform(size=(N_DAYS, N_PIXELS)) < 0.5, 4.0, 0.0)
        ),
        wind_speed_daily=_daily(np.full((N_DAYS, N_PIXELS), 3.0)),
        vpd_daily=_daily(np.clip(600.0 + 400.0 * season, 50.0, None)),
        lai_daily=_daily(lai),
        fine_root_carbon_daily=_daily(lai * 80.0),
        latitude=_static(LATITUDE),
        clay_content=_static(CLAY),
        sand_content=_static(SAND),
        dates_daily=DATES,
        **acm_gpp_et_config(),
    )


@pytest.fixture(scope="module")
def seam_result(acm_inputs):
    return _acm_gpp_et(**acm_inputs)


def _reference_1px(inputs: dict, i: int) -> dict[str, np.ndarray]:
    """Call the extension directly for pixel ``i``, building ``met`` by hand."""

    def col(name: str) -> np.ndarray:
        return inputs[name].values[:, i]

    met = np.full((12, N_DAYS), -9999.0)
    met[0] = np.arange(1, N_DAYS + 1)
    met[1] = col("temperature_min_daily")
    met[2] = col("temperature_max_daily")
    met[3] = col("shortwave_radiation_daily")
    met[4] = col("co2_daily")
    met[5] = DATES.day_of_year  # type: ignore[reportAttributeAccessIssue]
    met[6] = col("precipitation_daily") / 86400
    met[7] = col("temperature_daily")
    met[8] = col("wind_speed_daily")
    met[9] = col("vpd_daily")
    met[10] = col("lai_daily")
    met[11] = col("fine_root_carbon_daily")
    cfg = acm_gpp_et_config()
    pars = np.array(list(cfg.values()))
    fluxes, soil_water, wswp, status = acm_pixel(
        met, pars, LATITUDE[i], np.full(4, CLAY[i]), np.full(4, SAND[i])
    )
    assert status == 0
    columns = [*fluxes.T, soil_water, wswp]
    return dict(zip(_ACM_OUTPUT_NAMES, columns, strict=True))


class TestSeam:
    @pytest.mark.parametrize("i", range(N_PIXELS))
    def test_matches_direct_call(self, acm_inputs, seam_result, i):
        reference = _reference_1px(acm_inputs, i)
        for name in _ACM_OUTPUT_NAMES:
            np.testing.assert_array_equal(
                seam_result[name].isel(pixel=i).values, reference[name]
            )

    def test_canonical_dims_and_coords(self, seam_result):
        for da in seam_result.values():
            assert da.dims == ("time", "pixel")
            np.testing.assert_array_equal(da.coords["time"].values, DATES.values)

    def test_plausible_and_pixel_dependent(self, seam_result):
        gpp = seam_result["canopy_gpp_daily"].values
        assert np.all(np.isfinite(gpp))
        assert 0.5 < gpp.mean() < 20.0  # g C m-2 d-1, a temperate canopy
        assert not np.allclose(gpp[:, 0], gpp[:, 1])

    def test_repeat_call_is_identical(self, acm_inputs, seam_result):
        # Module state must be fully reset between calls, or the second pixel of
        # a block would inherit the first's soil water, snow, canopy storage...
        again = _acm_gpp_et(**acm_inputs)
        for name in _ACM_OUTPUT_NAMES:
            np.testing.assert_array_equal(again[name].values, seam_result[name].values)


def test_different_length_is_refused(acm_inputs, seam_result):
    # seam_result guarantees the module state is already sized to N_DAYS.
    def short(name: str) -> np.ndarray:
        return acm_inputs[name].values[:100, 0]

    with pytest.raises(RuntimeError, match="different length"):
        _acm_gpp_et_1px(
            temperature_min=short("temperature_min_daily"),
            temperature_max=short("temperature_max_daily"),
            temperature=short("temperature_daily"),
            shortwave_radiation=short("shortwave_radiation_daily"),
            co2=short("co2_daily"),
            precipitation=short("precipitation_daily"),
            wind_speed=short("wind_speed_daily"),
            vpd=short("vpd_daily"),
            lai=short("lai_daily"),
            fine_root_carbon=short("fine_root_carbon_daily"),
            latitude=LATITUDE[0],
            clay_content=CLAY[0],
            sand_content=SAND[0],
            run_day=np.arange(1.0, 101.0),
            day_of_year=np.arange(1.0, 101.0),
            pars=np.array([1.89, -1.99, 100.0, 2.0, 9.0]),
        )


def test_runs_in_a_driver(acm_inputs):
    """The registered section name resolves and the node runs in a DAG."""
    from conduit import build_driver
    from conduit.importing import registered_module

    module = registered_module("acm_gpp_et")
    assert module is not None
    inputs = {
        k: v
        for k, v in acm_inputs.items()
        if isinstance(v, xr.DataArray) or k == "dates_daily"
    }
    dr = build_driver([module.import_path], {}, registered=[module])
    out = dr.execute(["canopy_gpp_daily"], inputs=inputs)  # type: ignore[reportArgumentType]
    assert out["canopy_gpp_daily"].sizes == {"time": N_DAYS, "pixel": N_PIXELS}
