"""The models' declared `Freq` contracts must hold against real pipeline data.

The model wrappers no longer receive a ``dates_*`` node; each reads its calendar
off a time-bearing input instead, and declares the frequency it expects with a
`xarray_annotated.temporal.Freq` marker. Those declarations are load-bearing in
two places — conduit's build-time contract check, and the frequency clustering in
``satterc graph`` — so they need to be right, not merely present.

The rest of the suite runs with validation switched off (see conftest), so every
check here passes ``enabled=True`` explicitly — without it the policy context
manager would leave the master switch off and the assertions would hold
vacuously.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from conduit import build_driver
from xarray_annotated.temporal import FreqError, check_freq, policy

from satterc import frequencies as _time
from satterc.models import pmodel, rothc, sgam, splash

MODULES = {
    "splash": splash,
    "pmodel": pmodel,
    "sgam": sgam,
    "rothc": rothc,
}


def _series(dates):
    return xr.DataArray(
        np.zeros((len(dates), 1)),
        dims=["time", "pixel"],
        coords={"time": dates, "pixel": [0]},
    )


class TestDeclaredOffsets:
    """The offsets satterc declares are the ones its resampling produces."""

    @pytest.mark.parametrize(
        ("marker", "dates"),
        [
            (_time.DAILY, pd.date_range("2020-01-01", periods=90, freq="D")),
            (_time.WEEKLY, pd.date_range("2020-01-01", periods=30, freq="7D")),
            (_time.MONTHLY, pd.date_range("2020-01-31", periods=24, freq="1ME")),
        ],
    )
    def test_marker_accepts_its_own_frequency(self, marker, dates):
        with policy(enabled=True, on_mismatch="error"):
            check_freq(_series(dates), marker, "series")

    def test_weekly_marker_is_unanchored(self):
        # A resample landing on a Wednesday is still weekly; pinning the phase
        # would reject a perfectly good pipeline.
        dates = pd.date_range("2020-01-01", periods=30, freq="W-WED")
        with policy(enabled=True, on_mismatch="error"):
            check_freq(_series(dates), _time.WEEKLY, "series")

    def test_daily_marker_rejects_weekly_data(self):
        dates = pd.date_range("2020-01-01", periods=30, freq="7D")
        with policy(enabled=True, on_mismatch="error"), pytest.raises(FreqError):
            check_freq(_series(dates), _time.DAILY, "series")


class TestModelContractsAgainstPipelineData:
    """Every model's contracts are checked over the DAG built from the test config."""

    @pytest.mark.parametrize("name", sorted(MODULES))
    def test_build_driver_passes_contract_check(self, name, pipeline_config):
        # build_driver runs conduit's whole-DAG contract check, which compares
        # every internal edge where both ends declare units, dims, dtype or freq.
        # A model declaring a frequency its own upstream node contradicts fails
        # here, before any compute.
        with policy(enabled=True, on_mismatch="error"):
            build_driver(
                [f"satterc.models.{name}"],
                pipeline_config.driver_config,
            )

    def test_all_models_together_pass_contract_check(self, pipeline_config):
        with policy(enabled=True, on_mismatch="error"):
            build_driver(
                [f"satterc.models.{name}" for name in sorted(MODULES)],
                pipeline_config.driver_config,
            )


class TestTimeIndexHelper:
    def test_reads_the_sole_time_coordinate(self):
        dates = pd.date_range("2020-01-01", periods=12, freq="1ME")
        result = _time.time_index(_series(dates), "series")
        pd.testing.assert_index_equal(result, pd.DatetimeIndex(dates))

    def test_rejects_an_array_with_no_time_dimension(self):
        static = xr.DataArray([1.0, 2.0], dims=["pixel"], coords={"pixel": [0, 1]})
        with pytest.raises(ValueError, match="time"):
            _time.time_index(static, "static")
