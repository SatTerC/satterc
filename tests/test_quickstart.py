"""The scaffolding route has to actually produce a pipeline that runs.

`satterc setup` → `satterc data-gen` → `satterc run` is what the quickstart tells
a new user to do, and it is the only path where the two scaffolding halves are
exercised against each other: the config generator decides what a model needs,
and the data generator has to be able to fabricate it.

Nothing else covers that seam. `test_contracts.py` runs `examples/config.toml`,
which is hand-written — so it validates the models, not the generator. A gap in
the generator (a resample it fails to emit, a variable it leaves unbound, a rate
it coarsens as a mean) shows up here and nowhere else.
"""

import numpy as np
import pytest
import xarray as xr
from typer.testing import CliRunner

from satterc.cli import app

MODELS = "splash,pmodel,sgam,rothc"
GRID = ("2", "2")
DURATION = "2y"


@pytest.fixture(scope="module")
def quickstart(tmp_path_factory):
    """Run the documented route end to end, returning the working directory."""
    workdir = tmp_path_factory.mktemp("quickstart")
    (workdir / "outputs").mkdir()
    runner = CliRunner()

    generated = runner.invoke(
        app, ["setup", "-m", MODELS, "-d", "-o", str(workdir / "config.toml")]
    )
    assert generated.exit_code == 0, generated.output

    config = workdir / "config.toml"
    data = runner.invoke(
        app,
        [
            "data-gen",
            "generate",
            str(config),
            "--grid",
            *GRID,
            "--duration",
            DURATION,
        ],
    )
    assert data.exit_code == 0, data.output
    return workdir


class TestQuickstart:
    def test_dry_run_passes(self, quickstart):
        """Catches an unbound input or a contract mismatch without computing."""
        result = CliRunner().invoke(
            app, ["run", str(quickstart / "config.toml"), "--dry-run"]
        )
        assert result.exit_code == 0, result.output

    def test_pipeline_runs_and_writes_every_section(self, quickstart):
        result = CliRunner().invoke(app, ["run", str(quickstart / "config.toml")])
        assert result.exit_code == 0, result.output
        for section in ("daily", "weekly", "monthly"):
            assert (quickstart / "outputs" / f"{section}.nc").exists()

    def test_outputs_are_finite(self, quickstart):
        """A NaN here means synthetic data drove a model outside its domain."""
        CliRunner().invoke(app, ["run", str(quickstart / "config.toml")])
        for section in ("daily", "weekly", "monthly"):
            dataset = xr.open_dataset(quickstart / "outputs" / f"{section}.nc")
            for name, values in dataset.data_vars.items():
                assert not np.isnan(values.values).any(), f"{section}.{name} has NaN"

    def test_soil_carbon_is_physically_plausible(self, quickstart):
        """RothC integrates over the whole run, so it fails loudly if its drivers
        are nonsense — it is the most sensitive check available here."""
        CliRunner().invoke(app, ["run", str(quickstart / "config.toml")])
        monthly = xr.open_dataset(quickstart / "outputs" / "monthly.nc")
        soc = monthly.soil_organic_carbon.values
        assert (soc > 0).all()
        assert 10.0 < soc.mean() < 1000.0


class TestGeneratedConfigShape:
    """What the generator must emit for the four-model pipeline to wire up."""

    def test_gpp_is_an_input_despite_being_a_model_output(self, quickstart):
        """pmodel produces `gpp_weekly`; sgam's disturbances want `gpp_daily`.

        Resampling only coarsens, so the weekly output cannot supply the daily
        consumer and it has to be loaded from a file.
        """
        config = (quickstart / "config.toml").read_text()
        assert '"gpp",' in config.split("[inputs.weekly]")[0]

    def test_model_outputs_are_resampled_for_coarser_consumers(self, quickstart):
        """sgam produces `disturbances_daily` and consumes `disturbances_weekly`."""
        config = (quickstart / "config.toml").read_text()
        assert '"disturbances",' in config

    def test_rates_are_accumulated_not_averaged(self, quickstart):
        """`mm d-1` onto RothC's `mm` is a total: a summing node, not a resample.

        A `[[resample]]` preserves units, so coarsening this one as a mean would
        hand RothC a rate labelled as a total.
        """
        config = (quickstart / "config.toml").read_text()
        for name in ("precipitation_monthly", "potential_evapotranspiration_monthly"):
            assert f'name = "{name}"' in config
            assert f"{name.rsplit('_', 1)[0]}_daily.resample" in config
