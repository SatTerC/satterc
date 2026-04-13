from pathlib import Path

import pytest
import xarray as xr

from satterc.config import load_config
from satterc.driver import build_driver
from satterc.pipeline.inputs import grid
from satterc.setup_utils.data_gen import generate_synthetic_data


TEST_CONFIG_PATH = Path(__file__).parent / "test_config.toml"

GRID = (2, 2)
N_DAYS = 365
SEED = 42


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory):
    """Generate synthetic data once per test session."""
    data_dir = tmp_path_factory.mktemp("synthetic_data")

    config = load_config(TEST_CONFIG_PATH)

    daily_path = str(data_dir / "daily.nc")
    weekly_path = str(data_dir / "weekly.nc")
    monthly_path = str(data_dir / "monthly.nc")
    static_path = str(data_dir / "static.nc")

    config["driver_config"]["daily_inputs_path"] = daily_path
    config["driver_config"]["weekly_inputs_path"] = weekly_path
    config["driver_config"]["monthly_inputs_path"] = monthly_path
    config["driver_config"]["static_inputs_path"] = static_path

    generate_synthetic_data(
        config=config,
        grid=GRID,
        n_days=N_DAYS,
        seed=SEED,
    )

    return data_dir


@pytest.fixture(scope="session")
def daily_ds(synthetic_data_dir):
    """Load daily synthetic dataset."""
    return xr.open_dataset(synthetic_data_dir / "daily.nc", decode_coords="all")


@pytest.fixture(scope="session")
def weekly_ds(synthetic_data_dir):
    """Load weekly synthetic dataset."""
    return xr.open_dataset(synthetic_data_dir / "weekly.nc", decode_coords="all")


@pytest.fixture(scope="session")
def monthly_ds(synthetic_data_dir):
    """Load monthly synthetic dataset."""
    return xr.open_dataset(synthetic_data_dir / "monthly.nc", decode_coords="all")


@pytest.fixture(scope="session")
def static_ds(synthetic_data_dir):
    """Load static synthetic dataset."""
    return xr.open_dataset(synthetic_data_dir / "static.nc", decode_coords="all")


@pytest.fixture(scope="session")
def common_grid_ds(daily_ds, weekly_ds, monthly_ds, static_ds):
    """Compute the common grid once per session."""
    return grid.common_grid(
        loaded_daily_inputs=daily_ds,
        loaded_weekly_inputs=weekly_ds,
        loaded_monthly_inputs=monthly_ds,
        loaded_static_inputs=static_ds,
    )


@pytest.fixture(scope="session")
def stacked_grid_ds(common_grid_ds):
    """Compute the stacked grid once per session."""
    return grid.stacked_grid(common_grid_ds)


@pytest.fixture(scope="session")
def pipeline_config(synthetic_data_dir):
    """Load test config with all paths pointing to the synthetic data dir."""
    config = load_config(TEST_CONFIG_PATH)
    config["driver_config"]["daily_inputs_path"] = str(synthetic_data_dir / "daily.nc")
    config["driver_config"]["weekly_inputs_path"] = str(
        synthetic_data_dir / "weekly.nc"
    )
    config["driver_config"]["monthly_inputs_path"] = str(
        synthetic_data_dir / "monthly.nc"
    )
    config["driver_config"]["static_inputs_path"] = str(
        synthetic_data_dir / "static.nc"
    )
    config["driver_config"]["daily_outputs_path"] = str(
        synthetic_data_dir / "out_daily.nc"
    )
    config["driver_config"]["weekly_outputs_path"] = str(
        synthetic_data_dir / "out_weekly.nc"
    )
    config["driver_config"]["monthly_outputs_path"] = str(
        synthetic_data_dir / "out_monthly.nc"
    )
    return config


@pytest.fixture(scope="session")
def pipeline_driver(pipeline_config):
    """Build Hamilton driver for integration tests."""
    return build_driver(
        pipeline_config["modules"],
        pipeline_config["driver_config"],
    )
