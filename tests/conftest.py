from pathlib import Path

import pytest
import xarray as xr

from satterc.config import load_config
from satterc.setup_utils.data_gen import generate_synthetic_data


TEST_CONFIG_PATH = Path(__file__).parent / "test_config.toml"

GRID = (2, 2)
N_DAYS = 365
SEED = 42


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory):
    """Generate synthetic data once per test session.

    Creates test data in a temporary directory that's automatically
    cleaned up after the test session ends.
    """
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


@pytest.fixture
def daily_ds(synthetic_data_dir):
    """Load daily synthetic dataset."""
    return xr.open_dataset(synthetic_data_dir / "daily.nc", decode_coords="all")


@pytest.fixture
def weekly_ds(synthetic_data_dir):
    """Load weekly synthetic dataset."""
    return xr.open_dataset(synthetic_data_dir / "weekly.nc", decode_coords="all")


@pytest.fixture
def monthly_ds(synthetic_data_dir):
    """Load monthly synthetic dataset."""
    return xr.open_dataset(synthetic_data_dir / "monthly.nc", decode_coords="all")


@pytest.fixture
def static_ds(synthetic_data_dir):
    """Load static synthetic dataset."""
    return xr.open_dataset(synthetic_data_dir / "static.nc", decode_coords="all")
