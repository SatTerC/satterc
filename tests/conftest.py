from pathlib import Path

import pytest
import xarray as xr
from conduit import build_driver, load_config, load_inputs
from conduit.specs import AnnotationPolicySpec

from satterc.scaffold.data_gen import generate_synthetic_data

# The fixtures build DAGs and run models over deliberately small, synthetic
# arrays. Contract validation is conduit's job and is exercised by conduit's own
# suite; here it would only add noise (and a two-timestep fixture cannot have an
# inferable frequency at all), so the package-wide switch is off.
AnnotationPolicySpec(enabled=False).apply()

TEST_CONFIG_PATH = Path(__file__).parent / "test_config.toml"

GRID = (2, 2)
N_DAYS = 365
SEED = 42


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory):
    """Generate synthetic data once per test session."""
    data_dir = tmp_path_factory.mktemp("synthetic_data")

    config = load_config(TEST_CONFIG_PATH)

    config.input_specs["daily"].path = str(data_dir / "daily.nc")
    config.input_specs["weekly"].path = str(data_dir / "weekly.nc")
    config.input_specs["monthly"].path = str(data_dir / "monthly.nc")
    config.input_specs["static"].path = str(data_dir / "static.nc")

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
def pipeline_config(synthetic_data_dir):
    """Load test config with all paths pointing to the synthetic data dir."""
    config = load_config(TEST_CONFIG_PATH)
    config.input_specs["daily"].path = str(synthetic_data_dir / "daily.nc")
    config.input_specs["weekly"].path = str(synthetic_data_dir / "weekly.nc")
    config.input_specs["monthly"].path = str(synthetic_data_dir / "monthly.nc")
    config.input_specs["static"].path = str(synthetic_data_dir / "static.nc")
    return config


@pytest.fixture(scope="session")
def pipeline_inputs(pipeline_config):
    """Load all inputs using conduit's load_inputs()."""
    return load_inputs(pipeline_config.input_specs)


def driver_from_config(config, **kwargs):
    """Build a Hamilton driver from a `ParsedConfig`.

    Everything conduit needs to resolve a module has to be forwarded: `base` is
    the directory a relative `.py` `_import_path` resolves against, and
    `registered_modules` names the sections that came from an entry-point
    registration rather than an `_import_path`. satterc's own sections are the
    latter, so a driver built without `registered` cannot report where a failing
    module came from.
    """
    return build_driver(
        config.modules,
        config.driver_config,
        node_specs=config.node_specs,
        base=config.base,
        registered=config.registered_modules,
        **kwargs,
    )


@pytest.fixture(scope="session")
def pipeline_driver(pipeline_config):
    """Build Hamilton driver for integration tests."""
    return driver_from_config(pipeline_config)
