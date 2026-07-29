import sys
from pathlib import Path

import pytest
import xarray as xr
from conduit import build_driver, load_config, load_inputs
from conduit.specs import AnnotationPolicySpec

from satterc.setup_utils.data_gen import generate_synthetic_data

# The fixtures build DAGs and run models over deliberately small, synthetic
# arrays. Contract validation is conduit's job and is exercised by conduit's own
# suite; here it would only add noise (and a two-timestep fixture cannot have an
# inferable frequency at all), so the package-wide switch is off.
AnnotationPolicySpec(enabled=False).apply()

TEST_CONFIG_PATH = Path(__file__).parent / "test_config.toml"

#: Every conduit `[[node]]` fails to build on Python 3.14.
#:
#: conduit generates a node's function body by `exec` with no return annotation,
#: then injects the declared contract with `fn.__annotations__["return"] = ...`
#: (conduit/dag/node.py). xarray-annotated's `declare_units` / `declare_freq` /
#: `declare_schema` then wrap it with `functools.wraps`. On 3.13 that copies the
#: mutated `__annotations__` dict; on 3.14, PEP 749 swapped `__annotations__` for
#: `__annotate__` in `functools.WRAPPER_ASSIGNMENTS`, so the wrapper inherits the
#: *original compiled* annotate function and the injected return hint is dropped.
#: Hamilton then rejects the node with "Missing type hint for return value".
#:
#: Nothing in satterc can work around this — conduit builds the node module. The
#: mark is `strict`, so it fails loudly once upstream is fixed and can be removed.
NODES_BROKEN_ON_PY314 = pytest.mark.xfail(
    sys.version_info >= (3, 14),
    reason=(
        "conduit [[node]] lowering loses its injected return annotation on "
        "Python 3.14: functools.wraps copies __annotate__ rather than "
        "__annotations__ (PEP 749). Upstream bug; see the conduit issue."
    ),
    raises=ValueError,
    strict=True,
)

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


@pytest.fixture(scope="session")
def pipeline_driver(pipeline_config):
    """Build Hamilton driver for integration tests."""
    return build_driver(
        pipeline_config.modules,
        pipeline_config.driver_config,
        node_specs=pipeline_config.node_specs,
    )
