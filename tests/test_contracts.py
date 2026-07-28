"""Generated data must satisfy the contracts the models declare.

The rest of the suite runs with contract validation off (see `conftest`): the
fixtures are deliberately tiny, and a two-timestep array has no inferable
frequency. That leaves nothing checking that what `data-gen` writes actually
matches the units and frequencies the model modules declare — a table row with
the wrong units would pass every other test and only fail in a real run.

This module closes that gap, at the cost of being the slow test: it generates a
full two years over the example config — the real four-model pipeline — and
executes it with validation turned on.
"""

from pathlib import Path

import pytest
from conduit import ParsedConfig, build_driver, get_final_vars, load_config, load_inputs
from conduit.specs import AnnotationPolicySpec

from satterc.setup_utils.data_gen import generate_synthetic_data

EXAMPLE_CONFIG = Path(__file__).parent.parent / "examples" / "config.toml"

# Two years, because RothC needs a year to equilibrate and frequency inference
# needs enough timesteps to have something to infer from.
GRID = (2, 2)
N_DAYS = 730


@pytest.fixture(scope="module")
def validated_config(tmp_path_factory) -> ParsedConfig:
    """The example config, pointed at freshly generated data in a temp dir."""
    data_dir = tmp_path_factory.mktemp("contract_data")
    config = load_config(EXAMPLE_CONFIG)

    for label, spec in config.input_specs.items():
        spec.path = str(data_dir / f"{label}{Path(spec.path).suffix}")

    generate_synthetic_data(config=config, grid=GRID, n_days=N_DAYS, seed=42)
    return config


@pytest.fixture
def strict_contracts():
    """Turn contract validation on, and put it back afterwards.

    `AnnotationPolicySpec.apply` writes to xarray-annotated's *process-global*
    policy, so this has to be restored or it leaks into whatever runs next.

    ``on_inexact`` stays at ``"convert"`` rather than ``"error"``: the example
    config relies on pint converting SGAM's ``g m-2`` pools into RothC's
    ``t ha-1``, which is by design. Converting *correctly labelled* units is
    right, so a row whose values are simply mislabelled is beyond what any
    contract check can catch. What this does catch is a unit that is not
    dimensionally convertible to the declared one, a missing unit, and data
    written at the wrong frequency — the three ways a new table row realistically
    goes wrong.
    """
    AnnotationPolicySpec(
        enabled=True,
        on_missing="error",
        on_inexact="convert",
        on_mismatch="error",
        on_uninferable="error",
    ).apply()
    try:
        yield
    finally:
        AnnotationPolicySpec(enabled=False).apply()


@pytest.mark.usefixtures("strict_contracts")
class TestGeneratedDataSatisfiesContracts:
    def test_driver_builds(self, validated_config):
        """The build-time check compares every declared contract across the graph."""
        build_driver(
            validated_config.modules,
            validated_config.driver_config,
            node_specs=validated_config.node_specs,
        )

    def test_pipeline_executes(self, validated_config):
        """Validation of the real arrays: units and frequency of what was written."""
        driver = build_driver(
            validated_config.modules,
            validated_config.driver_config,
            node_specs=validated_config.node_specs,
        )
        inputs = load_inputs(validated_config.input_specs)
        final_vars: list = get_final_vars(validated_config.output_specs)
        results = driver.execute(final_vars, inputs=inputs)
        assert set(results) == set(final_vars)
