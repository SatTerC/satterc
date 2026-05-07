from pathlib import Path
from typing import Annotated

import typer

from ..config import load_config
from ..dag.driver import build_driver
from ..io import load_inputs, get_outputs, save_outputs

app = typer.Typer(help="Execute a pipeline defined in a configuration file.")


@app.command()
def run(
    config_file: Annotated[
        Path, typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True)
    ],
    allow_overrides: Annotated[
        bool,
        typer.Option(
            "--allow-overrides",
            help="Allow later modules to override earlier ones.",
        ),
    ] = False,
) -> None:
    """Execute a pipeline defined in a configuration file."""
    parsed = load_config(config_file)

    inputs = load_inputs(parsed.input_specs)

    dr = build_driver(
        modules=parsed.modules,
        config=parsed.driver_config,
        allow_module_overrides=allow_overrides,
    )

    if parsed.output_specs:
        target_vars = [
            var if freq == "static" else f"{var}_{freq}"
            for freq, spec in parsed.output_specs.items()
            for var in spec.vars
        ]
        results = dr.execute(target_vars, inputs=inputs)  # type: ignore[reportArgumentType]
        output_datasets = get_outputs(results, parsed.output_specs)
        save_outputs(output_datasets, parsed.output_specs)
