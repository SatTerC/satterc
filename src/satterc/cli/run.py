from pathlib import Path
from typing import Annotated

import typer

from ..config import load_config
from ..driver import build_driver

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

    dr = build_driver(
        modules=parsed["modules"],
        config=parsed["driver_config"],
        allow_module_overrides=allow_overrides,
    )

    dr.execute(parsed["targets"])
