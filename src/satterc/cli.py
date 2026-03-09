from pathlib import Path
import subprocess
import tomllib
from typing import Annotated

import typer

from .driver import build_driver
from ._version import __version__

app = typer.Typer(
    help="Command-line interface for the SatTerC framework.",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command()
def run(
    config_file: Annotated[
        Path, typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True)
    ],
) -> None:
    """Execute a pipeline defined in a configuration file."""
    typer.secho("Not yet implemented!", fg=typer.colors.YELLOW)


@app.command()
def graph(
    config_file: Annotated[
        Path, typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True)
    ],
    output: Annotated[
        str, typer.Option("-o", "--output", help="Name of output file")
    ] = "pipeline",
    allow_overrides: Annotated[
        bool,
        typer.Option(
            "--allow-overrides",
            help="Allow later modules to override earlier ones.",
        ),
    ] = False,
) -> None:
    """Visualise a pipeline defined in a configuration file.

    Attention
    ---------
    This requires graphviz to be installed.
    """
    with config_file.open("rb") as file:
        config = tomllib.load(file)

    modules = config.get("modules", None)
    driver_config = config.get("config", None)
    graphviz_kwargs = config.get("graphviz", None)

    dr = build_driver(
        modules=modules,
        config=driver_config,
        allow_module_overrides=allow_overrides,
    )

    output_path = Path(output).with_suffix(".dot")

    dr.display_all_functions(
        output_file_path=str(output_path), graphviz_kwargs=graphviz_kwargs
    )

    # TODO: is there a better way than this?
    subprocess.run(
        ["dot", "-Tpng", str(output_path), "-o", str(output_path.with_suffix(".png"))]
    )


@app.command()
def version() -> None:
    typer.echo(f"satterc version {__version__}")


def main() -> None:
    app()
