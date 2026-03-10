from pathlib import Path
import subprocess
import tomllib
from typing import Annotated

import typer
from hamilton import graph_types
import xarray as xr

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
    output: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="Path to output directory",
        ),
    ] = Path("./outputs"),
    allow_overrides: Annotated[
        bool,
        typer.Option(
            "--allow-overrides",
            help="Allow later modules to override earlier ones.",
        ),
    ] = False,
) -> None:
    """Execute a pipeline defined in a configuration file."""
    typer.secho("Not yet implemented!", fg=typer.colors.YELLOW)

    with config_file.open("rb") as file:
        config = tomllib.load(file)

    modules = config.get("modules", None)
    driver_config = config.get("config", None)

    # TODO: dynamically create node that combines outputs into a dataset and saves to disk.
    dr = build_driver(
        modules=modules,
        config=driver_config,
        allow_module_overrides=allow_overrides,
    )

    output_paths = {
        "daily_outputs_path": output / "daily.nc",
        "weekly_outputs_path": output / "weekly.nc",
        "monthly_outputs_path": output / "monthly.nc",
    }

    dr.execute(
        ["daily_outputs_file", "weekly_outputs_file", "monthly_outputs_file"],
        inputs=output_paths,
    )


# TODO: refine this and move out of cli
def custom_style(
    *, node: graph_types.HamiltonNode, node_class: str
) -> tuple[dict, str | None, str | None]:
    """Custom style function for the visualization."""
    if node.tags.get("module") == "satterc.inputs.static":
        style = ({"fillcolor": "aquamarine"}, node_class, "static inputs")

    elif node.type is xr.DataArray and "_daily" in node.name:
        style = ({"fillcolor": "orange"}, node_class, "Daily")

    elif node.type is xr.DataArray and "_weekly" in node.name:
        style = ({"fillcolor": "yellow"}, node_class, "Weekly")

    elif node.type is xr.DataArray and "_monthly" in node.name:
        style = ({"fillcolor": "brown"}, node_class, "Monthly")

    else:
        style = ({}, node_class, None)

    return style


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
    png: Annotated[bool, typer.Option(help="Convert to PNG format")] = False,
    pdf: Annotated[bool, typer.Option(help="Convert to PDF format")] = False,
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
        output_file_path=str(output_path),
        graphviz_kwargs=graphviz_kwargs,
        custom_style_function=custom_style,
    )
    # dr.display_upstream_of(
    #    "soil_organic_carbon_monthly",
    #    output_file_path=str(output_path),
    #    graphviz_kwargs=graphviz_kwargs,
    # )

    # TODO: is there a better way than this?
    if png:
        subprocess.run(
            [
                "dot",
                "-Tpng",
                str(output_path),
                "-o",
                str(output_path.with_suffix(".png")),
            ]
        )
    if pdf:
        subprocess.run(
            [
                "dot",
                "-Tpdf",
                str(output_path),
                "-o",
                str(output_path.with_suffix(".pdf")),
            ]
        )


@app.command()
def version() -> None:
    typer.echo(f"satterc version {__version__}")


def main() -> None:
    app()
