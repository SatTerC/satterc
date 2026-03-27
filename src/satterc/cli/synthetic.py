from pathlib import Path
import tomllib
import re
from typing import Annotated

import typer


app = typer.Typer(help="Generate synthetic input data for testing.")


DURATION_PATTERN = re.compile(r"^(\d+)([ymd])$")


def _parse_duration(duration: str) -> int:
    match = DURATION_PATTERN.match(duration.lower())
    if not match:
        raise typer.BadParameter(
            f"Invalid duration format: '{duration}'. Expected format like '2y', '6m', '30d'."
        )
    value, unit = match.groups()
    value = int(value)
    if unit == "d":
        return value
    elif unit == "m":
        return int(value * 30.44)
    elif unit == "y":
        return int(value * 365.25)


@app.command()
def synthetic(
    config_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to TOML configuration file.",
        ),
    ],
    grid: Annotated[
        tuple[int, int],
        typer.Option(
            "--grid",
            "-g",
            help="Grid dimensions as 'n_lat,n_lon'.",
        ),
    ] = (1, 1),
    duration: Annotated[
        str,
        typer.Option(
            "--duration",
            "-d",
            help="Time duration (e.g., '2y' for 2 years, '6m' for 6 months, '30d' for 30 days).",
        ),
    ] = "2y",
) -> None:
    """Generate synthetic input data for Hamilton DAG testing."""
    n_lat, n_lon = grid
    if n_lat <= 0 or n_lon <= 0:
        raise typer.BadParameter("Grid dimensions must be positive integers.")

    n_days = _parse_duration(duration)

    with config_file.open("rb") as f:
        config = tomllib.load(f)

    typer.echo(f"Config file: {config_file}")
    typer.echo(f"Grid dimensions: {n_lat} x {n_lon}")
    typer.echo(f"Duration: {duration} ({n_days} days)")

    inputs = config.get("inputs", {})
    typer.echo(f"Input sections found: {list(inputs.keys())}")

    typer.echo("\n[TODO] Implement synthetic data generation logic here.")
