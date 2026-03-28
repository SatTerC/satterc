from pathlib import Path
import re
from typing import Annotated

import typer
from typer import Abort

from ..config import load_config
from ..synthetic_data import generate_synthetic_data


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


def _validate_output_paths(
    config: dict,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Validate/create directories, prompt for overwrite if files exist. Returns paths."""
    frequencies = ["daily", "weekly", "monthly", "static"]

    paths = []
    dirs_to_create = []
    files_to_overwrite = []

    for freq in frequencies:
        path_str = config["driver_config"].get(f"{freq}_inputs_path")
        if path_str is None:
            continue

        path = Path(path_str)
        paths.append(path)

        if not path.parent.exists():
            dirs_to_create.append(path.parent)
        elif path.is_file():
            files_to_overwrite.append(path)

    return paths, dirs_to_create, files_to_overwrite


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
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            "-s",
            help="Random seed for reproducibility.",
        ),
    ] = 42,
) -> None:
    """Generate synthetic input data for Hamilton DAG testing."""
    n_lat, n_lon = grid
    if n_lat <= 0 or n_lon <= 0:
        raise typer.BadParameter("Grid dimensions must be positive integers.")

    n_days = _parse_duration(duration)

    config = load_config(config_file)

    paths, dirs_to_create, files_to_overwrite = _validate_output_paths(config)

    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

    if dirs_to_create:
        typer.echo(f"Created directories: {', '.join(str(d) for d in dirs_to_create)}")

    if files_to_overwrite:
        typer.echo(
            f"Files already exist: {', '.join(str(p) for p in files_to_overwrite)}"
        )
        if not typer.confirm("Overwrite existing files?", default=False):
            raise Abort()

    typer.echo("Generating synthetic data:")
    typer.echo(f"  Config file: {config_file}")
    typer.echo(f"  Grid dimensions: {n_lat} x {n_lon}")
    typer.echo(f"  Duration: {duration} ({n_days} days)")
    typer.echo(f"  Random seed: {seed}")

    generate_synthetic_data(
        config=config,
        grid=(n_lat, n_lon),
        n_days=n_days,
        seed=seed,
    )

    typer.echo("Data saved to:")
    for p in paths:
        typer.echo(f"  {p}")

    typer.echo("Done!")
