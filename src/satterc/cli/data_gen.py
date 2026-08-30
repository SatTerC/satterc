"""Generate synthetic input data for testing."""

import re
from datetime import date
from pathlib import Path
from typing import Annotated

import typer
from conduit import ParsedConfig, load_config
from typer import Abort

from ..scaffold.data_gen import generate_synthetic_data
from ..scaffold.data_gen.spec import (
    DEFAULT_LAT_RANGE,
    DEFAULT_LON_RANGE,
    DEFAULT_START_DATE,
)

app = typer.Typer(help="Generate synthetic input data for testing.")


DURATION_PATTERN = re.compile(r"^(\d+)([ymd])$")


def _parse_duration(duration: str) -> int:
    match = DURATION_PATTERN.match(duration.lower())
    if not match:
        raise typer.BadParameter(
            f"Invalid duration format: '{duration}'. "
            f"Expected format like '2y', '6m', '30d'."
        )
    value, unit = match.groups()
    value = int(value)
    if unit == "d":
        return value
    elif unit == "m":
        return int(value * 30.44)
    elif unit == "y":
        return int(value * 365.25)
    raise ValueError(f"Invalid duration unit: {unit}")


def _parse_bbox(bbox: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Parse ``'lat_min,lat_max,lon_min,lon_max'`` into lat and lon ranges.

    Taken as one comma-separated string rather than four option values because a
    western longitude starts with ``-``, which click reads as the start of
    another option.
    """
    parts = bbox.split(",")
    if len(parts) != 4:
        raise typer.BadParameter(
            f"Invalid bounding box: '{bbox}'. Expected four comma-separated "
            f"numbers, 'lat_min,lat_max,lon_min,lon_max' (e.g. '50,54,-4,2')."
        )
    try:
        lat_min, lat_max, lon_min, lon_max = (float(p) for p in parts)
    except ValueError:
        raise typer.BadParameter(
            f"Invalid bounding box: '{bbox}'. All four values must be numbers."
        ) from None
    if lat_min > lat_max or lon_min > lon_max:
        raise typer.BadParameter(
            f"Invalid bounding box: '{bbox}'. Each min must not exceed its max."
        )
    if not (lat_min >= -90.0 and lat_max <= 90.0):
        raise typer.BadParameter(f"Latitudes must lie in [-90, 90]: '{bbox}'.")
    if not (lon_min >= -180.0 and lon_max <= 180.0):
        raise typer.BadParameter(f"Longitudes must lie in [-180, 180]: '{bbox}'.")
    return (lat_min, lat_max), (lon_min, lon_max)


def _parse_start_date(start_date: str) -> str:
    """Validate an ISO ``YYYY-MM-DD`` start date."""
    try:
        date.fromisoformat(start_date)
    except ValueError:
        raise typer.BadParameter(
            f"Invalid start date: '{start_date}'. Expected ISO 'YYYY-MM-DD'."
        ) from None
    return start_date


def _validate_output_paths(
    config: ParsedConfig,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Validate or create directories, prompt for overwrite if files exist.

    Returns paths, directories to create, and files to overwrite.
    """
    frequencies = ["daily", "weekly", "monthly", "static"]

    paths = []
    dirs_to_create = []
    files_to_overwrite = []

    for freq in frequencies:
        spec = config.input_specs.get(freq)
        if spec is None:
            continue

        path = Path(spec.path)
        paths.append(path)

        if not path.parent.exists():
            dirs_to_create.append(path.parent)
        elif path.is_file():
            files_to_overwrite.append(path)

    return paths, dirs_to_create, files_to_overwrite


@app.command("data-gen")
def generate(
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
            help=(
                "Time duration (e.g., '2y' for 2 years, '6m' for 6 months, "
                "'30d' for 30 days)."
            ),
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
    bbox: Annotated[
        str,
        typer.Option(
            "--bbox",
            "-b",
            help=(
                "Bounding box as 'lat_min,lat_max,lon_min,lon_max' in degrees "
                "(e.g. '50,54,-4,2'). The climate follows the box, so a tropical "
                "one comes out warm and aseasonal."
            ),
        ),
    ] = ",".join(str(v) for v in (*DEFAULT_LAT_RANGE, *DEFAULT_LON_RANGE)),
    start_date: Annotated[
        str,
        typer.Option(
            "--start-date",
            help="ISO date of the first day, 'YYYY-MM-DD'.",
        ),
    ] = DEFAULT_START_DATE,
) -> None:
    """Generate synthetic input data for every input section of a config."""
    n_lat, n_lon = grid
    if n_lat <= 0 or n_lon <= 0:
        raise typer.BadParameter("Grid dimensions must be positive integers.")

    n_days = _parse_duration(duration)
    lat_range, lon_range = _parse_bbox(bbox)
    start_date = _parse_start_date(start_date)

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
    typer.echo(f"  Duration: {duration} ({n_days} days) from {start_date}")
    typer.echo(f"  Bounding box: {lat_range} lat, {lon_range} lon")
    typer.echo(f"  Random seed: {seed}")

    generate_synthetic_data(
        config=config,
        grid=(n_lat, n_lon),
        n_days=n_days,
        seed=seed,
        lat_range=lat_range,
        lon_range=lon_range,
        start_date=start_date,
    )

    typer.echo("Data saved to:")
    for p in paths:
        typer.echo(f"  {p}")

    typer.echo("Done!")
