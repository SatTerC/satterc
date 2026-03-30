"""CLI for generating configuration files."""

import re
from pathlib import Path

import typer

from ..config import Config
from ..setup_utils import generate_config, get_builtin_models
from ..setup_utils.data_gen import generate_synthetic_data

app = typer.Typer(help="Generate a configuration file for SatTerC.")

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


def _parse_selections(selection_str: str) -> list[str]:
    """Parse comma/space-separated selection into list of items."""
    items = re.split(r"[,\s]+", selection_str.strip())
    return [item for item in items if item]


def _display_models(models: list[str], selected: set[str], n_cols: int = 4) -> None:
    """Display available models with [x] next to selected ones in columns."""
    typer.echo("\nAvailable models:")
    for i in range(0, len(models), n_cols):
        row = models[i : i + n_cols]
        parts = []
        for j, model in enumerate(row):
            idx = i + j + 1
            marker = "x" if model in selected else " "
            parts.append(f"[{marker}] {idx}: {model}")
        typer.echo("  " + "  ".join(parts))


def _toggle_selections(
    current: list[str],
    selections: list[str],
    available: set[str] | None = None,
) -> list[str]:
    """Toggle items in/out of selection list.

    Args:
        current: Currently selected items
        selections: Items to toggle
        available: Optional set of valid items for validation

    Returns:
        Updated list with toggled items
    """
    for item in selections:
        if available and item not in available:
            continue
        if item in current:
            current.remove(item)
        else:
            current.append(item)
    return current


def _select_builtin_models() -> list[str]:
    """Interactive selection loop for built-in models.

    Returns a list of selected built-in model names.
    """
    builtin_models = get_builtin_models()
    selected: list[str] = []
    available_set = set(builtin_models)

    while True:
        typer.echo(f"\nSelected: {', '.join(selected) or '(none)'}")
        _display_models(builtin_models, set(selected))
        typer.echo("  (enter numbers or names, comma or space separated, 0 when done)")

        choice = typer.prompt(
            "\nSelect models",
            default="",
            show_default=False,
            prompt_suffix="\n> ",
        ).strip()

        if choice == "" or choice == "0":
            if not selected:
                typer.echo("  Error: You must select at least one model!")
                continue
            break

        selections = _parse_selections(choice)

        validated_selections = []
        for token in selections:
            if token.isdigit():
                idx = int(token)
                if 1 <= idx <= len(builtin_models):
                    validated_selections.append(builtin_models[idx - 1])
                else:
                    typer.echo(f"  Invalid number: {token}")
            else:
                if token in available_set:
                    validated_selections.append(token)
                else:
                    typer.echo(f"  Unknown model: {token}")

        _toggle_selections(selected, validated_selections)

    return selected


def _select_custom_modules() -> list[str]:
    """Interactive selection loop for custom module paths.

    Returns a list of custom module paths.
    """
    selected: list[str] = []

    while True:
        typer.echo(f"\nSelected: {', '.join(selected) or '(none)'}")

        choice = typer.prompt(
            "Enter module paths (comma or space separated, 0 when done)",
            default="",
            show_default=False,
            prompt_suffix="\n> ",
        ).strip()

        if choice == "" or choice == "0":
            break

        selections = _parse_selections(choice)

        for item in selections:
            action = "Removed" if item in selected else "Added"
            typer.echo(f"  {action}: {item}")

        _toggle_selections(selected, selections)

    return selected


def _select_models() -> tuple[list[str], list[str]]:
    """Interactive model selection loop.

    Returns a tuple of (builtin_models, custom_modules).
    """
    selected_builtin = _select_builtin_models()
    selected_custom = _select_custom_modules()
    return selected_builtin, selected_custom


@app.command()
def setup(
    output: Path = typer.Option(
        Path("config.toml"),
        "-o",
        "--output",
        help="Output path for the configuration file.",
    ),
    defaults: bool = typer.Option(
        False,
        "-d",
        "--defaults",
        help="Use default values without prompting.",
    ),
) -> None:
    """Generate a configuration file for SatTerC."""
    if defaults:
        builtin_models = get_builtin_models()
        custom_modules: list[str] = []
        paths = dict(Config.PATH_DEFAULTS)
    else:
        typer.echo("SatTerC Configuration Generator")
        typer.echo("=" * 35)
        typer.echo()

        builtin_models, custom_modules = _select_models()

        use_defaults = typer.confirm(
            "Use default input/output paths?",
            default=True,
        )

        if use_defaults:
            paths = dict(Config.PATH_DEFAULTS)
        else:
            typer.echo("\nInput file paths:")
            paths = {}
            paths["inputs_daily"] = typer.prompt(
                "Daily input path",
                default=Config.PATH_DEFAULTS["inputs_daily"],
            )
            paths["inputs_weekly"] = typer.prompt(
                "Weekly input path",
                default=Config.PATH_DEFAULTS["inputs_weekly"],
            )
            paths["inputs_monthly"] = typer.prompt(
                "Monthly input path",
                default=Config.PATH_DEFAULTS["inputs_monthly"],
            )
            paths["inputs_static"] = typer.prompt(
                "Static input path",
                default=Config.PATH_DEFAULTS["inputs_static"],
            )

            typer.echo("\nOutput file paths:")
            paths["outputs_daily"] = typer.prompt(
                "Daily output path",
                default=Config.PATH_DEFAULTS["outputs_daily"],
            )
            paths["outputs_weekly"] = typer.prompt(
                "Weekly output path",
                default=Config.PATH_DEFAULTS["outputs_weekly"],
            )
            paths["outputs_monthly"] = typer.prompt(
                "Monthly output path",
                default=Config.PATH_DEFAULTS["outputs_monthly"],
            )

        typer.echo()
        output = typer.prompt(
            "Output config path",
            default=output,
            type=Path,
        )

    typer.echo(f"\nGenerating {output}... ", nl=False)
    config = generate_config(builtin_models, custom_modules, paths)
    config.dump(output)
    typer.echo("Done!")

    if not defaults:
        generate_data = typer.confirm(
            "\nGenerate synthetic input data?",
            default=False,
        )

        if generate_data:
            typer.echo("\nSynthetic data generation options:")
            grid_str = typer.prompt(
                "Grid dimensions (n_lat,n_lon)",
                default="1,1",
            )
            try:
                n_lat, n_lon = map(int, grid_str.split(","))
                if n_lat <= 0 or n_lon <= 0:
                    raise ValueError("Dimensions must be positive")
            except Exception:
                typer.echo("  Invalid grid format. Using default (1,1).")
                n_lat, n_lon = 1, 1

            duration_str = typer.prompt(
                "Duration (e.g., 2y, 6m, 30d)",
                default="2y",
            )
            n_days = _parse_duration(duration_str)

            seed = typer.prompt(
                "Random seed",
                default="42",
                type=int,
            )

            typer.echo("\nGenerating synthetic data...")
            parsed_config = Config.load(output).parse()

            config_dir = output.parent.resolve()
            for freq in ["daily", "weekly", "monthly", "static"]:
                path_key = f"{freq}_inputs_path"
                if path_key in parsed_config["driver_config"]:
                    rel_path = parsed_config["driver_config"][path_key]
                    full_path = config_dir / rel_path
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    parsed_config["driver_config"][path_key] = str(full_path)

            generate_synthetic_data(
                config=parsed_config,
                grid=(n_lat, n_lon),
                n_days=n_days,
                seed=seed,
            )
            typer.echo("Data generation complete!")
