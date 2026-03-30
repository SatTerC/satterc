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


def _display_remaining_models(remaining: list[str], n_cols: int = 4) -> None:
    """Display remaining unselected models in columns."""
    typer.echo("\nAvailable models:")
    if remaining:
        for i in range(0, len(remaining), n_cols):
            row = remaining[i : i + n_cols]
            parts = []
            for j, model in enumerate(row):
                idx = i + j + 1
                parts.append(f"[{idx}] {model}")
            typer.echo("  " + "  ".join(parts))
    else:
        typer.echo("  (all built-in models selected)")
    typer.echo("  (type any non-number for custom module path)")


def _select_models() -> tuple[list[str], list[str]]:
    """Interactive model selection loop.

    Returns a tuple of (builtin_models, custom_modules).
    """
    builtin_models = get_builtin_models()
    remaining = list(builtin_models)
    selected_builtin: list[str] = []
    selected_custom: list[str] = []

    while True:
        all_selected = selected_builtin + selected_custom
        typer.echo(
            f"\nModels selected: {', '.join(all_selected) if all_selected else '(none)'}"
        )
        _display_remaining_models(remaining)

        choice = typer.prompt(
            "\nSelect models (e.g. '1', '1 2', 'mypackage.mymodule', or leave blank to continue)",
            default="",
            show_default=False,
            prompt_suffix="\n> ",
        ).strip()

        if choice == "":
            if not all_selected:
                typer.echo("  Error: You must select at least one model!")
                continue
            break

        if choice == "0":
            if not all_selected:
                typer.echo("  Error: You must select at least one model!")
                continue
            break

        tokens = choice.split()
        indices_to_remove = []
        custom_paths = []

        for token in tokens:
            if token.isdigit():
                idx = int(token)
                if 1 <= idx <= len(remaining):
                    indices_to_remove.append(idx - 1)
                else:
                    typer.echo(f"  Invalid number: {token}")
            else:
                if token in selected_custom:
                    typer.echo(f"  '{token}' already selected")
                else:
                    custom_paths.append(token)

        for idx in sorted(indices_to_remove, reverse=True):
            model_name = remaining[idx]
            selected_builtin.append(model_name)
            remaining.remove(model_name)

        for custom_path in custom_paths:
            selected_custom.append(custom_path)
            typer.echo(f"  Added custom module: {custom_path}")

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
