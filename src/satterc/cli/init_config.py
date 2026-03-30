"""CLI for generating configuration files."""

from pathlib import Path

import typer

from ..config_generator import generate_config
from ..pipeline import models

app = typer.Typer(help="Generate a configuration file for SatTerC.")

PATH_DEFAULTS = {
    "inputs_daily": "inputs/daily.nc",
    "inputs_weekly": "inputs/weekly.nc",
    "inputs_monthly": "inputs/monthly.nc",
    "inputs_static": "inputs/static.nc",
    "outputs_daily": "outputs/daily.nc",
    "outputs_weekly": "outputs/weekly.nc",
    "outputs_monthly": "outputs/monthly.nc",
}


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
    builtin_models = list(models.__all__)
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
def init_config(
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
        builtin_models = list(models.__all__)
        custom_modules: list[str] = []
        paths = dict(PATH_DEFAULTS)
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
            paths = dict(PATH_DEFAULTS)
        else:
            typer.echo("\nInput file paths:")
            paths = {}
            paths["inputs_daily"] = typer.prompt(
                "Daily input path",
                default=PATH_DEFAULTS["inputs_daily"],
            )
            paths["inputs_weekly"] = typer.prompt(
                "Weekly input path",
                default=PATH_DEFAULTS["inputs_weekly"],
            )
            paths["inputs_monthly"] = typer.prompt(
                "Monthly input path",
                default=PATH_DEFAULTS["inputs_monthly"],
            )
            paths["inputs_static"] = typer.prompt(
                "Static input path",
                default=PATH_DEFAULTS["inputs_static"],
            )

            typer.echo("\nOutput file paths:")
            paths["outputs_daily"] = typer.prompt(
                "Daily output path",
                default=PATH_DEFAULTS["outputs_daily"],
            )
            paths["outputs_weekly"] = typer.prompt(
                "Weekly output path",
                default=PATH_DEFAULTS["outputs_weekly"],
            )
            paths["outputs_monthly"] = typer.prompt(
                "Monthly output path",
                default=PATH_DEFAULTS["outputs_monthly"],
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
