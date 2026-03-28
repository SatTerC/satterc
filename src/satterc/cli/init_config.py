from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(help="Generate a configuration file for SatTerC.")

BUILTIN_MODELS = {
    "1": "splash",
    "2": "pmodel",
    "3": "sgam",
    "4": "rothc",
}

MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "splash": {
        "soil_moisture_init_max_iter": 10,
        "soil_moisture_init_max_diff": 1.0,
    },
    "pmodel": {
        "method_optchi": "prentice14",
        "method_jmaxlim": "wang17",
        "method_kphio": "temperature",
        "method_arrhenius": "simple",
    },
    "sgam": {},
    "rothc": {},
}

PATH_DEFAULTS = {
    "inputs_daily": "inputs/daily.nc",
    "inputs_weekly": "inputs/weekly.nc",
    "inputs_monthly": "inputs/monthly.nc",
    "inputs_static": "inputs/static.nc",
    "outputs_daily": "outputs/daily.nc",
    "outputs_weekly": "outputs/weekly.nc",
    "outputs_monthly": "outputs/monthly.nc",
}


def _infer_model_params(model_name: str) -> dict[str, Any]:
    """Return default parameters for a model.

    For builtin models, returns hardcoded defaults from MODEL_DEFAULTS.
    For custom modules, attempts to import and call the model's *_parameters() function.
    Falls back to empty dict if the module doesn't have a parameters function.
    """
    if model_name in MODEL_DEFAULTS:
        return MODEL_DEFAULTS[model_name]

    try:
        module = import_module(model_name)
        for attr_name in dir(module):
            if attr_name.endswith("_parameters"):
                param_func = getattr(module, attr_name)
                if callable(param_func):
                    import inspect

                    sig = inspect.signature(param_func)
                    defaults = {
                        p.name: p.default
                        for p in sig.parameters.values()
                        if p.default is not inspect.Parameter.empty
                    }
                    return defaults
    except ImportError:
        pass

    return {}


def _infer_required_data(model_names: list[str]) -> dict[str, list[str]]:
    """Infer required input/output variables based on model signatures.

    Currently returns empty lists for all sections. Future work will involve
    signature inspection or Hamilton DAG utilities to determine required data.
    """
    return {
        "inputs_daily": [],
        "inputs_weekly": [],
        "inputs_monthly": [],
        "inputs_static": [],
        "resample_daily_to_weekly": [],
        "resample_daily_to_monthly": [],
        "resample_weekly_to_monthly": [],
        "outputs_daily": [],
        "outputs_weekly": [],
        "outputs_monthly": [],
    }


def _display_remaining_models(remaining: list[str]) -> None:
    """Display remaining unselected models."""
    typer.echo("\nSelect models:")
    if remaining:
        for i, model in enumerate(remaining, 1):
            typer.echo(f"  [{i}] {model}")
    else:
        typer.echo("  (all built-in models selected)")
    typer.echo("  (type any non-number for custom module path)")


def _select_models() -> list[str]:
    """Interactive model selection loop."""
    builtin_models = list(BUILTIN_MODELS.values())
    remaining = list(builtin_models)
    selected: list[str] = []

    while True:
        typer.echo(
            f"\nModels selected: {', '.join(selected) if selected else '(none)'}"
        )
        _display_remaining_models(remaining)

        choice = typer.prompt("\nSelect models (Enter to continue)").strip()

        if choice == "":
            if not selected:
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
                if token in selected:
                    typer.echo(f"  '{token}' already selected")
                else:
                    custom_paths.append(token)

        for idx in sorted(indices_to_remove, reverse=True):
            model_name = remaining[idx]
            selected.append(model_name)
            remaining.remove(model_name)

        for custom_path in custom_paths:
            selected.append(custom_path)
            typer.echo(f"  Added custom module: {custom_path}")

    return selected


def _generate_toml(
    selected_models: list[str],
    paths: dict[str, str],
) -> str:
    """Generate TOML configuration string."""
    modules = [f"models.{m}" for m in selected_models if not m.startswith("models.")]
    modules += [m for m in selected_models if m.startswith("models.")]
    modules += [
        "inputs.daily",
        "inputs.weekly",
        "inputs.monthly",
        "inputs.static",
        "resample",
        "outputs.daily",
        "outputs.weekly",
        "outputs.monthly",
    ]

    required_data = _infer_required_data(selected_models)
    rothc_params = _infer_model_params("rothc")

    lines = [
        "modules = [",
    ]
    for module in modules:
        lines.append(f'  "{module}",')
    lines.append("]")

    lines.append("")
    lines.append("[extra_config]")
    if "rothc" in selected_models and "n_years_spinup" in rothc_params:
        lines.append(f"n_years_spinup = {rothc_params['n_years_spinup']}")
    else:
        lines.append("n_years_spinup = 1")

    for model in selected_models:
        model_key = model.split(".")[-1] if "." in model else model
        params = _infer_model_params(model_key)
        if params:
            lines.append("")
            lines.append(f"[models.{model_key}]")
            for key, value in params.items():
                if isinstance(value, str):
                    lines.append(f'{key} = "{value}"')
                else:
                    lines.append(f"{key} = {value}")

    lines.append("")
    lines.append("[inputs.daily]")
    lines.append(f'path = "{paths["inputs_daily"]}"')
    lines.append("vars = [")
    for var in required_data["inputs_daily"]:
        lines.append(f'  "{var}",')
    lines.append("]")

    lines.append("")
    lines.append("[inputs.weekly]")
    lines.append(f'path = "{paths["inputs_weekly"]}"')
    lines.append("vars = [")
    for var in required_data["inputs_weekly"]:
        lines.append(f'  "{var}",')
    lines.append("]")

    lines.append("")
    lines.append("[inputs.monthly]")
    lines.append(f'path = "{paths["inputs_monthly"]}"')
    lines.append("vars = [")
    for var in required_data["inputs_monthly"]:
        lines.append(f'  "{var}",')
    lines.append("]")

    lines.append("")
    lines.append("[inputs.static]")
    lines.append(f'path = "{paths["inputs_static"]}"')
    lines.append("vars = [")
    for var in required_data["inputs_static"]:
        lines.append(f'  "{var}",')
    lines.append("]")

    lines.append("")
    lines.append("[resample]")
    lines.append("daily_to_weekly = [")
    for var in required_data["resample_daily_to_weekly"]:
        lines.append(f'  "{var}",')
    lines.append("]")
    lines.append("")
    lines.append("daily_to_monthly = [")
    for var in required_data["resample_daily_to_monthly"]:
        lines.append(f'  "{var}",')
    lines.append("]")
    lines.append("")
    lines.append("weekly_to_monthly = [")
    for var in required_data["resample_weekly_to_monthly"]:
        lines.append(f'  "{var}",')
    lines.append("]")

    lines.append("")
    lines.append("[outputs.daily]")
    lines.append(f'path = "{paths["outputs_daily"]}"')
    lines.append("vars = [")
    for var in required_data["outputs_daily"]:
        lines.append(f'  "{var}",')
    lines.append("]")

    lines.append("")
    lines.append("[outputs.weekly]")
    lines.append(f'path = "{paths["outputs_weekly"]}"')
    lines.append("vars = [")
    for var in required_data["outputs_weekly"]:
        lines.append(f'  "{var}",')
    lines.append("]")

    lines.append("")
    lines.append("[outputs.monthly]")
    lines.append(f'path = "{paths["outputs_monthly"]}"')
    lines.append("vars = [")
    for var in required_data["outputs_monthly"]:
        lines.append(f'  "{var}",')
    lines.append("]")

    return "\n".join(lines) + "\n"


@app.command()
def init_config(
    output: Path = typer.Option(
        Path("config.toml"),
        "-o",
        "--output",
        help="Output path for the configuration file.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Use default values without prompting.",
    ),
) -> None:
    """Generate a configuration file for SatTerC."""
    if quiet:
        selected_models = list(BUILTIN_MODELS.values())
        paths = dict(PATH_DEFAULTS)
    else:
        typer.echo("SatTerC Configuration Generator")
        typer.echo("=" * 35)
        typer.echo()

        selected_models = _select_models()

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
    toml_content = _generate_toml(selected_models, paths)
    output.write_text(toml_content)
    typer.echo("Done!")
