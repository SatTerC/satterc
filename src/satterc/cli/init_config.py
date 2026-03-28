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


def _display_model_selection(available: list[str], selected: list[str]) -> None:
    """Display available models with selection status."""
    builtin_labels = {
        "splash": "splash",
        "pmodel": "pmodel",
        "sgam": "sgam",
        "rothc": "rothc",
        "custom": "custom path",
    }

    for model in available:
        marker = "[x]" if model in selected else "[ ]"
        label = builtin_labels.get(model, model)
        typer.echo(f"  {marker} {label}")


def _select_models() -> list[str]:
    """Interactive model selection loop."""
    available = list(BUILTIN_MODELS.values()) + ["custom"]
    selected: list[str] = []

    typer.echo("\nAvailable models:")
    typer.echo("  (type number to select/deselect, 0 to continue)")

    while True:
        typer.echo()
        for i, model in enumerate(available, 1):
            marker = "[x]" if model in selected else "[ ]"
            label = "custom path" if model == "custom" else model
            typer.echo(f"  [{i}] {marker} {label}")

        if selected:
            typer.echo(f"\n  Selected: {len(selected)} model(s)")

        typer.echo()
        choice = typer.prompt("Select model (0 to continue)").strip()

        if choice == "":
            typer.echo(
                "  Error: No selection made. Type a number or 0 to continue.\n",
                err=True,
            )
            continue

        if choice == "0":
            if not selected:
                typer.echo(
                    "  Error: You must select at least one model!\n",
                    err=True,
                )
                continue
            break

        if choice == "5":
            if "custom" not in selected:
                custom_path = typer.prompt("  Enter custom model path")
                if custom_path:
                    selected.append(custom_path)
            else:
                typer.echo("  Custom path already selected.\n")
            continue

        if choice in BUILTIN_MODELS:
            model_name = BUILTIN_MODELS[choice]
            if model_name in selected:
                selected.remove(model_name)
                typer.echo(f"  Deselected: {model_name}\n")
            else:
                selected.append(model_name)
                typer.echo(f"  Selected: {model_name}\n")
        else:
            typer.echo(f"  Invalid choice: {choice}\n")

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
