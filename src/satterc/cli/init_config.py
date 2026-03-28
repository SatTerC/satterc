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
    builtin_models = list(BUILTIN_MODELS.values())
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


def _format_list(items: list[Any], indent: int = 2) -> str:
    """Format a list as a TOML array."""
    prefix = " " * indent
    if not items:
        return "[]"
    lines = ["["]
    for item in items:
        if isinstance(item, str):
            lines.append(f'{prefix}  "{item}",')
        else:
            lines.append(f"{prefix}  {item},")
    lines.append(f"{prefix}]")
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    """Format a simple value for TOML."""
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _dict_to_toml(d: dict) -> str:
    """Serialize config dict to TOML string.

    Handles the specific structure used by this project:
    - modules: list[str]
    - extra_modules: list[str] (optional)
    - extra_config: dict[str, Any]
    - models: dict[str, dict[str, Any]]
    - inputs/outputs: dict[str, dict[str, Any]]  (each with path + vars)
    - resample: dict[str, list[str]]
    """
    lines = []

    if "modules" in d:
        lines.append("modules = [")
        for m in d["modules"]:
            lines.append(f'  "{m}",')
        lines.append("]")

    if "extra_modules" in d:
        lines.append("")
        lines.append("extra_modules = [")
        for m in d["extra_modules"]:
            lines.append(f'  "{m}",')
        lines.append("]")

    if "extra_config" in d:
        lines.append("")
        lines.append("[extra_config]")
        for k, v in d["extra_config"].items():
            lines.append(f"{k} = {_format_value(v)}")

    if "models" in d:
        for model_name, params in d["models"].items():
            if params:
                lines.append("")
                lines.append(f"[models.{model_name}]")
                for k, v in params.items():
                    lines.append(f"{k} = {_format_value(v)}")

    if "inputs" in d:
        for section in ["daily", "weekly", "monthly", "static"]:
            if section in d["inputs"]:
                data = d["inputs"][section]
                lines.append("")
                lines.append(f"[inputs.{section}]")
                lines.append(f'path = "{data["path"]}"')
                lines.append(f"vars = {_format_list(data['vars'])}")

    if "resample" in d:
        lines.append("")
        lines.append("[resample]")
        for key in ["daily_to_weekly", "daily_to_monthly", "weekly_to_monthly"]:
            if key in d["resample"]:
                lines.append(f"{key} = {_format_list(d['resample'][key])}")

    if "outputs" in d:
        for section in ["daily", "weekly", "monthly"]:
            if section in d["outputs"]:
                data = d["outputs"][section]
                lines.append("")
                lines.append(f"[outputs.{section}]")
                lines.append(f'path = "{data["path"]}"')
                lines.append(f"vars = {_format_list(data['vars'])}")

    return "\n".join(lines)


def _generate_toml(
    selected_models: tuple[list[str], list[str]],
    paths: dict[str, str],
) -> str:
    """Generate TOML configuration string."""
    builtin_models, custom_modules = selected_models

    modules = [f"models.{m}" for m in builtin_models]
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

    required_data = _infer_required_data(builtin_models)
    rothc_params = _infer_model_params("rothc")

    config: dict[str, Any] = {
        "modules": modules,
    }

    if custom_modules:
        config["extra_modules"] = custom_modules

    config["extra_config"] = {
        "n_years_spinup": rothc_params.get("n_years_spinup", 1)
        if "rothc" in builtin_models
        else 1
    }

    config["models"] = {}
    for model in builtin_models:
        params = _infer_model_params(model)
        if params:
            config["models"][model] = params

    config["inputs"] = {
        "daily": {"path": paths["inputs_daily"], "vars": required_data["inputs_daily"]},
        "weekly": {
            "path": paths["inputs_weekly"],
            "vars": required_data["inputs_weekly"],
        },
        "monthly": {
            "path": paths["inputs_monthly"],
            "vars": required_data["inputs_monthly"],
        },
        "static": {
            "path": paths["inputs_static"],
            "vars": required_data["inputs_static"],
        },
    }

    config["resample"] = {
        "daily_to_weekly": required_data["resample_daily_to_weekly"],
        "daily_to_monthly": required_data["resample_daily_to_monthly"],
        "weekly_to_monthly": required_data["resample_weekly_to_monthly"],
    }

    config["outputs"] = {
        "daily": {
            "path": paths["outputs_daily"],
            "vars": required_data["outputs_daily"],
        },
        "weekly": {
            "path": paths["outputs_weekly"],
            "vars": required_data["outputs_weekly"],
        },
        "monthly": {
            "path": paths["outputs_monthly"],
            "vars": required_data["outputs_monthly"],
        },
    }

    return _dict_to_toml(config) + "\n"


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
        selected_models = (list(BUILTIN_MODELS.values()), [])
        paths = dict(PATH_DEFAULTS)
    else:
        typer.echo("SatTerC Configuration Generator")
        typer.echo("=" * 35)
        typer.echo()

        selected_models = _select_models()

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
    toml_content = _generate_toml(selected_models, paths)
    output.write_text(toml_content)
    typer.echo("Done!")
