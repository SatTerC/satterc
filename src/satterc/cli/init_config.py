from __future__ import annotations

from importlib import import_module
import inspect
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(help="Generate a configuration file for SatTerC.")


def _get_builtin_models() -> list[str]:
    """Get list of builtin models from __all__."""
    from satterc.pipeline import models

    return list(models.__all__)


def _infer_model_params(model_name: str) -> dict[str, Any]:
    """Extract parameters from <model_name>_parameters() function signature."""
    builtin_models = _get_builtin_models()
    if model_name in builtin_models:
        try:
            module = import_module(f"satterc.pipeline.models.{model_name}")
        except ImportError:
            return {}
    else:
        try:
            module = import_module(model_name)
        except ImportError:
            return {}

    param_func_name = f"{model_name}_parameters"
    if hasattr(module, param_func_name):
        param_func = getattr(module, param_func_name)
        sig = inspect.signature(param_func)
        return {
            p.name: p.default
            for p in sig.parameters.values()
            if p.default is not inspect.Parameter.empty
        }
    return {}


PATH_DEFAULTS = {
    "inputs_daily": "inputs/daily.nc",
    "inputs_weekly": "inputs/weekly.nc",
    "inputs_monthly": "inputs/monthly.nc",
    "inputs_static": "inputs/static.nc",
    "outputs_daily": "outputs/daily.nc",
    "outputs_weekly": "outputs/weekly.nc",
    "outputs_monthly": "outputs/monthly.nc",
}


def _get_model_outputs() -> set[str]:
    """Discover all output node names from builtin models.

    Reads the model source files and extracts:
    - Fields declared in @extract_fields(...) decorators
    - Functions with _daily, _weekly, _monthly, _static suffixes
    """
    import re
    from pathlib import Path

    outputs: set[str] = set()
    models_dir = Path(__file__).parent.parent / "pipeline" / "models"

    for model_name in _get_builtin_models():
        model_file = models_dir / f"{model_name}.py"
        if not model_file.exists():
            continue

        content = model_file.read_text()

        for match in re.finditer(r"@extract_fields\(\s*([^\)]+)\)", content, re.DOTALL):
            fields_str = match.group(1)
            outputs.update(re.findall(r'["\'](\w+)["\']', fields_str))

        for match in re.finditer(r"def (\w+)\(", content):
            func_name = match.group(1)
            if func_name.endswith(("_daily", "_weekly", "_monthly", "_static")):
                outputs.add(func_name)

    return outputs


def _infer_required_data(model_names: list[str]) -> dict[str, list[str]]:
    """Infer required input variables from model function signatures.

    Analyzes each model's main function signature and extracts parameters
    that are external inputs (not derived from other models or internal).
    """
    from satterc.pipeline import models

    known_upstream_params = _get_model_outputs()

    def classify_param(param_name: str) -> str | None:
        if param_name.endswith("_parameters"):
            return None
        if param_name.startswith("dates_"):
            return None
        if param_name in (
            "plant_type",
            "leaf_pool_init",
            "stem_pool_init",
            "root_pool_init",
            "organic_carbon_stocks",
        ):
            return "inputs_static"

        if param_name.endswith("_daily"):
            return "inputs_daily"
        if param_name.endswith("_weekly"):
            return "inputs_weekly"
        if param_name.endswith("_monthly"):
            return "inputs_monthly"
        return None

    result = {
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

    for model_name in model_names:
        module = getattr(models, model_name)
        main_func = getattr(module, model_name)
        sig = inspect.signature(main_func)

        for param_name in sig.parameters:
            if param_name in known_upstream_params:
                continue

            if param_name.endswith("_weekly"):
                base = param_name[:-7]
                if f"{base}_daily" in known_upstream_params:
                    continue
            if param_name.endswith("_monthly"):
                base = param_name[:-8]
                if (
                    f"{base}_daily" in known_upstream_params
                    or f"{base}_weekly" in known_upstream_params
                ):
                    continue

            category = classify_param(param_name)
            if category:
                var_name = param_name
                for suffix in ("_daily", "_weekly", "_monthly", "_static"):
                    if var_name.endswith(suffix):
                        var_name = var_name[: -len(suffix)]
                        break
                if var_name not in result[category]:
                    result[category].append(var_name)

    return result


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
    builtin_models = _get_builtin_models()
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
        for section in ["daily", "weekly", "monthly", "static"]:
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
        selected_models = (_get_builtin_models(), [])
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
