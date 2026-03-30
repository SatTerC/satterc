"""Configuration generator for SatTerC.

This module contains the logic for generating configuration files
by introspecting the Hamilton driver and discovering required inputs.
"""

from __future__ import annotations

from importlib import import_module
import inspect
from typing import Any

from hamilton import driver
from hamilton.settings import ENABLE_POWER_USER_MODE

from .pipeline import models, inputs, outputs, resample


PATH_DEFAULTS = {
    "inputs_daily": "inputs/daily.nc",
    "inputs_weekly": "inputs/weekly.nc",
    "inputs_monthly": "inputs/monthly.nc",
    "inputs_static": "inputs/static.nc",
    "outputs_daily": "outputs/daily.nc",
    "outputs_weekly": "outputs/weekly.nc",
    "outputs_monthly": "outputs/monthly.nc",
}


def get_builtin_models() -> list[str]:
    """Get list of builtin models from __all__."""
    return list(models.__all__)


def get_model_params(model_name: str) -> dict[str, Any]:
    """Extract parameters from <model_name>_parameters() function signature."""
    builtin_models = get_builtin_models()
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


def infer_required_data(model_names: list[str]) -> dict[str, list[str]]:
    """Infer required input variables using Hamilton's list_available_variables.

    Builds a driver with all modules and extracts external inputs,
    categorizing them by frequency suffix.
    """
    model_modules = [getattr(models, name) for name in model_names]

    all_modules = model_modules + [
        inputs.daily,
        inputs.weekly,
        inputs.monthly,
        inputs.static,
        inputs.grid,
        resample,
        outputs.daily,
        outputs.weekly,
        outputs.monthly,
    ]

    model_params = {}
    for name in model_names:
        params = get_model_params(name)
        model_params.update(params)

    config = {
        ENABLE_POWER_USER_MODE: True,
        **model_params,
        "daily_inputs_path": "inputs/daily.nc",
        "weekly_inputs_path": "inputs/weekly.nc",
        "monthly_inputs_path": "inputs/monthly.nc",
        "static_inputs_path": "inputs/static.nc",
        "daily_inputs_vars": ["*"],
        "weekly_inputs_vars": ["*"],
        "monthly_inputs_vars": ["*"],
        "static_inputs_vars": ["*"],
        "daily_outputs_path": "outputs/daily.nc",
        "weekly_outputs_path": "outputs/weekly.nc",
        "monthly_outputs_path": "outputs/monthly.nc",
        "daily_outputs_vars": [],
        "weekly_outputs_vars": [],
        "monthly_outputs_vars": [],
        "daily_to_weekly": [],
        "daily_to_monthly": [],
        "weekly_to_monthly": [],
    }

    dr = driver.Builder().with_modules(*all_modules).with_config(config).build()

    all_vars = dr.list_available_variables()
    external_inputs = [v for v in all_vars if v.is_external_input]

    # Skip config keys plus additional patterns for internal variables
    skip_patterns = list(config.keys()) + ["_outputs_list", "var"]

    daily = []
    weekly = []
    monthly = []
    static = []

    for v in external_inputs:
        name = v.name
        if any(p in name for p in skip_patterns):
            continue
        if name.endswith("_daily"):
            daily.append(name[:-6])
        elif name.endswith("_weekly"):
            weekly.append(name[:-7])
        elif name.endswith("_monthly"):
            monthly.append(name[:-8])
        else:
            static.append(name)

    return {
        "inputs_daily": sorted(set(daily)),
        "inputs_weekly": sorted(set(weekly)),
        "inputs_monthly": sorted(set(monthly)),
        "inputs_static": sorted(set(static)),
        "resample_daily_to_weekly": [],
        "resample_daily_to_monthly": [],
        "resample_weekly_to_monthly": [],
        "outputs_daily": [],
        "outputs_weekly": [],
        "outputs_monthly": [],
    }


def format_list(items: list[Any], indent: int = 2) -> str:
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


def format_value(value: Any) -> str:
    """Format a simple value for TOML."""
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def dict_to_toml(d: dict) -> str:
    """Serialize config dict to TOML string."""
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
            lines.append(f"{k} = {format_value(v)}")

    if "models" in d:
        for model_name, params in d["models"].items():
            if params:
                lines.append("")
                lines.append(f"[models.{model_name}]")
                for k, v in params.items():
                    lines.append(f"{k} = {format_value(v)}")

    if "inputs" in d:
        for section in ["daily", "weekly", "monthly", "static"]:
            if section in d["inputs"]:
                data = d["inputs"][section]
                lines.append("")
                lines.append(f"[inputs.{section}]")
                lines.append(f'path = "{data["path"]}"')
                lines.append(f"vars = {format_list(data['vars'])}")

    if "resample" in d:
        lines.append("")
        lines.append("[resample]")
        for key in ["daily_to_weekly", "daily_to_monthly", "weekly_to_monthly"]:
            if key in d["resample"]:
                lines.append(f"{key} = {format_list(d['resample'][key])}")

    if "outputs" in d:
        for section in ["daily", "weekly", "monthly", "static"]:
            if section in d["outputs"]:
                data = d["outputs"][section]
                lines.append("")
                lines.append(f"[outputs.{section}]")
                lines.append(f'path = "{data["path"]}"')
                lines.append(f"vars = {format_list(data['vars'])}")

    return "\n".join(lines)


def generate_config(
    builtin_models: list[str],
    custom_modules: list[str],
    paths: dict[str, str],
) -> str:
    """Generate TOML configuration string.

    Parameters
    ----------
    builtin_models : list[str]
        List of builtin model names (e.g., ["splash", "pmodel"]).
    custom_modules : list[str]
        List of custom module paths.
    paths : dict[str, str]
        Dictionary mapping path keys to file paths.

    Returns
    -------
    str
        TOML configuration string.
    """
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

    required_data = infer_required_data(builtin_models)
    rothc_params = get_model_params("rothc")

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
        params = get_model_params(model)
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

    return dict_to_toml(config) + "\n"
