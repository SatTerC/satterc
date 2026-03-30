"""Configuration generator for SatTerC.

This module contains the logic for generating configuration files
by introspecting the Hamilton driver and discovering required inputs.
"""

from importlib import import_module
import inspect
from typing import Any
from types import ModuleType

from hamilton import driver
from hamilton.settings import ENABLE_POWER_USER_MODE
import xarray as xr

from .pipeline import models


def analyze_model_module(
    module: ModuleType, config: dict[str, Any]
) -> tuple[list[str], list[str], list[str]]:
    """Analyze a model module to discover its inputs and outputs.

    Parameters
    ----------
    module : ModuleType
        A Hamilton module (e.g., models.splash).
    config : dict[str, Any]
        Configuration dict with model parameters.

    Returns
    -------
    tuple
        A tuple containing three lists of strings:
        - 'data_inputs': list of external inputs of type xarray.DataArray
        - 'non_data_inputs': list of config inputs (not DataArray type)
        - 'data_outputs': list of produced outputs of type xarray.DataArray
    """
    dr = driver.Builder().with_modules(module).with_config(config).build()
    all_vars = dr.list_available_variables()

    data_inputs = []
    data_outputs = []
    non_data_inputs = []

    for v in all_vars:
        if v.is_external_input:
            if v.type == xr.DataArray:
                data_inputs.append(v.name)
            else:
                non_data_inputs.append(v.name)
        else:
            if v.type == xr.DataArray:
                data_outputs.append(v.name)

    return data_inputs, non_data_inputs, data_outputs


def _strip_suffix(name: str) -> tuple[str, str | None]:
    """Strip frequency suffix from variable name.

    Parameters
    ----------
    name : str
        Variable name (e.g., 'temperature_celcius_daily').

    Returns
    -------
    tuple
        (base_name, frequency) or (name, None) if no suffix found.
    """
    for suffix in ("_daily", "_weekly", "_monthly", "_static"):
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    return name, None


def get_model_params(model_name: str) -> dict[str, Any]:
    """Extract parameters from <model_name>_parameters() function signature."""
    builtin_models = list(models.__all__)
    module_path = (
        f"satterc.pipeline.models.{model_name}"
        if model_name in builtin_models
        else model_name
    )

    try:
        module = import_module(module_path)
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
    """Infer required data using analyze_model_module.

    Uses data_inputs from each model to construct input lists,
    filters out model outputs, and determines resample lists.
    """
    base_config = {ENABLE_POWER_USER_MODE: True}

    # Collect all model inputs and outputs
    all_data_inputs: set[str] = set()
    model_output_bases: set[str] = set()
    all_model_outputs: list[str] = []

    for model_name in model_names:
        module = getattr(models, model_name)
        data_inputs, _, data_outputs = analyze_model_module(module, base_config)

        # Store full input names (with suffix) for categorization
        all_data_inputs.update(data_inputs)
        all_model_outputs.extend(data_outputs)

        # Track base names of model outputs
        for output in data_outputs:
            base, _ = _strip_suffix(output)
            model_output_bases.add(base)

    # Filter out inputs that are model outputs (compare bases)
    # Also filter out grid variables (provided by inputs.grid, not from files)
    grid_vars = {"latitude", "longitude"}

    inputs_to_keep: set[str] = set()
    for name in all_data_inputs:
        base, freq = _strip_suffix(name)
        if base not in model_output_bases and base not in grid_vars:
            inputs_to_keep.add(name)

    # Categorize inputs by frequency
    daily: set[str] = set()
    weekly: set[str] = set()
    monthly: set[str] = set()
    static: set[str] = set()

    for name in inputs_to_keep:
        base, freq = _strip_suffix(name)
        if freq == "_daily":
            daily.add(base)
        elif freq == "_weekly":
            weekly.add(base)
        elif freq == "_monthly":
            monthly.add(base)
        else:
            static.add(base)

    # Determine resample lists (priority: daily -> weekly -> monthly)
    daily_to_weekly = daily & weekly
    weekly_to_monthly = weekly & monthly
    daily_to_monthly = (daily & monthly) - weekly

    # Remove resampled variables from input lists
    weekly = weekly - daily_to_weekly - weekly_to_monthly
    monthly = monthly - daily_to_monthly - weekly_to_monthly

    # Categorize model outputs for output file lists
    outputs_daily: list[str] = []
    outputs_weekly: list[str] = []
    outputs_monthly: list[str] = []

    for output in all_model_outputs:
        base, freq = _strip_suffix(output)
        if freq == "_daily":
            outputs_daily.append(base)
        elif freq == "_weekly":
            outputs_weekly.append(base)
        elif freq == "_monthly":
            outputs_monthly.append(base)

    return {
        "inputs_daily": sorted(daily),
        "inputs_weekly": sorted(weekly),
        "inputs_monthly": sorted(monthly),
        "inputs_static": sorted(static),
        "resample_daily_to_weekly": sorted(daily_to_weekly),
        "resample_daily_to_monthly": sorted(daily_to_monthly),
        "resample_weekly_to_monthly": sorted(weekly_to_monthly),
        "outputs_daily": sorted(set(outputs_daily)),
        "outputs_weekly": sorted(set(outputs_weekly)),
        "outputs_monthly": sorted(set(outputs_monthly)),
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
