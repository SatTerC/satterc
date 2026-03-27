"""Configuration parsing for SatTerC."""

from pathlib import Path
import tomllib
from typing import Any


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and parse a TOML config file.

    Parameters
    ----------
    config_path : str | Path
        Path to the TOML config file.

    Returns
    -------
    dict
        Parsed config with modules, driver_config, and targets keys.
    """
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    modules = _build_modules(config)

    inputs = _flatten_inputs(config.get("inputs", {}))
    outputs = _flatten_outputs(config.get("outputs", {}))

    driver_config = {
        **config.get("extra_config", {}),
        **config.get("resample", {}),
        **inputs,
        **outputs,
    }

    driver_config.update(_flatten_models(config))

    targets = _get_targets(config.get("outputs", {}))

    return {
        "modules": modules,
        "driver_config": driver_config,
        "targets": targets,
    }


def _get_models(config: dict) -> tuple[list[str], dict[str, Any]]:
    """Return the list of model modules, and the contributions to driver_config."""
    modules = []
    driver_config = {}

    for name, params in config.get("models", {}).items():
        modules.append(f"models.{name}")
        driver_config.update(params)

    return modules, driver_config


def _build_modules(config: dict) -> list[str]:
    """Build the modules list from config sections and model names.

    Parameters
    ----------
    config : dict
        The parsed TOML config.

    Returns
    -------
    list[str]
        List of modules to include (e.g., ["inputs.daily", "models.splash"]).

    Raises
    ------
    ValueError
        If a duplicate module is specified.
    """
    modules = list(config.get("extra_modules", []))

    for name in config.get("models", {}):
        modules.append(f"models.{name}")

    for freq in config.get("inputs", {}):
        modules.append(f"inputs.{freq}")

    for freq in config.get("outputs", {}):
        modules.append(f"outputs.{freq}")

    reserved = {"extra_modules", "extra_config", "models", "inputs", "outputs"}
    for key in config:
        if key not in reserved and isinstance(config[key], dict):
            modules.append(key)

    if len(modules) != len(set(modules)):
        raise ValueError("Duplicate module specified")

    return modules


def _flatten_models(config: dict) -> dict:
    """Flatten model section parameters to driver_config.

    Parameters
    ----------
    config : dict
        The parsed TOML config.

    Returns
    -------
    dict
        Flattened model parameters.
    """
    flat = {}
    for model_name, model_params in config.get("models", {}).items():
        if not isinstance(model_params, dict):
            continue
        for param_name, param_value in model_params.items():
            if param_name in flat:
                raise ValueError(
                    f"Duplicate parameter: {param_name} appears in multiple [models.*] sections"
                )
            flat[param_name] = param_value
    return flat


def _flatten_inputs(config_inputs: dict) -> dict:
    flat = {}
    for freq in ["daily", "weekly", "monthly", "static"]:
        if freq in config_inputs:
            section = config_inputs[freq]
            flat[f"{freq}_inputs_path"] = section.get("path")
            flat[f"{freq}_inputs_vars"] = section.get("vars", [])
    return flat


def _flatten_outputs(config_outputs: dict) -> dict:
    flat = {}
    for freq in ["daily", "weekly", "monthly", "static"]:
        if freq in config_outputs:
            section = config_outputs[freq]
            flat[f"{freq}_outputs_path"] = section.get("path")
            flat[f"{freq}_outputs_vars"] = section.get("vars", [])
    return flat


def _get_targets(config_outputs: dict) -> list[str]:
    targets = []
    for freq in ["daily", "weekly", "monthly", "static"]:
        if freq in config_outputs:
            targets.append(f"save_{freq}_outputs")
    return targets
