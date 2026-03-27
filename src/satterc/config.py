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
        **config.get("config", {}),
        **config.get("resample", {}),
        **inputs,
        **outputs,
    }

    targets = _get_targets(config.get("outputs", {}))

    return {
        "modules": modules,
        "driver_config": driver_config,
        "targets": targets,
    }


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
    """
    modules = []

    model_names = config.get("modules", [])
    modules.extend(model_names)

    config_inputs = config.get("inputs", {})
    for freq in ["daily", "weekly", "monthly", "static"]:
        if freq in config_inputs:
            modules.append(f"inputs.{freq}")

    config_outputs = config.get("outputs", {})
    for freq in ["daily", "weekly", "monthly", "static"]:
        if freq in config_outputs:
            modules.append(f"outputs.{freq}")

    if "resample" in config:
        modules.append("resample")

    return modules


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
