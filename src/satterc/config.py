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

    modules = config.pop("modules")
    extra_config = config.pop("extra_config", {})

    config_flat = _flatten_config(config)

    for section in config_flat:
        if section not in modules:
            raise ValueError(f"Section [{section}] not defined in modules list")

    driver_config = {}
    targets = []

    for section_name, params in config_flat.items():
        # inputs.freq and outputs.freq are treated differently
        if section_name.startswith("inputs."):
            _, freq = section_name.split(".", 1)
            driver_config[f"{freq}_inputs_path"] = params.get("path")
            driver_config[f"{freq}_inputs_vars"] = params.get("vars")
        elif section_name.startswith("outputs."):
            _, freq = section_name.split(".", 1)
            driver_config[f"{freq}_outputs_path"] = params.get("path")
            driver_config[f"{freq}_outputs_vars"] = params.get("vars")
            targets.append(f"save_{freq}_outputs")
        else:
            driver_config |= params

    driver_config |= extra_config

    return {
        "modules": modules,
        "driver_config": driver_config,
        "targets": targets,
    }


def _flatten_config(config: dict) -> dict[str, Any]:
    """Flatten nested TOML sections into dot-notation keys.

    Parameters
    ----------
    config : dict
        The parsed TOML config.

    Returns
    -------
    dict
        Flattened sections with dot-notation keys (e.g., "inputs.daily").
    """
    flat = {}
    for key, value in config.items():
        if key in ("modules", "extra_config"):
            continue
        if isinstance(value, dict) and all(isinstance(v, dict) for v in value.values()):
            for subkey, subvalue in value.items():
                flat[f"{key}.{subkey}"] = subvalue
        else:
            flat[key] = value
    return flat
