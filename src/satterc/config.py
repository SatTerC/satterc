"""Configuration management for SatTerC."""

import os
import tomllib
from pathlib import Path
from typing import Any


class Config:
    """Configuration class with loading, parsing, and serialization."""

    PATH_DEFAULTS = {
        "inputs_daily": "inputs/daily.nc",
        "inputs_weekly": "inputs/weekly.nc",
        "inputs_monthly": "inputs/monthly.nc",
        "inputs_static": "inputs/static.nc",
        "outputs_daily": "outputs/daily.nc",
        "outputs_weekly": "outputs/weekly.nc",
        "outputs_monthly": "outputs/monthly.nc",
    }

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize with a config dict."""
        self._data = data

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Config":
        """Load config from a TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls(data)

    def parse(self) -> dict[str, Any]:
        """Parse config into {modules, driver_config, targets}."""
        data = dict(self._data)

        modules = data.pop("modules")
        extra_modules = data.pop("extra_modules", [])
        extra_config = data.pop("extra_config", {})

        config_flat = _flatten_config(data)

        for section in config_flat:
            if section not in modules:
                raise ValueError(f"Section [{section}] not defined in modules list")

        driver_config = {}
        targets = []

        for section_name, params in config_flat.items():
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
            "extra_modules": extra_modules,
            "driver_config": driver_config,
            "targets": targets,
        }

    def dump(self, path: str | os.PathLike, overwrite_ok: bool = False) -> None:
        """Write config to a TOML file."""
        toml_str = self._dump()
        path = Path(path)
        if path.exists() and not overwrite_ok:
            raise FileExistsError(
                f"There is already a file at {path}! Consider passing `overwrite_ok=True`."
            )
        path.write_text(toml_str)

    def __str__(self) -> str:
        """Return TOML string representation."""
        return self._dump()

    def _dump(self) -> str:
        """Serialize config dict to TOML string."""
        d = self._data
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
    return Config.load(config_path).parse()


def _flatten_config(config: dict) -> dict[str, Any]:
    """Flatten nested TOML sections into dot-notation keys."""
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
