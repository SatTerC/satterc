"""Configuration management for SatTerC."""

import os
import tomllib
from pathlib import Path
from typing import Any


class Config:
    """Configuration class with loading, parsing, and serialization."""

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize with a config dict."""
        self._data = data

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Config":
        """Load config from a TOML file."""
        path = Path(path).resolve()
        with open(path, "rb") as f:
            data = tomllib.load(f)
        _resolve_paths(data, base=path.parent)
        return cls(data)

    def parse(self) -> dict[str, Any]:
        """Parse config into {modules, driver_config, targets}.

        Every TOML section implies a module to load:
        - [inputs.*]  and [outputs.*] — I/O modules, derived from file extension
        - [models.*]  — built-in model modules
        - [resample]  — temporal resampling module
        - [extra_config] — reserved: merges keys into driver_config, no module loaded
        - [pkg.mod]   — external importable module (max 2-component path for now)
        """
        data = dict(self._data)
        extra_config = data.pop("extra_config", {})

        config_flat = _flatten_config(data)

        driver_config: dict[str, Any] = {}
        targets: list[str] = []
        modules: list[str] = []

        for section_name, params in config_flat.items():
            depth = section_name.count(".") + 1

            if section_name.startswith("inputs."):
                freq = section_name.split(".", 1)[1]
                if "path" in params:
                    fmt = _infer_format(params["path"])
                    driver_config[f"{freq}_inputs_path"] = params["path"]
                    driver_config[f"{freq}_inputs_vars"] = params.get("vars") or []
                    driver_config[f"{freq}_inputs_format"] = fmt
                    modules.append(f"inputs.{freq}")
                else:
                    # Helper section with no file path (e.g. [inputs.grid])
                    modules.append(section_name)

            elif section_name.startswith("outputs."):
                freq = section_name.split(".", 1)[1]
                vars_ = params.get("vars") or []
                if vars_:
                    fmt = _infer_format(params["path"])
                    driver_config[f"{freq}_outputs_path"] = params["path"]
                    driver_config[f"{freq}_outputs_vars"] = vars_
                    driver_config[f"{freq}_outputs_format"] = fmt
                    targets.append(f"save_{freq}_outputs")
                    modules.append(f"outputs.{freq}")
                # else: empty vars → skip (no module, no target, no config keys)

            elif section_name.startswith("models."):
                conflicts = set(params.keys()) & set(driver_config.keys())
                if conflicts:
                    raise ValueError(
                        f"Parameter(s) {sorted(conflicts)} in [{section_name}] conflict "
                        f"with an already-defined key. Use a model-specific prefix to "
                        f"disambiguate (e.g. pmodel_method_kphio)."
                    )
                driver_config |= params
                modules.append(section_name)

            elif section_name == "resample":
                driver_config |= params
                modules.append("resample")

            else:
                # External module: section name is an importable path.
                # NOTE: Only 2-component paths (pkg.mod) are supported for now.
                # Support for deeper paths (pkg.sub.mod) may be added in future.
                has_nested = any(isinstance(v, dict) for v in params.values())
                if depth > 2 or has_nested:
                    raise ValueError(
                        f"[{section_name}] has {depth} path components or contains "
                        f"sub-sections. External module paths must be 2 components "
                        f"(e.g. mypackage.mymodule). "
                        f"Support for deeper paths may be added in future."
                    )
                conflicts = set(params.keys()) & set(driver_config.keys())
                if conflicts:
                    raise ValueError(
                        f"Parameter(s) {sorted(conflicts)} in [{section_name}] conflict "
                        f"with an already-defined key."
                    )
                driver_config |= params
                modules.append(section_name)

        driver_config |= extra_config

        return {"modules": modules, "driver_config": driver_config, "targets": targets}

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

    _KNOWN_SECTIONS = frozenset({"extra_config", "models", "inputs", "resample", "outputs"})

    def _dump(self) -> str:
        """Serialize config dict to TOML string."""
        d = self._data
        lines = []

        if "extra_config" in d:
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
            for section, data in d["inputs"].items():
                lines.append("")
                lines.append(f"[inputs.{section}]")
                if "path" in data:
                    lines.append(f'path = "{data["path"]}"')
                    lines.append(f"vars = {_format_list(data.get('vars', []))}")

        if "resample" in d:
            lines.append("")
            lines.append("[resample]")
            for key in ["daily_to_weekly", "daily_to_monthly", "weekly_to_monthly"]:
                if key in d["resample"]:
                    lines.append(f"{key} = {_format_list(d['resample'][key])}")

        if "outputs" in d:
            for section, data in d["outputs"].items():
                if "path" in data:
                    lines.append("")
                    lines.append(f"[outputs.{section}]")
                    lines.append(f'path = "{data["path"]}"')
                    lines.append(f"vars = {_format_list(data.get('vars', []))}")

        for key, value in d.items():
            if key not in self._KNOWN_SECTIONS and isinstance(value, dict):
                for subkey, params in value.items():
                    lines.append("")
                    lines.append(f"[{key}.{subkey}]")
                    for k, v in params.items():
                        lines.append(f"{k} = {_format_value(v)}")

        return "\n".join(lines)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and parse a TOML config file."""
    return Config.load(config_path).parse()


def _resolve_paths(data: dict, base: Path) -> None:
    """Resolve relative paths in-place, relative to the config file's directory."""
    for section in ("inputs", "outputs"):
        for params in data.get(section, {}).values():
            if "path" in params and not Path(params["path"]).is_absolute():
                params["path"] = str(base / params["path"])


def _flatten_config(config: dict) -> dict[str, Any]:
    """Flatten nested TOML sections into dot-notation keys.

    Dicts whose values are all dicts are treated as namespaces and flattened
    one level (e.g. {inputs: {daily: {path, vars}}} → {"inputs.daily": {path, vars}}).
    Dicts with any non-dict value are kept as-is (e.g. resample, model params).
    """
    flat = {}
    for key, value in config.items():
        if (
            isinstance(value, dict)
            and value
            and all(isinstance(v, dict) for v in value.values())
        ):
            for subkey, subvalue in value.items():
                flat[f"{key}.{subkey}"] = subvalue
        else:
            flat[key] = value
    return flat


def _infer_format(path: str) -> str:
    """Derive 'netcdf' or 'flat' from file extension.

    netcdf: .nc, .netcdf, .zarr, or no extension (bare zarr directory)
    flat:   .csv, .parquet, .pq, .json, .yaml, .yml, .toml
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".nc", ".netcdf", ".zarr") or not ext:
        return "netcdf"
    if ext in (".csv", ".parquet", ".pq", ".json", ".yaml", ".yml", ".toml"):
        return "flat"
    raise ValueError(
        f"Cannot determine format from extension '{ext}' in path '{path}'. "
        f"Expected .nc/.netcdf/.zarr (netcdf) or "
        f".csv/.parquet/.json/.yaml/.yml/.toml (flat)."
    )


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
