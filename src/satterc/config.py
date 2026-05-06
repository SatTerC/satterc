"""Configuration management for SatTerC."""

import os
import tomllib
import tomli_w
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self


_RESAMPLE_FREQ_MAP: dict[tuple[str, str], str] = {
    ("daily", "weekly"): "7D",
    ("daily", "monthly"): "1ME",
    ("weekly", "monthly"): "1ME",
    # TODO: expose frequency strings as config options to support e.g. "W" (week-ending
    # Sunday) vs "7D" (rolling 7-day), or "MS" (month-start) vs "1ME" (month-end)
}

_VALID_AGGFUNCS: frozenset[str] = frozenset({"mean", "sum"})
# TODO: extend to allow any valid xarray DataArrayResample method:
#   min, max, std, var, first, last, median, count


@dataclass
class ResampleSpec:
    """Specification for a single [[resample]] entry."""

    vars: list[str]
    from_: str  # trailing underscore avoids clash with Python keyword 'from'
    to: str
    aggfunc: str = "mean"

    @property
    def freq(self) -> str:
        """xarray resample frequency string derived from from_/to pair."""
        return _RESAMPLE_FREQ_MAP[(self.from_, self.to)]

    @classmethod
    def from_config(cls, entry: dict) -> "ResampleSpec":
        """Construct and validate from a raw [[resample]] TOML entry."""
        from_freq = entry["from"]
        to_freq = entry["to"]
        aggfunc = entry.get("aggfunc", "mean")
        vars_ = entry["vars"]

        if (from_freq, to_freq) not in _RESAMPLE_FREQ_MAP:
            raise ValueError(
                f"Unsupported resample direction '{from_freq}' → '{to_freq}'. "
                f"Supported: {sorted(_RESAMPLE_FREQ_MAP)}"
            )
        if aggfunc not in _VALID_AGGFUNCS:
            raise ValueError(
                f"Unsupported aggfunc '{aggfunc}'. Supported: {sorted(_VALID_AGGFUNCS)}"
            )

        return cls(vars=vars_, from_=from_freq, to=to_freq, aggfunc=aggfunc)


@dataclass
class ParsedConfig:
    """Parsed pipeline configuration, ready to pass to build_driver."""

    modules: list[str]
    driver_config: dict[str, Any]
    targets: list[str] = field(default_factory=list)


class Config:
    """Configuration class with loading, parsing, and serialization."""

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize with a config dict."""
        self._data = data

    @classmethod
    def load(cls, path: str | os.PathLike) -> Self:
        """Load config from a TOML file."""
        path = Path(path).resolve()
        with open(path, "rb") as f:
            data = tomllib.load(f)
        _resolve_paths(data, base=path.parent)
        return cls(data)

    @classmethod
    def loads(cls, toml_str: str) -> Self:
        """Load config from a TOML string."""
        return cls(
            tomllib.loads(toml_str)
        )  # TODO: should we resolve paths relative to cwd?

    def parse(self) -> ParsedConfig:
        """Parse config into a ParsedConfig.

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
                # params is a list from [[resample]] array-of-tables
                seen_outputs: set[str] = set()
                specs: list[ResampleSpec] = []
                for entry in params:
                    spec = ResampleSpec.from_config(entry)
                    for var in spec.vars:
                        out = f"{var}_{spec.to}"
                        if out in seen_outputs:
                            raise ValueError(
                                f"Duplicate resample output '{out}' in [[resample]]"
                            )
                        seen_outputs.add(out)
                    specs.append(spec)
                driver_config["resample_specs"] = specs
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

        return ParsedConfig(
            modules=modules, driver_config=driver_config, targets=targets
        )

    def dump(self, path: str | os.PathLike, overwrite_ok: bool = False) -> None:
        """Write config to a TOML file."""
        toml_str = self.dumps()
        path = Path(path)
        if path.exists() and not overwrite_ok:
            raise FileExistsError(
                f"There is already a file at {path}! Consider passing `overwrite_ok=True`."
            )
        path.write_text(toml_str)

    def __str__(self) -> str:
        """Return TOML string representation."""
        return self.dumps()

    def dumps(self) -> str:
        """Dump config to a TOML str."""
        return tomli_w.dumps(self._data)


def load_config(config_path: str | Path) -> ParsedConfig:
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
