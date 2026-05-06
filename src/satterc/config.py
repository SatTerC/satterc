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
    source_freq: str
    target_freq: str
    aggfunc: str = "mean"

    @property
    def freq(self) -> str:
        """xarray resample frequency string derived from source_freq/target_freq pair."""
        return _RESAMPLE_FREQ_MAP[(self.source_freq, self.target_freq)]

    @classmethod
    def from_config(cls, entry: dict) -> "ResampleSpec":
        """Construct and validate from a raw [[resample]] TOML entry."""
        source_freq = entry["from_freq"]
        target_freq = entry["to_freq"]
        aggfunc = entry.get("aggfunc", "mean")
        vars_ = entry["vars"]

        if (source_freq, target_freq) not in _RESAMPLE_FREQ_MAP:
            raise ValueError(
                f"Unsupported resample direction '{source_freq}' → '{target_freq}'. "
                f"Supported: {sorted(_RESAMPLE_FREQ_MAP)}"
            )
        if aggfunc not in _VALID_AGGFUNCS:
            raise ValueError(
                f"Unsupported aggfunc '{aggfunc}'. Supported: {sorted(_VALID_AGGFUNCS)}"
            )

        return cls(
            vars=vars_,
            source_freq=source_freq,
            target_freq=target_freq,
            aggfunc=aggfunc,
        )


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

        Recognised top-level sections (processed directly):
        - [inputs.*]      — I/O modules; freq derived from subsection key
        - [outputs.*]     — I/O modules; freq derived from subsection key
        - [models.*]      — built-in model modules
        - [[resample]]    — temporal resampling module

        All other top-level sections are treated as external modules and must
        include a '_import_path = "pkg.module"' key specifying the importable
        module path. The key is stripped before merging remaining params into
        driver_config.
        """
        data = dict(self._data)

        driver_config: dict[str, Any] = {}
        targets: list[str] = []
        modules: list[str] = []

        if "grid" in data:
            data.pop("grid")
            modules.append("grid")

        for freq, params in data.pop("inputs", {}).items():
            if "path" not in params:
                raise ValueError(
                    f"[inputs.{freq}] is missing a 'path' key. "
                    f"Input sections must specify a file path."
                )
            driver_config[f"{freq}_inputs_path"] = params["path"]
            driver_config[f"{freq}_inputs_vars"] = params.get("vars") or []
            driver_config[f"{freq}_inputs_format"] = _infer_format(params["path"])
            modules.append(f"inputs.{freq}")

        for freq, params in data.pop("outputs", {}).items():
            vars_ = params.get("vars") or []
            if not vars_:
                raise ValueError(
                    f"[outputs.{freq}] has no 'vars'. "
                    f"Output sections must list at least one variable, "
                    f"or be removed from the config."
                )
            if "path" not in params:
                raise ValueError(
                    f"[outputs.{freq}] is missing a 'path' key. "
                    f"Output sections must specify a file path."
                )
            driver_config[f"{freq}_outputs_path"] = params["path"]
            driver_config[f"{freq}_outputs_vars"] = vars_
            driver_config[f"{freq}_outputs_format"] = _infer_format(params["path"])
            targets.append(f"save_{freq}_outputs")
            modules.append(f"outputs.{freq}")

        for model_name, params in data.pop("models", {}).items():
            _merge_params(f"models.{model_name}", params, driver_config)
            modules.append(f"models.{model_name}")

        seen_outputs: set[str] = set()
        specs: list[ResampleSpec] = []
        for entry in data.pop("resample", []):
            spec = ResampleSpec.from_config(entry)
            for var in spec.vars:
                out = f"{var}_{spec.target_freq}"
                if out in seen_outputs:
                    raise ValueError(
                        f"Duplicate resample output '{out}' in [[resample]]"
                    )
                seen_outputs.add(out)
            specs.append(spec)
        if specs:
            driver_config["resample_specs"] = specs
            modules.append("resample")

        for section_label, params in data.items():
            params = dict(params)
            import_path = params.pop("_import_path", None)
            if import_path is None:
                raise ValueError(
                    f"Section [{section_label!r}] is missing '_import_path'. "
                    f"All non-built-in sections must include "
                    f"'_import_path = \"pkg.module\"'."
                )
            if not _is_valid_module_path(import_path):
                raise ValueError(
                    f"'_import_path = {import_path!r}' in [{section_label!r}] "
                    f"is not a valid dotted module path."
                )
            _merge_params(section_label, params, driver_config)
            modules.append(import_path)

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


def _is_valid_module_path(path: str) -> bool:
    """Return True if path is a non-empty dotted Python identifier."""
    return bool(path) and all(part.isidentifier() for part in path.split("."))


def _merge_params(section: str, params: dict, driver_config: dict) -> None:
    """Merge params into driver_config, raising ValueError on key conflicts."""
    conflicts = set(params) & set(driver_config)
    if conflicts:
        raise ValueError(
            f"Parameter(s) {sorted(conflicts)} in [{section}] conflict "
            f"with an already-defined key. Use a module-specific prefix to "
            f"disambiguate (e.g. pmodel_method_kphio)."
        )
    driver_config |= params


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
