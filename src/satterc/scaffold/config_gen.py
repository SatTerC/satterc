"""Configuration generator for SatTerC.

Generates a config file by introspecting the chosen model modules: which inputs
they need, at which frequencies, and which of those another model already
produces.
"""

import inspect
import typing
from importlib import import_module
from types import ModuleType
from typing import Any

import pint
import xarray as xr
from conduit.config import Config
from hamilton import driver
from hamilton.settings import ENABLE_POWER_USER_MODE
from xarray_annotated import unwrap_annotated

from .. import models as model_modules
from .. import temporal
from ..temporal import resample_offset
from .bridges import Bridge, bridge_for


def _analyze_model_module(
    module: ModuleType, config: dict[str, Any], units: dict[str, str] | None = None
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

    Declared units are collected into ``units`` as a side table, keyed by node
    name. `_resample_entry` needs them to tell a plain aggregation from a
    rate-to-total conversion.
    """
    dr = driver.Builder().with_modules(module).with_config(config).build()
    all_vars = dr.list_available_variables()
    units = {} if units is None else units

    data_inputs = []
    data_outputs = []
    non_data_inputs = []

    # Inputs carry their declared units on the node type; outputs do not, because
    # `@extract_fields` drops the `Annotated` metadata when it splits a model's
    # return TypedDict into one node per field. Read those from the TypedDict.
    for name, hint in _output_hints(module).items():
        for meta in getattr(hint, "__metadata__", ()):
            if isinstance(meta, str):
                units.setdefault(name, meta)
                break

    for v in all_vars:
        for meta in getattr(v.type, "__metadata__", ()):
            if isinstance(meta, str):
                units.setdefault(v.name, meta)
                break
        # Signature-native unit declarations make some node types
        # ``Annotated[DataArray, "<unit>"]``; unwrap so they still count as data.
        v_type = unwrap_annotated(v.type)
        if v.is_external_input:
            if v_type == xr.DataArray:
                data_inputs.append(v.name)
            else:
                non_data_inputs.append(v.name)
        else:
            if v_type == xr.DataArray:
                data_outputs.append(v.name)

    return data_inputs, non_data_inputs, data_outputs


def _output_hints(module: ModuleType) -> dict[str, Any]:
    """Return each node function's declared output type, keyed by node name.

    A model returns a TypedDict that `@extract_fields` splits into one node per
    field. That decorator does not carry the fields' `Annotated` metadata onto
    the resulting nodes, so the units are only reachable through the return
    annotation itself.
    """
    hints: dict[str, Any] = {}
    for obj in vars(module).values():
        if not callable(obj) or getattr(obj, "__module__", None) != module.__name__:
            continue
        try:
            returned = typing.get_type_hints(obj, include_extras=True).get("return")
        except (NameError, TypeError):
            continue
        if returned is None:
            continue
        if typing.is_typeddict(returned):
            hints.update(typing.get_type_hints(returned, include_extras=True))
        else:
            hints.setdefault(getattr(obj, "__name__", ""), returned)
    return hints


def _strip_suffix(name: str) -> tuple[str, str | None]:
    """Strip a frequency suffix from a variable name.

    The suffixes are a satterc naming convention, not framework behaviour, and
    match the `Freq` contracts the model modules declare (see
    `satterc.temporal`). They decide which input file a variable belongs in.

    Parameters
    ----------
    name : str
        Variable name (e.g., 'temperature_daily').

    Returns
    -------
    tuple
        (base_name, frequency) or (name, None) if no suffix found.
    """
    for suffix in ("_daily", "_weekly", "_monthly", "_static"):
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    return name, None


def get_builtin_models() -> list[str]:
    """Get list of builtin models."""
    from ..scaffold import BuiltinModels

    return [m.value for m in BuiltinModels]


def get_model_params(model_name: str) -> dict[str, Any]:
    """Extract keyword-only parameters with defaults from a model's nodes.

    Every public function the module defines is read, not just the model node.
    A module's settings are gathered into ``<node>_params`` container nodes (see
    `satterc.models.splash.splash_params`), and a module with more than one
    settings-bearing node has more than one such container — RothC's per-PFT
    DPM/RPM ratios sit on `satterc.models.rothc.dpm_rpm_ratio_params`, not on
    the model node. Scanning the module finds all of them, and the result is
    still one flat dict because conduit merges a section's keys into one flat
    driver config regardless of which node consumes them.
    """
    builtin_models = get_builtin_models()
    module_path = (
        f"satterc.models.{model_name}" if model_name in builtin_models else model_name
    )

    try:
        module = import_module(module_path)
    except ImportError:
        return {}

    params: dict[str, Any] = {}
    for name, func in vars(module).items():
        if name.startswith("_") or not inspect.isfunction(func):
            continue
        if func.__module__ != module.__name__:
            continue
        params.update(
            {
                p.name: p.default
                for p in inspect.signature(func).parameters.values()
                if p.kind == inspect.Parameter.KEYWORD_ONLY
                and p.default is not inspect.Parameter.empty
            }
        )
    return params


#: Suffixes ordered fine to coarse, so an index comparison answers "can this be
#: resampled onto that?" — resampling only ever coarsens.
_SUFFIX_ORDER = ("_daily", "_weekly", "_monthly")


def _label(suffix: str | None) -> str:
    """Turn ``"_daily"`` into ``"daily"``, the form config labels use."""
    return (suffix or "_static").lstrip("_")


def _finest_source(produced: set[str | None], wanted: str | None) -> str | None:
    """Return the finest produced frequency that could supply ``wanted``.

    ``None`` when nothing produced can: either the base is not a model output at
    all, or every frequency it is produced at is coarser than what the consumer
    wants. Resampling coarsens only, so a coarser producer is no help.
    """
    if wanted not in _SUFFIX_ORDER:
        # Static: a producer at any frequency is a time series, not a constant.
        return wanted if wanted in produced else None
    candidates = [
        freq
        for freq in produced
        if freq in _SUFFIX_ORDER
        and _SUFFIX_ORDER.index(freq) <= _SUFFIX_ORDER.index(wanted)
    ]
    return min(candidates, key=_SUFFIX_ORDER.index) if candidates else None


def _accumulates(source_units: str | None, target_units: str | None) -> bool:
    """Report whether ``target_units`` is ``source_units`` accumulated over time.

    ``mm d-1`` coarsened onto ``mm`` is a *total*, not a mean: the consumer wants
    the rate integrated over the period. Deciding it from the declared units is
    what keeps this general — nothing here knows which variables are fluxes.

    Returns False when the two agree dimensionally (an ordinary mean) and when
    they are unrelated (a mismatch the contract check should report rather than
    this quietly papering over).
    """
    if not source_units or not target_units:
        return False
    try:
        registry = pint.get_application_registry()
        source = registry.Unit(source_units)
        target = registry.Unit(target_units)
    except (pint.UndefinedUnitError, TypeError, ValueError):
        return False
    if source.dimensionality == target.dimensionality:
        return False
    return (source * registry.Unit("day")).dimensionality == target.dimensionality


def _bridge_factor(
    bridge: Bridge, source_units: str | None, target_units: str | None
) -> float | None:
    """Return the multiplier taking ``source_units`` to ``target_units``.

    A bridge with an explicit ``factor`` uses it: those pairs are not
    dimensionally the same quantity, so no registry can derive them. Otherwise
    the two units are the same quantity written differently and pint knows the
    number — asking it keeps a conversion factor from being restated (and
    eventually mistyped) here.

    ``None`` where the declared units are missing or pint cannot relate them,
    which means the table has drifted from the models and the caller should not
    silently emit a wrong node.
    """
    if bridge.factor is not None:
        return bridge.factor
    if not source_units or not target_units:
        return None
    try:
        registry = pint.get_application_registry()
        return float(registry.Quantity(1.0, source_units).to(target_units).magnitude)
    except (pint.DimensionalityError, pint.UndefinedUnitError, TypeError, ValueError):
        return None


def _infer_required_data(model_names: list[str]) -> dict[str, Any]:
    """Infer required data using analyze_model_module.

    Uses data_inputs from each model to construct input lists,
    filters out model outputs, and determines resample lists.
    """
    base_config = {ENABLE_POWER_USER_MODE: True}

    # Collect all model inputs and outputs
    all_data_inputs: set[str] = set()
    model_output_bases: set[str] = set()
    all_model_outputs: list[str] = []
    units: dict[str, str] = {}

    for model_name in model_names:
        module = getattr(model_modules, model_name)
        data_inputs, _, data_outputs = _analyze_model_module(module, base_config, units)

        # Store full input names (with suffix) for categorization
        all_data_inputs.update(data_inputs)
        all_model_outputs.extend(data_outputs)

        # Track base names of model outputs
        for output in data_outputs:
            base, _ = _strip_suffix(output)
            model_output_bases.add(base)

    # What each model *produces*, and at which frequency. A base may be produced
    # at one frequency and wanted at another, which is what decides below whether
    # it can be resampled or has to be loaded from a file.
    produced_at: dict[str, set[str | None]] = {}
    for output in all_model_outputs:
        base, freq = _strip_suffix(output)
        produced_at.setdefault(base, set()).add(freq)

    # Grid variables come from the grid module, not from files.
    grid_vars = {"latitude", "longitude"}

    daily: set[str] = set()
    weekly: set[str] = set()
    monthly: set[str] = set()
    static: set[str] = set()
    # (base, source_freq, target_freq) for each coarsening a consumer needs.
    resamples: set[tuple[str, str, str]] = set()
    # Coarsenings that also restate the units, so must be nodes rather than
    # `[[resample]]` entries. See `_accumulates`.
    accumulations: list[dict[str, Any]] = []
    # Units-restating nodes for producer/consumer pairs that disagree. Not a
    # coarsening, so tracked apart from `accumulations`. See `scaffold.bridges`.
    bridge_nodes: list[dict[str, Any]] = []

    def _record(base: str, source: str, target: str) -> None:
        """File one coarsening as either a resample or a units-restating node."""
        source_name = f"{base}_{source}"
        target_name = f"{base}_{target}"
        if _accumulates(units.get(source_name), units.get(target_name)):
            # A rate wanted as a per-period total. `[[resample]]` preserves units,
            # so summing there would hand the consumer an `mm d-1` array still
            # labelled as such where it declares `mm`. A derive node can restate
            # them, so emit one of those instead.
            freq_offset = temporal.offset(target)
            accumulations.append(
                {
                    "name": target_name,
                    "inputs": [source_name],
                    "expression": (
                        f"{source_name}.resample(time='{freq_offset}').sum()"
                    ),
                    "units": units[target_name],
                    "freq": freq_offset,
                }
            )
        else:
            resamples.add((base, source, target))

    def _record_bridge(
        base: str, freq: str | None, produced_at: dict[str, set[str | None]]
    ) -> bool:
        """Emit the units-restating node for ``base``, if a bridge supplies it.

        False when no bridge covers this name, when its source is not produced
        at the frequency wanted, or when the factor cannot be established. In
        every case the caller falls through to loading the variable from a file.
        """
        bridge = bridge_for(base)
        if bridge is None or freq not in produced_at.get(bridge.source, set()):
            return False
        source_name = f"{bridge.source}{freq or ''}"
        target_name = f"{base}{freq or ''}"
        target_units = units.get(target_name)
        factor = _bridge_factor(bridge, units.get(source_name), target_units)
        if factor is None or target_units is None:
            return False
        bridge_nodes.append(
            {
                "name": target_name,
                "inputs": [source_name],
                "expression": f"{source_name} * {factor:.6g}",
                "units": target_units,
                "freq": temporal.offset(_label(freq)),
            }
        )
        return True

    by_frequency: dict[str | None, set[str]] = {
        "_daily": daily,
        "_weekly": weekly,
        "_monthly": monthly,
    }

    for name in sorted(all_data_inputs):
        base, freq = _strip_suffix(name)
        if base in grid_vars:
            continue

        produced = produced_at.get(base, set())
        if freq in produced:
            # Produced at exactly the frequency wanted: nothing to load or do.
            continue

        source = _finest_source(produced, freq)
        if source is not None:
            # Produced finer than wanted, so coarsen it rather than loading it.
            _record(base, _label(source), _label(freq))
            continue

        if _record_bridge(base, freq, produced_at):
            # Produced at this frequency under another name, in another model's
            # honest units. Convert it rather than loading a second copy.
            continue
        # Not produced at all, or only *coarser* than wanted. Resampling cannot
        # refine, so this has to come from a file even when a model does produce
        # the same name — pmodel's weekly `gpp` cannot supply sgam's daily one.
        by_frequency.get(freq, static).add(base)

    # A variable loaded at a fine frequency and also wanted coarser is coarsened
    # rather than loaded twice. Priority: daily → weekly → monthly.
    for base in sorted(daily & weekly):
        _record(base, "daily", "weekly")
    for base in sorted(weekly & monthly):
        _record(base, "weekly", "monthly")
    for base in sorted((daily & monthly) - weekly):  # direct hop, no intermediate
        _record(base, "daily", "monthly")

    # A variable that is derived — by either route — must not also be loaded.
    derived: set[tuple[str, str]] = {(base, target) for base, _, target in resamples}
    for entry in accumulations:
        entry_base, entry_suffix = _strip_suffix(entry["name"])
        derived.add((entry_base, _label(entry_suffix)))
    inputs_daily = daily - {b for b, t in derived if t == "daily"}
    inputs_weekly = weekly - {b for b, t in derived if t == "weekly"}
    inputs_monthly = monthly - {b for b, t in derived if t == "monthly"}

    def _pairs(source: str, target: str) -> list[str]:
        return sorted(b for b, s, t in resamples if s == source and t == target)

    resample_daily_to_weekly = _pairs("daily", "weekly")
    resample_daily_to_monthly = _pairs("daily", "monthly")
    resample_weekly_to_monthly = _pairs("weekly", "monthly")

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
        "inputs_daily": sorted(inputs_daily),
        "inputs_weekly": sorted(inputs_weekly),
        "inputs_monthly": sorted(inputs_monthly),
        "inputs_static": sorted(static),
        "bridges": bridge_nodes,
        "accumulations": accumulations,
        "resample_daily_to_weekly": resample_daily_to_weekly,
        "resample_daily_to_monthly": resample_daily_to_monthly,
        "resample_weekly_to_monthly": resample_weekly_to_monthly,
        "outputs_daily": sorted(set(outputs_daily)),
        "outputs_weekly": sorted(set(outputs_weekly)),
        "outputs_monthly": sorted(set(outputs_monthly)),
    }


def generate_config(
    builtin_models: list[str],
    custom_modules: list[str],
    paths: dict[str, str],
) -> Config:
    """Generate a Config object.

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
    Config
        Configuration object.
    """
    required_data = _infer_required_data(builtin_models)

    config_data: dict[str, Any] = {}

    # One flat section per model, in conduit's external-module form. There is no
    # short-name registry to lean on: conduit resolves every non-built-in section
    # by its dotted `_import_path`.
    for model in builtin_models:
        config_data[model] = {
            "_import_path": f"satterc.models.{model}",
            **get_model_params(model),
        }

    # conduit names each input node `{var}{suffix}`, with the suffix defaulting to
    # `_<section label>`. That is exactly satterc's convention for the temporal
    # sections, but static variables are consumed under bare names
    # (`elevation`, not `elevation_static`), so that section opts out of a suffix.
    freq_keys = ("daily", "weekly", "monthly", "static")
    config_data["inputs"] = {
        freq: {
            "path": paths[f"inputs_{freq}"],
            "vars": required_data[f"inputs_{freq}"],
            **({"suffix": ""} if freq == "static" else {}),
        }
        for freq in freq_keys
        if required_data[f"inputs_{freq}"]
    }

    resample_list = []
    for k in (
        "resample_daily_to_weekly",
        "resample_daily_to_monthly",
        "resample_weekly_to_monthly",
    ):
        vars_ = required_data[k]
        if vars_:
            direction = k.removeprefix("resample_")
            from_freq, to_freq = direction.split("_to_")
            resample_list.append(
                {
                    "vars": vars_,
                    "from": from_freq,
                    "to": to_freq,
                    "freq": resample_offset(from_freq, to_freq),
                    # aggfunc omitted → defaults to "mean" at parse time. A
                    # variable needing "sum" is a rate wanted as a total, and is
                    # emitted as a node instead — see `_accumulates`.
                }
            )

    # Nodes before resamples: a reader meets the derivations that change what a
    # variable *means* before the ones that only change its cadence.
    # Bridges first: a units restatement is the shorter story, and the reader
    # meets `gpp_weekly` before anything that consumes it.
    nodes = required_data["bridges"] + required_data["accumulations"]
    if nodes:
        config_data["node"] = nodes
    if resample_list:
        config_data["resample"] = resample_list

    output_freqs = ("daily", "weekly", "monthly")
    config_data["outputs"] = {
        freq: {
            "path": paths[f"outputs_{freq}"],
            "vars": required_data[f"outputs_{freq}"],
        }
        for freq in output_freqs
        if required_data[f"outputs_{freq}"]
    }

    for mod_path in custom_modules:
        # The section label is free-form; only `_import_path` is semantic. Use the
        # module's own last component so the config reads naturally.
        label = mod_path.rsplit(".", 1)[-1]
        config_data[label] = {
            "_import_path": mod_path,
            **get_model_params(mod_path),
        }

    return Config(config_data)
