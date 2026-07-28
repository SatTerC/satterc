"""Generate synthetic input data for a satterc config.

Everything is generated at daily resolution — by an explicit generator in
`satterc.setup_utils.data_gen.daily` / `.static` where one exists, and by a
name-heuristic fallback otherwise — then aggregated to the coarser files with
`conduit.transforms.resample`.

The aggregation happens *after* the DAG rather than inside it. conduit's resample
is a plain function lowered into a config-generated node, not a Hamilton module
that can be added to a driver, so there is nothing to wire in here; calling it
directly is both simpler and closer to what the pipeline itself will do.
"""

import inspect
import json
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from conduit import IOSpec, ParsedConfig
from conduit.formats import dataset_to_frame, format_for, write_frame, write_in_group
from conduit.gridded.io import unstack_if_gridded
from conduit.transforms import resample
from hamilton import driver
from hamilton.settings import ENABLE_POWER_USER_MODE

from . import daily, static
from .fallback import build_fallback_module

#: Target offsets for the coarser input files. Mirrors
#: `satterc.setup_utils.config_gen.RESAMPLE_FREQ_MAP` and the `Freq` contracts the
#: model modules declare, so generated data satisfies them.
_WEEKLY_FREQ = "7D"
_MONTHLY_FREQ = "1ME"


def _set_random_seed(seed: int) -> None:
    np.random.seed(seed)


def _spec_vars(label: str, spec: "IOSpec | None") -> list[str]:
    """Return the file variable names an input section asks for.

    conduit lets ``vars`` be omitted (bind everything in the file) or given as a
    ``{node = file_var}`` mapping. Generation runs before the file exists, so the
    omitted form has nothing to enumerate and is rejected here; the mapping form
    is read for its file-side names, which are what gets written.
    """
    if spec is None:
        return []
    if spec.vars is None:
        raise ValueError(
            f"[inputs.{label}] omits 'vars', so there is nothing to generate: "
            f"synthetic data is built from the variable list, and the file it "
            f"would otherwise be read from does not exist yet. List the "
            f"variables explicitly to generate this section."
        )
    if isinstance(spec.vars, dict):
        return list(spec.vars.values())
    return list(spec.vars)


def _save_dataset_with_crs(ds: xr.Dataset, path: str | PathLike) -> None:
    """Save a dataset in whichever format ``path``'s extension names.

    Dispatch is conduit's — `conduit.formats` is the one table of supported
    formats — with two additions. Tabular formats (CSV, Parquet) are flattened to
    a time-indexed frame and cannot carry CRS; JSON is written here because
    conduit reads it but does not write it, and a single-pixel static section is
    the one case that wants it. NetCDF and Zarr get a CRS global attribute so
    `conduit.io.load_inputs` takes its geospatial path on the way back in.
    """
    p = Path(path)
    fmt = format_for(p)

    if fmt.key == "json":
        data = {str(var): float(ds[var].values.flat[0]) for var in ds.data_vars}
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
        return

    if fmt.group == "table":
        write_frame(dataset_to_frame(ds), p)
        return

    ds.attrs["crs"] = "EPSG:4326"
    write_in_group(ds, p, "dataset")


def _known_daily_fns() -> set[str]:
    return {
        name
        for name, obj in inspect.getmembers(daily, inspect.isfunction)
        if not name.startswith("_")
    }


def _known_static_fns() -> set[str]:
    return {
        name
        for name, obj in inspect.getmembers(static, inspect.isfunction)
        if not name.startswith("_")
    }


def generate_synthetic_data(
    config: ParsedConfig,
    grid: tuple[int, int],
    n_days: int,
    seed: int = 42,
) -> None:
    """Generate synthetic input data using a Hamilton DAG.

    Parameters
    ----------
    config : ParsedConfig
        Parsed configuration from `conduit.load_config`. Input paths in
        ``config.input_specs`` are used as the destinations for the generated
        files.
    grid : tuple[int, int]
        Grid dimensions as (n_lat, n_lon).
    n_days : int
        Number of days to generate.
    seed : int
        Random seed for reproducibility.
    """
    _set_random_seed(seed)

    n_lat, n_lon = grid

    daily_spec = config.input_specs.get("daily")
    weekly_spec = config.input_specs.get("weekly")
    monthly_spec = config.input_specs.get("monthly")
    static_spec = config.input_specs.get("static")

    daily_vars: set[str] = set(_spec_vars("daily", daily_spec))
    weekly_vars: set[str] = set(_spec_vars("weekly", weekly_spec))
    monthly_vars: set[str] = set(_spec_vars("monthly", monthly_spec))
    static_vars: list[str] = _spec_vars("static", static_spec)

    driver_config: dict[str, Any] = {
        ENABLE_POWER_USER_MODE: True,
        "n_lat": n_lat,
        "n_lon": n_lon,
        "n_days": n_days,
        "start_date": "2020-01-01",
        "seed": seed,
    }

    # Everything temporal is produced daily and aggregated below, so a variable
    # that only ever appears in the weekly or monthly file still needs a daily
    # generator (or a fallback).
    all_temporal_vars = daily_vars | weekly_vars | monthly_vars
    monthly_targets = all_temporal_vars

    known_daily = _known_daily_fns()
    known_static = _known_static_fns()
    unknown_daily = [v for v in all_temporal_vars if f"{v}_daily" not in known_daily]
    unknown_static = [v for v in static_vars if v not in known_static]

    modules = [daily, static]
    if unknown_daily or unknown_static:
        modules.append(build_fallback_module(unknown_daily, unknown_static))

    dr = (
        driver.Builder()
        .with_modules(*modules)
        .with_config(driver_config)
        .allow_module_overrides()
        .build()
    )

    targets = [f"{v}_daily" for v in sorted(all_temporal_vars)] + static_vars
    results = dr.execute(targets)  # type: ignore[reportArgumentType]

    def _at(var: str, freq: str | None) -> xr.DataArray:
        source = results[f"{var}_daily"]
        return source if freq is None else resample(source, freq=freq)

    if daily_vars and daily_spec:
        daily_ds = unstack_if_gridded(
            xr.merge([_at(v, None) for v in sorted(daily_vars)])
        )
        _save_dataset_with_crs(daily_ds, daily_spec.path)

    if weekly_vars and weekly_spec:
        weekly_ds = unstack_if_gridded(
            xr.merge([_at(v, _WEEKLY_FREQ) for v in sorted(weekly_vars)])
        )
        _save_dataset_with_crs(weekly_ds, weekly_spec.path)

    if monthly_targets and monthly_spec:
        monthly_ds = unstack_if_gridded(
            xr.merge([_at(v, _MONTHLY_FREQ) for v in sorted(monthly_targets)])
        )
        _save_dataset_with_crs(monthly_ds, monthly_spec.path)

    if static_spec:
        static_ds = unstack_if_gridded(xr.merge([results[v] for v in static_vars]))  # type: ignore[reportArgumentType]
        _save_dataset_with_crs(static_ds, static_spec.path)
