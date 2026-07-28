"""Generate synthetic input data for a satterc config.

Everything is generated at daily resolution — from the tables in
`satterc.setup_utils.data_gen.daily` / `.static` where the variable has an entry,
and from a name-heuristic fallback otherwise — then aggregated to the coarser
files with `conduit.transforms.resample`.

Generation is driven by `satterc.setup_utils.data_gen.spec.Resolver`, which
memoises each variable and resolves dependencies between them on demand. It is
deliberately not a Hamilton DAG: the pipeline's own graph is conduit's business,
whereas this is a handful of generators in one process, and a plain resolver
keeps the tables free of framework wiring.
"""

import json
from os import PathLike
from pathlib import Path

import xarray as xr
from conduit import IOSpec, ParsedConfig
from conduit.formats import dataset_to_frame, format_for, write_frame, write_in_group
from conduit.gridded.io import unstack_if_gridded
from conduit.transforms import resample

from .daily import DAILY_VARS
from .fallback import fallback_var
from .spec import Grid, Resolver
from .static import STATIC_VARS

#: Target offsets for the coarser input files. Mirrors
#: `satterc.setup_utils.config_gen.RESAMPLE_FREQ_MAP` and the `Freq` contracts the
#: model modules declare, so generated data satisfies them.
_WEEKLY_FREQ = "7D"
_MONTHLY_FREQ = "1ME"


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


def generate_synthetic_data(
    config: ParsedConfig,
    grid: tuple[int, int],
    n_days: int,
    seed: int = 42,
) -> None:
    """Generate synthetic input data for every input section of a config.

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
        Random seed for reproducibility. Each variable draws from its own stream,
        derived from this seed and the variable's name, so its values do not
        depend on what else the config asked for.
    """
    n_lat, n_lon = grid

    daily_spec = config.input_specs.get("daily")
    weekly_spec = config.input_specs.get("weekly")
    monthly_spec = config.input_specs.get("monthly")
    static_spec = config.input_specs.get("static")

    daily_vars = set(_spec_vars("daily", daily_spec))
    weekly_vars = set(_spec_vars("weekly", weekly_spec))
    monthly_vars = set(_spec_vars("monthly", monthly_spec))
    static_vars = _spec_vars("static", static_spec)

    resolver = Resolver(
        grid=Grid(n_lat=n_lat, n_lon=n_lon, n_days=n_days),
        daily_vars=DAILY_VARS,
        static_vars=STATIC_VARS,
        fallback=fallback_var,
        seed=seed,
    )

    def _at(var: str, freq: str | None) -> xr.DataArray:
        # Everything temporal is produced daily and aggregated here, so a
        # variable that only ever appears in the weekly or monthly file is still
        # generated by a daily entry (or a fallback).
        source = resolver.daily(var)
        return source if freq is None else resample(source, freq=freq)

    sections = [
        (daily_vars, daily_spec, None),
        (weekly_vars, weekly_spec, _WEEKLY_FREQ),
        (monthly_vars, monthly_spec, _MONTHLY_FREQ),
    ]
    for names, spec, freq in sections:
        if names and spec:
            ds = unstack_if_gridded(xr.merge([_at(v, freq) for v in sorted(names)]))
            _save_dataset_with_crs(ds, spec.path)

    if static_vars and static_spec:
        static_ds = unstack_if_gridded(
            xr.merge([resolver.static(v) for v in static_vars])
        )
        _save_dataset_with_crs(static_ds, static_spec.path)
