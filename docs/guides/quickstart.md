---
title: Quickstart
icon: lucide/rocket
---

# Quickstart

This page runs a one-model pipeline end to end on synthetic data: SPLASH, which
computes soil moisture, evapotranspiration and runoff from daily climate. No
real data is needed.

## Prerequisites

Follow the [Installation guide](installation.md) to install SatTerC and its
optional dependencies.

## Step 1: generate a config

The interactive setup command writes a config file:

```sh
satterc setup
```

It asks which models you want. Choose `splash`, and accept the default paths.

To skip the prompts:

```sh
satterc setup --models splash --defaults
```

/// admonition | `setup` is experimental
    type: warning

The command and the configs it generates may change. See the
[CLI reference](../reference/cli.md#satterc-setup) for its current limitations.
///

This creates a `config.toml` file that looks like:

```toml
[splash]
soil_moisture_init_method = "equilibrate"
soil_moisture_init_max_iter = 10
soil_moisture_init_max_diff = 1.0

[inputs.daily]
path = "inputs/daily.nc"
vars = [
    "precipitation",
    "sunshine_fraction",
    "temperature",
]

[inputs.static]
path = "inputs/static.nc"
vars = [
    "elevation",
    "max_soil_moisture",
]
suffix = ""

[outputs.daily]
path = "outputs/daily.nc"
vars = [
    "actual_evapotranspiration",
    "runoff",
    "soil_moisture",
]
```

A model is a section named after the model.
SatTerC's models are ordinary catena modules; SatTerC registers them with catena, so a section resolves to a module by its name alone.
A module catena has no registration for needs `_import_path` to say where it lives.
Node names are `{var}{suffix}`, and the suffix defaults to the section label, so
`temperature` under `[inputs.daily]` becomes the node `temperature_daily`, which
is what `splash`'s parameter is called. Static variables are consumed under bare
names, so that section sets `suffix = ""`.

See catena's [configuration reference][catena-config] for every section.

## Step 2: generate synthetic data

The config says which variables the pipeline needs, so the generator can work
from it:

```sh
mkdir -p inputs
satterc data-gen config.toml --grid 1 1 --duration 2y --seed 42
```

This writes NetCDF files at the input paths named in the config.

/// admonition | SPLASH needs a full calendar year
    type: note

SPLASH estimates its initial soil moisture by iterating over the first calendar
year from the start date. `--duration 1y` generates 365 days, and the default
start date falls in a leap year, so it fails with "Cannot equilibrate - less than
one year of data". Use `2y` or more.
///

## Step 3: check the pipeline before running it

```sh
satterc run config.toml --dry-run
```

This parses the config, loads the inputs, builds the DAG and validates every
declared contract (units, dimensions and temporal frequency) without computing
anything:

```
Dry run for config.toml
  ✓ config parsed
  ✓ inputs loaded: 7 variable(s) from 2 source(s)
  - input checks: none configured
  ✓ DAG built (static contract check passed)
  ✓ execution plan valid: 3 output node(s) reachable
  ✓ input contracts validated (...)
  ✓ output paths writable: 1 destination(s)
Dry run passed.
```

## Step 4: visualise the pipeline

Draw the DAG:

```sh
satterc graph config.toml --pdf
```

This writes `pipeline.pdf`: every node and its dependencies, grouped and coloured
by declared frequency.

## Step 5: run the pipeline

```sh
mkdir -p outputs
satterc run config.toml
```

This reads the inputs, executes the DAG and writes `outputs/daily.nc`. The config
that produced it, and its SHA-256, are stamped into the file's attributes, so you
can trace a result back to the config that made it.

## Step 6: inspect the results

Load the output in Python:

```python
import xarray as xr

ds = xr.open_dataset("outputs/daily.nc")
print(ds)
ds["soil_moisture"].plot()
```

## Next steps

- Read about the [built-in models](../models/index.md) and how to compose them
- Browse the [recipes](../recipes/my_first_pipeline.md) for interactive notebooks (run `just export-all` to generate)
- Read catena's [contracts][catena-contracts] page for what the checks can and
  cannot catch
- Read catena's [scale-up guide][catena-scale] for caching, blocking and
  parallel subset runs

[catena-config]: https://github.com/NERC-CEH/catena/blob/develop/docs/reference/configuration.md
[catena-contracts]: https://github.com/NERC-CEH/catena/blob/develop/docs/concepts/contracts.md
[catena-scale]: https://github.com/NERC-CEH/catena/blob/develop/docs/guides/scale-up.md
