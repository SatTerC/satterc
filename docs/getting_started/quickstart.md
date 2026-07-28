---
title: Quickstart
icon: lucide/rocket
---

# Quickstart: Your First Pipeline

Get a SatTerC pipeline running in five minutes. This guide walks through a
minimal pipeline that computes soil moisture and evapotranspiration from
synthetic data.

## Prerequisites

Follow the [Installation guide](installation.md) to install SatTerC and its
optional dependencies.

## Step 1: Generate a Config

Use the interactive setup command to create a configuration file:

```sh
satterc setup
```

This will prompt you to select models. Choose `splash` for this tutorial, and
accept the default paths when prompted.

Or skip the prompts and use defaults:

```sh
satterc setup --models splash --defaults
```

This creates a `config.toml` file that looks like:

```toml
[splash]
_import_path = "satterc.models.splash"
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

A model is just a section carrying `_import_path`: SatTerC's models are ordinary
conduit modules, with nothing special about them beyond being shipped in this
package. Node names are `{var}{suffix}`, and the suffix defaults to the section
label — so `temperature` under `[inputs.daily]` becomes the node
`temperature_daily`, which is what `splash`'s parameter is called. Static
variables are consumed under bare names, so that section sets `suffix = ""`.

See conduit's [configuration reference][conduit-config] for every section.

## Step 2: Generate Synthetic Data

Create test data from the config:

```sh
mkdir -p inputs
satterc data-gen generate config.toml --grid 1 1 --duration 2y --seed 42
```

This creates NetCDF files at the paths specified in your config.

/// admonition | SPLASH needs more than a year
    type: note

SPLASH estimates its initial soil moisture by iterating over a full year, so
`--duration 1y` fails with "Cannot equilibrate - less than one year of data".
Use `2y` or more.
///

## Step 3: Check the Pipeline Before Running It

```sh
satterc run config.toml --dry-run
```

This parses the config, loads the inputs, builds the DAG and validates every
declared contract — units, dimensions and temporal frequency — without computing
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

## Step 4: Visualise the Pipeline

See what the DAG looks like:

```sh
satterc graph config.toml --pdf
```

This produces `pipeline.pdf` showing all nodes and their dependencies, with nodes
grouped and coloured by their declared frequency.

## Step 5: Run the Pipeline

```sh
mkdir -p outputs
satterc run config.toml
```

This reads the input data, executes the DAG, and writes `outputs/daily.nc`. The
config that produced it — and its SHA-256 — are stamped into the file's
attributes, so a result can always be traced back to its inputs.

## Step 6: Inspect the Results

Load the output in Python:

```python
import xarray as xr

ds = xr.open_dataset("outputs/daily.nc")
print(ds)
ds["soil_moisture"].plot()
```

## Next Steps

- Learn about the [built-in models](../models/index.md) and how to compose them
- Browse the Examples for interactive notebooks (run `just export-all` to generate)
- Read conduit's [contracts][conduit-contracts] page for what the checks can and
  cannot catch
- Read conduit's [scale-up guide][conduit-scale] for caching, blocking and
  parallel subset runs

[conduit-config]: https://github.com/NERC-CEH/conduit/blob/develop/docs/reference/configuration.md
[conduit-contracts]: https://github.com/NERC-CEH/conduit/blob/develop/docs/concepts/contracts.md
[conduit-scale]: https://github.com/NERC-CEH/conduit/blob/develop/docs/guides/scale-up.md
