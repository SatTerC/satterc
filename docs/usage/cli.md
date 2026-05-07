---
title: CLI Guide
icon: lucide/terminal
---

# Using the Command-Line Interface

SatTerC provides a `satterc` command for running pipelines from the terminal.
This guide walks through a typical workflow from setup to execution.

For a complete reference of all commands, arguments, and options, see the [CLI Reference](../api/satterc.cli/index.md).

## The Workflow

A typical SatTerC CLI workflow has four steps:

```
setup → data-gen → graph → run
```

1. **Generate a config** with `satterc setup`
2. **Create test data** with `satterc data-gen generate`
3. **Visualise the pipeline** with `satterc graph`
4. **Execute the pipeline** with `satterc run`

Let's walk through each step.

## Step 1: Generate a Configuration

The `setup` command creates a TOML configuration file. You can run it interactively or with defaults.

### Interactive mode

```sh
satterc setup
```

This walks you through:

1. Selecting built-in models (type numbers or names, toggle with re-entry)
2. Optionally adding custom module paths
3. Confirming input/output paths (or entering custom ones)
4. Optionally generating synthetic data right away

### Non-interactive mode

```sh
satterc setup --models splash pmodel --defaults
```

This generates a config with the specified models and placeholder paths, no prompts.

### Custom output path

```sh
satterc setup --models splash --output my_pipeline.toml
```

The generated config includes all input variables required by the selected models, placeholder output sections, and any resampling steps needed to bridge temporal frequencies.

## Step 2: Generate Synthetic Data

Before running on real data, test your pipeline with synthetic inputs:

```sh
satterc data-gen generate config.toml
```

This creates NetCDF files at the paths specified in your config. By default it generates data for a single site over 2 years.

### Custom grid and duration

```sh
satterc data-gen generate config.toml --grid 4,4 --duration 6m --seed 42
```

This produces a 4×4 grid of synthetic data covering 6 months.

The duration format is a number followed by a unit: `2y` (years), `6m` (months), `30d` (days).

## Step 3: Visualise the Pipeline

Before running, inspect the DAG to verify the structure looks correct:

```sh
satterc graph config.toml --pdf
```

This produces `pipeline.pdf` showing all nodes and their dependencies. The graph is colour-coded:

| Colour | Frequency |
|--------|-----------|
| Aquamarine | Static inputs |
| Orange | Daily |
| Yellow | Weekly |
| Brown | Monthly |

You can also output as PNG:

```sh
satterc graph config.toml --png
```

/// admonition | Note
    type: note

Requires [graphviz](https://graphviz.org/) to be installed.
///

## Step 4: Run the Pipeline

Execute the pipeline:

```sh
satterc run config.toml
```

This reads the config, builds the DAG, executes all required nodes in dependency order, and writes output files as specified in the `[outputs.*]` sections.

## Inspecting Results

The output files are NetCDF (or whatever format you specified). Load them in Python:

```python
import xarray as xr

ds = xr.open_dataset("outputs/daily.nc")
print(ds)
ds["soil_moisture"].plot()
```

## Getting Help

Every command supports `-h` / `--help`:

```sh
satterc -h
satterc setup -h
satterc data-gen generate -h
```

For detailed documentation on each CLI module's functions and parameters, see the [CLI Reference](../api/satterc.cli/index.md).
