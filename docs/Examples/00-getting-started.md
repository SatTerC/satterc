---
title: 00 Getting Started
marimo-version: 0.23.0
width: medium
header: |-
  # /// script
  # requires-python = ">=3.13"
  # dependencies = [
  #   "satterc",
  #   "marimo",
  #   "matplotlib",
  # ]
  # ///
---

# Getting started with SatTerC

This notebook walks through running a SatTerC pipeline step by step.
It is aimed at users who are new to SatTerC, and assumes only basic familiarity with Python.
<!---->
## Running this notebook

### Option A — standalone, using `uv` (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager and installer.
If you have it installed, download this file and run:

```bash
uv run 00-getting-started.py
```

`uv` will read the dependency list embedded at the top of this file, install everything
it needs into a temporary isolated environment, and open the notebook in your browser.
You do not need to install SatTerC separately.

### Option B — using an existing Python environment

If SatTerC is already installed in a Python environment (for example, the project
development environment), activate that environment and run:

```bash
marimo run 00-getting-started.py
```

```python {.marimo}
import tempfile
import tomllib
from pathlib import Path

import marimo as mo
import matplotlib.pyplot as plt

from satterc import build_driver
from satterc.config import Config
from satterc.setup_utils.data_gen import generate_synthetic_data
```

## Step 1: Configure the pipeline

A SatTerC pipeline is described by a configuration file written in
[TOML](https://toml.io/en/) — a simple, human-readable format.
The config tells SatTerC:

- **`modules`** — which model components to activate
- **`inputs`** — where to find the input data files and which variables to load from them
- **`outputs`** — which computed variables to save, and where to write them

In this example we run only the **SPLASH** water balance model, which simulates
how precipitation is partitioned into evapotranspiration, soil moisture, and runoff.
It needs just three daily inputs and two static (time-invariant) inputs.

In this notebook the config is embedded directly as a string, so you do not need
any external files to get started. At the end we show how to save it to a file
and adapt it for your own data.

```python {.marimo}
config_toml = """
modules = [
  "models.splash",
  "inputs.daily",
  "inputs.static",
  "outputs.daily",
]

[inputs.daily]
path = "daily.nc"
vars = [
  "precipitation_mm",
  "sunshine_fraction",
  "temperature_celcius",
]

[inputs.static]
path = "static.nc"
vars = [
  "elevation",
  "max_soil_moisture",
]

[outputs.daily]
path = "results/daily.nc"
vars = [
  "actual_evapotranspiration",
  "soil_moisture",
  "runoff",
]
"""
```

## Step 2: Generate synthetic input data

SatTerC reads input data from [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) files
— a standard scientific data format for gridded (spatial) datasets.

Since we do not have real data to hand, we will use SatTerC's built-in synthetic data
generator to produce realistic stand-in inputs.
The generated data covers a small 4×4 grid of pixels over one year.

> **If you have real data**, skip ahead to the *Using your own data* section at the bottom
> of this notebook before running the pipeline.

```python {.marimo}
# Parse the embedded config string
_tmpdir = Path(tempfile.mkdtemp())
parsed_config = Config(tomllib.loads(config_toml)).parse()

# Redirect the input paths to files we will generate in a temporary directory
parsed_config["driver_config"]["daily_inputs_path"] = str(_tmpdir / "daily.nc")
parsed_config["driver_config"]["static_inputs_path"] = str(_tmpdir / "static.nc")

# Generate synthetic data — this may take a few seconds
generate_synthetic_data(config=parsed_config, grid=(4, 4), n_days=730, seed=42)

print(f"Synthetic data written to: {_tmpdir}")
```

## Step 3: Build the pipeline

SatTerC represents a pipeline as a Directed Acyclic Graph (DAG) — a network of nodes
where each node is a computation, and edges show which computations depend on which others.

Building the pipeline means constructing this graph from the modules and configuration
you specified. Below we visualise the portion of the graph running from the daily
precipitation input through to the soil moisture output.

```python {.marimo}
dr = build_driver(
    modules=parsed_config["modules"],
    config=parsed_config["driver_config"],
)
```

```python {.marimo}
dr.visualize_path_between(
    "precipitation_mm_daily",
    "soil_moisture_daily",
    show_legend=False,
    graphviz_kwargs={"graph_attr": {"ratio": "compress", "size": "10,15"}},
)
```

## Step 4: Run the pipeline

We run the pipeline by calling `dr.execute()` and naming the outputs we want.

By default, the pipeline saves results to NetCDF files on disk (the `save_*_outputs` nodes).
Here we instead request the merged output dataset directly as an in-memory object
— useful for exploration and plotting without writing any files.

```python {.marimo}
outputs = dr.execute(["merged_daily_outputs"])
outputs
```

## Step 5: Inspect the results

Let us plot the simulated soil moisture for each pixel over the year.
Soil moisture rises after precipitation events and falls during dry periods —
a clear seasonal signal should be visible.

```python {.marimo}
_outputs = dr.execute(["soil_moisture_daily"])
soil_moisture = _outputs["soil_moisture_daily"]

n_pixels = soil_moisture.sizes["pixel"]
fig, axes = plt.subplots(n_pixels, 1, figsize=(10, 3 * n_pixels), sharex=True)
for i, ax in enumerate(axes):
    soil_moisture[:, i].plot(ax=ax)
    ax.set_title(f"Pixel {i}")
    ax.set_ylabel("Soil moisture (mm)")
fig.tight_layout()
fig
```

## Using your own data

To run the pipeline on real data instead of synthetic data, follow these steps.

### 1. Save the config to a file

Run the cell below. It will write the embedded config to a file called
`my_pipeline.toml` in your current working directory.

```python {.marimo}
_output_path = Path("my_pipeline.toml")

# Uncomment this line!
# _output_path.write_text(config_toml.strip())
# print(f"Config written to: {_output_path.resolve()}")
```

### 2. Edit the file

Open `my_pipeline.toml` in a text editor.
Under each `[inputs.*]` section, change the `path` value to point to your real NetCDF file.
Paths can be absolute or relative to the location of the config file. For example:

```toml
[inputs.daily]
path = "/data/my-site/daily.nc"
vars = [
  "precipitation_mm",
  "sunshine_fraction",
  "temperature_celcius",
]
```

### 3. Load the config from the file

Replace the config and data-generation cells in this notebook with:

```python
from satterc import load_config

parsed_config = load_config("my_pipeline.toml")
```

`load_config` reads the TOML file and resolves all paths relative to the file's location.

### 4. Remove the data generation cell

You no longer need to generate synthetic data — delete that cell.
The pipeline will load your real NetCDF files directly.