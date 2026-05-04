---
title: 00 Getting Started Csv
marimo-version: 0.23.4
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

# Getting started with SatTerC (CSV inputs)

This notebook mirrors the main *Getting started* notebook but loads input data from
a **CSV file** (daily timeseries) and a **JSON file** (static variables) instead of
NetCDF.  It is aimed at users working with single-site data who do not have gridded
files to hand.  The pipeline, configuration syntax, and outputs are identical.
<!---->
## Running this notebook

### Option A — standalone, using `uv` (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager and installer.
If you have it installed, download this file and run:

```bash
uv run 00-getting-started-csv.py
```

`uv` will read the dependency list embedded at the top of this file, install everything
it needs into a temporary isolated environment, and open the notebook in your browser.
You do not need to install SatTerC separately.

### Option B — using an existing Python environment

If SatTerC is already installed in a Python environment (for example, the project
development environment), activate that environment and run:

```bash
marimo run 00-getting-started-csv.py
```

```python {.marimo}
import json
import tempfile
import tomllib
from pathlib import Path

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from satterc import build_driver
from satterc.config import Config
```

## Step 1: Configure the pipeline

A SatTerC pipeline is described by a configuration file written in
[TOML](https://toml.io/en/) — a simple, human-readable format.
Every section in the config activates a pipeline component — `[models.splash]` runs the
SPLASH water-balance model, `[inputs.daily]` loads daily climate data from the given path,
and `[outputs.daily]` saves the named variables to disk when the pipeline finishes.

SatTerC detects the file format automatically from the extension, so pointing
`path` at a `.csv` file is all that is needed to switch from NetCDF to flat-file inputs.
Static variables are loaded from a `.json` file — a plain key→value mapping.
Note that `latitude` is included as a static variable here: in the gridded NetCDF
workflow it is derived from the CRS via `[inputs.grid]`, but for single-point data
it is most natural to treat it as a site property alongside `elevation`.

```python {.marimo}
config_toml = """
[models.splash]

[inputs.daily]
path = "daily.csv"
vars = [
  "precipitation_mm",
  "sunshine_fraction",
  "temperature_celcius",
]

[inputs.static]
path = "static.json"
vars = [
  "elevation",
  "latitude",
  "max_soil_moisture",
]

[outputs.daily]
path = "results/daily.csv"
vars = [
  "actual_evapotranspiration",
  "soil_moisture",
  "runoff",
]
"""
```

## Step 2: Generate synthetic input data

SatTerC's built-in data generator produces gridded NetCDF files, which is ideal
for spatial workflows.  For single-site CSV inputs we instead create the flat files
directly with [pandas](https://pandas.pydata.org/) and write them to a temporary
directory.

The data covers **one site** over a **two-year period**, using the same seasonal-cycle
logic as the built-in generator.

> **If you have real data**, skip ahead to the *Using your own data* section at the
> bottom of this notebook before running the pipeline.

```python {.marimo}
_tmpdir = Path(tempfile.mkdtemp())

# Parse the embedded config string
parsed_config = Config(tomllib.loads(config_toml)).parse()

# Redirect input paths to files we will generate in the temporary directory
parsed_config["driver_config"]["daily_inputs_path"] = str(_tmpdir / "daily.csv")
parsed_config["driver_config"]["static_inputs_path"] = str(_tmpdir / "static.json")

# --- Daily CSV ---
np.random.seed(42)
_n_days = 730
_t = np.arange(_n_days)
_dates = pd.date_range("2020-01-01", periods=_n_days, freq="D")

_temperature = (
    10.0
    + 10.0 * np.sin(2 * np.pi * _t / 365.25 - np.pi / 2)
    + np.random.uniform(-3, 3, _n_days)
)
_daily_precip = np.random.exponential(
    3.1 + np.sin(2 * np.pi * _t / 365.25), _n_days
)
_precipitation = np.where(np.random.random(_n_days) < 0.6, _daily_precip, 0.0)
_sunshine_fraction = np.clip(
    0.5
    + 0.3 * np.sin(2 * np.pi * _t / 365.25)
    + np.random.uniform(-0.15, 0.15, _n_days),
    0.0,
    1.0,
)

_df = pd.DataFrame(
    {
        "precipitation_mm": _precipitation,
        "sunshine_fraction": _sunshine_fraction,
        "temperature_celcius": _temperature,
    },
    index=_dates,
)
_df.index.name = "time"
_df.to_csv(_tmpdir / "daily.csv")

# --- Static JSON ---
with open(_tmpdir / "static.json", "w") as _f:
    json.dump(
        {"elevation": 100.0, "latitude": 52.0, "max_soil_moisture": 200.0},
        _f,
        indent=2,
    )

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

By default, the pipeline saves results to files on disk (the `save_*_outputs` nodes).
Here we instead request the merged output dataset directly as an in-memory object
— useful for exploration and plotting without writing any files.

```python {.marimo}
outputs = dr.execute(["merged_daily_outputs"])
outputs
```

## Step 5: Inspect the results

Let us plot the simulated soil moisture over the two-year period.
Soil moisture rises after precipitation events and falls during dry periods —
a clear seasonal signal should be visible.

```python {.marimo}
_outputs = dr.execute(["soil_moisture_daily"])
soil_moisture = _outputs["soil_moisture_daily"].isel(pixel=0)

fig, ax = plt.subplots(figsize=(10, 3))
soil_moisture.plot(ax=ax)
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

### 2. Prepare your data files

**Daily CSV** — one row per day, first column a parseable date:

```csv
time,precipitation_mm,sunshine_fraction,temperature_celcius
2020-01-01,3.2,0.45,8.1
2020-01-02,0.0,0.71,9.3
...
```

**Static JSON** — a plain key → scalar mapping:

```json
{
  "elevation": 150.0,
  "latitude": 51.5,
  "max_soil_moisture": 200.0
}
```

### 3. Edit the config file

Open `my_pipeline.toml` in a text editor.
Under each `[inputs.*]` section, change the `path` value to point to your real file.
Paths can be absolute or relative to the location of the config file. For example:

```toml
[inputs.daily]
path = "/data/my-site/daily.csv"
vars = [
  "precipitation_mm",
  "sunshine_fraction",
  "temperature_celcius",
]

[inputs.static]
path = "/data/my-site/static.json"
vars = [
  "elevation",
  "latitude",
  "max_soil_moisture",
]
```

### 4. Load the config from the file

Replace the config and data-generation cells in this notebook with:

```python
from satterc import load_config

parsed_config = load_config("my_pipeline.toml")
```

`load_config` reads the TOML file and resolves all paths relative to the file's location.

### 5. Remove the data generation cell

You no longer need to generate synthetic data — delete that cell.
The pipeline will load your real CSV and JSON files directly.