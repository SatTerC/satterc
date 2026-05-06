---
title: 00 Getting Started Csv
marimo-version: 0.23.5
width: medium
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

SatTerC reads input data from files — the format is detected automatically from
the file extension listed in the configuration. SatTerC's built-in synthetic
data generator supports NetCDF, Zarr, CSV, Parquet, and JSON output, making
it easy to produce realistic stand-in inputs in whatever format you need.

Since we do not have real data to hand, we will use the generator to produce
single-site CSV and JSON inputs. The generated data covers one virtual site
over a two-year period.

> **If you have real data**, skip ahead to the *Using your own data* section at the bottom
> of this notebook before running the pipeline.

```python {.marimo}
_tmpdir = Path(tempfile.mkdtemp())

# Parse the embedded config string
parsed_config = Config(tomllib.loads(config_toml)).parse()

# Redirect the input paths to files we will generate in a temporary directory
parsed_config.driver_config["daily_inputs_path"] = str(_tmpdir / "daily.csv")
parsed_config.driver_config["static_inputs_path"] = str(_tmpdir / "static.json")

# Generate synthetic data — this may take a few seconds
generate_synthetic_data(config=parsed_config, grid=(1, 1), n_days=730, seed=42)

print(f"Synthetic data written to: {_tmpdir}")
```

<!-- @output:Xref -->

<pre style="white-space: pre; overflow-x: auto;">Synthetic data written to: /tmp/tmp0nscc6eq
</pre>

## Step 3: Build the pipeline

SatTerC represents a pipeline as a Directed Acyclic Graph (DAG) — a network of nodes
where each node is a computation, and edges show which computations depend on which others.

Building the pipeline means constructing this graph from the modules and configuration
you specified. Below we visualise the portion of the graph running from the daily
precipitation input through to the soil moisture output.

```python {.marimo}
dr = build_driver(
    modules=parsed_config.modules,
    config=parsed_config.driver_config,
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

<!-- @output:RGSE -->

<img src="&lt;?xml version=&quot;1.0&quot; encoding=&quot;UTF-8&quot; standalone=&quot;no&quot;?&gt;
&lt;!DOCTYPE svg PUBLIC &quot;-//W3C//DTD SVG 1.1//EN&quot;
 &quot;http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd&quot;&gt;
&lt;!-- Generated by graphviz version 2.43.0 (0)
 --&gt;
&lt;!-- Title: %3 Pages: 1 --&gt;
&lt;svg width=&quot;720pt&quot; height=&quot;759pt&quot;
 viewBox=&quot;0.00 0.00 720.00 758.94&quot; xmlns=&quot;http://www.w3.org/2000/svg&quot; xmlns:xlink=&quot;http://www.w3.org/1999/xlink&quot;&gt;
&lt;g id=&quot;graph0&quot; class=&quot;graph&quot; transform=&quot;scale(0.71 0.71) rotate(0) translate(4 1068)&quot;&gt;
&lt;title&gt;%3&lt;/title&gt;
&lt;polygon fill=&quot;white&quot; stroke=&quot;transparent&quot; points=&quot;-4,4 -4,-1068 1013,-1068 1013,4 -4,4&quot;/&gt;
&lt;!-- daily_inputs_path --&gt;
&lt;g id=&quot;node1&quot; class=&quot;node&quot;&gt;
&lt;title&gt;daily_inputs_path&lt;/title&gt;
&lt;polygon fill=&quot;#ffffff&quot; stroke=&quot;black&quot; points=&quot;326,-50 120,-50 120,0 332,0 332,-44 326,-50&quot;/&gt;
&lt;polyline fill=&quot;none&quot; stroke=&quot;black&quot; points=&quot;326,-50 326,-44 &quot;/&gt;
&lt;polyline fill=&quot;none&quot; stroke=&quot;black&quot; points=&quot;332,-44 326,-44 &quot;/&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;158&quot; y=&quot;-35.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-weight=&quot;bold&quot; font-size=&quot;14.00&quot;&gt;daily_inputs_path&lt;/text&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;128&quot; y=&quot;-7.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-style=&quot;italic&quot; font-size=&quot;14.00&quot;&gt;/tmp/tmp0nscc6eq/daily.csv&lt;/text&gt;
&lt;/g&gt;
&lt;!-- daily_inputs_vars --&gt;
&lt;g id=&quot;node2&quot; class=&quot;node&quot;&gt;
&lt;title&gt;daily_inputs_vars&lt;/title&gt;
&lt;polygon fill=&quot;#ffffff&quot; stroke=&quot;black&quot; points=&quot;446,-118 0,-118 0,-68 452,-68 452,-112 446,-118&quot;/&gt;
&lt;polyline fill=&quot;none&quot; stroke=&quot;black&quot; points=&quot;446,-118 446,-112 &quot;/&gt;
&lt;polyline fill=&quot;none&quot; stroke=&quot;black&quot; points=&quot;452,-112 446,-112 &quot;/&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;159&quot; y=&quot;-103.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-weight=&quot;bold&quot; font-size=&quot;14.00&quot;&gt;daily_inputs_vars&lt;/text&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;8&quot; y=&quot;-75.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-style=&quot;italic&quot; font-size=&quot;14.00&quot;&gt;[&amp;#39;precipitation_mm&amp;#39;, &amp;#39;sunshine_fraction&amp;#39;, &amp;#39;temperature_celcius&amp;#39;]&lt;/text&gt;
&lt;/g&gt;
&lt;!-- daily_inputs_format --&gt;
&lt;g id=&quot;node3&quot; class=&quot;node&quot;&gt;
&lt;title&gt;daily_inputs_format&lt;/title&gt;
&lt;polygon fill=&quot;#ffffff&quot; stroke=&quot;black&quot; points=&quot;305,-186 141,-186 141,-136 311,-136 311,-180 305,-186&quot;/&gt;
&lt;polyline fill=&quot;none&quot; stroke=&quot;black&quot; points=&quot;305,-186 305,-180 &quot;/&gt;
&lt;polyline fill=&quot;none&quot; stroke=&quot;black&quot; points=&quot;311,-180 305,-180 &quot;/&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;149&quot; y=&quot;-171.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-weight=&quot;bold&quot; font-size=&quot;14.00&quot;&gt;daily_inputs_format&lt;/text&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;214.5&quot; y=&quot;-143.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-style=&quot;italic&quot; font-size=&quot;14.00&quot;&gt;flat&lt;/text&gt;
&lt;/g&gt;
&lt;!-- static_inputs_path --&gt;
&lt;g id=&quot;node4&quot; class=&quot;node&quot;&gt;
&lt;title&gt;static_inputs_path&lt;/title&gt;
&lt;polygon fill=&quot;#ffffff&quot; stroke=&quot;black&quot; points=&quot;331.5,-254 114.5,-254 114.5,-204 337.5,-204 337.5,-248 331.5,-254&quot;/&gt;
&lt;polyline fill=&quot;none&quot; stroke=&quot;black&quot; points=&quot;331.5,-254 331.5,-248 &quot;/&gt;
&lt;polyline fill=&quot;none&quot; stroke=&quot;black&quot; points=&quot;337.5,-248 331.5,-248 &quot;/&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;154.5&quot; y=&quot;-239.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-weight=&quot;bold&quot; font-size=&quot;14.00&quot;&gt;static_inputs_path&lt;/text&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;122.5&quot; y=&quot;-211.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-style=&quot;italic&quot; font-size=&quot;14.00&quot;&gt;/tmp/tmp0nscc6eq/static.json&lt;/text&gt;
&lt;/g&gt;
&lt;!-- static_inputs_vars --&gt;
&lt;g id=&quot;node5&quot; class=&quot;node&quot;&gt;
&lt;title&gt;static_inputs_vars&lt;/title&gt;
&lt;polygon fill=&quot;#ffffff&quot; stroke=&quot;black&quot; points=&quot;375.5,-322 70.5,-322 70.5,-272 381.5,-272 381.5,-316 375.5,-322&quot;/&gt;
&lt;polyline fill=&quot;none&quot; stroke=&quot;black&quot; points=&quot;375.5,-322 375.5,-316 &quot;/&gt;
&lt;polyline fill=&quot;none&quot; stroke=&quot;black&quot; points=&quot;381.5,-316 375.5,-316 &quot;/&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;155.5&quot; y=&quot;-307.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-weight=&quot;bold&quot; font-size=&quot;14.00&quot;&gt;static_inputs_vars&lt;/text&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;78.5&quot; y=&quot;-279.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-style=&quot;italic&quot; font-size=&quot;14.00&quot;&gt;[&amp;#39;elevation&amp;#39;, &amp;#39;latitude&amp;#39;, &amp;#39;max_soil_moisture&amp;#39;]&lt;/text&gt;
&lt;/g&gt;
&lt;!-- static_inputs_format --&gt;
&lt;g id=&quot;node6&quot; class=&quot;node&quot;&gt;
&lt;title&gt;static_inputs_format&lt;/title&gt;
&lt;polygon fill=&quot;#ffffff&quot; stroke=&quot;black&quot; points=&quot;308,-390 138,-390 138,-340 314,-340 314,-384 308,-390&quot;/&gt;
&lt;polyline fill=&quot;none&quot; stroke=&quot;black&quot; points=&quot;308,-390 308,-384 &quot;/&gt;
&lt;polyline fill=&quot;none&quot; stroke=&quot;black&quot; points=&quot;314,-384 308,-384 &quot;/&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;146&quot; y=&quot;-375.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-weight=&quot;bold&quot; font-size=&quot;14.00&quot;&gt;static_inputs_format&lt;/text&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;214.5&quot; y=&quot;-347.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-style=&quot;italic&quot; font-size=&quot;14.00&quot;&gt;flat&lt;/text&gt;
&lt;/g&gt;
&lt;!-- daily_outputs_path --&gt;
&lt;g id=&quot;node7&quot; class=&quot;node&quot;&gt;
&lt;title&gt;daily_outputs_path&lt;/title&gt;
&lt;polygon fill=&quot;#ffffff&quot; stroke=&quot;black&quot; points=&quot;302,-458 144,-458 144,-408 308,-408 308,-452 302,-458&quot;/&gt;
&lt;polyline fill=&quot;none&quot; stroke=&quot;black&quot; points=&quot;302,-458 302,-452 &quot;/&gt;
&lt;polyline fill=&quot;none&quot; stroke=&quot;black&quot; points=&quot;308,-452 302,-452 &quot;/&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;152&quot; y=&quot;-443.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-weight=&quot;bold&quot; font-size=&quot;14.00&quot;&gt;daily_outputs_path&lt;/text&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;170&quot; y=&quot;-415.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-style=&quot;italic&quot; font-size=&quot;14.00&quot;&gt;results/daily.csv&lt;/text&gt;
&lt;/g&gt;
&lt;!-- daily_outputs_vars --&gt;
&lt;g id=&quot;node8&quot; class=&quot;node&quot;&gt;
&lt;title&gt;daily_outputs_vars&lt;/title&gt;
&lt;polygon fill=&quot;#ffffff&quot; stroke=&quot;black&quot; points=&quot;410.5,-526 35.5,-526 35.5,-476 416.5,-476 416.5,-520 410.5,-526&quot;/&gt;
&lt;polyline fill=&quot;none&quot; stroke=&quot;black&quot; points=&quot;410.5,-526 410.5,-520 &quot;/&gt;
&lt;polyline fill=&quot;none&quot; stroke=&quot;black&quot; points=&quot;416.5,-520 410.5,-520 &quot;/&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;153&quot; y=&quot;-511.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-weight=&quot;bold&quot; font-size=&quot;14.00&quot;&gt;daily_outputs_vars&lt;/text&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;43.5&quot; y=&quot;-483.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-style=&quot;italic&quot; font-size=&quot;14.00&quot;&gt;[&amp;#39;actual_evapotranspiration&amp;#39;, &amp;#39;soil_moisture&amp;#39;, &amp;#39;runoff&amp;#39;]&lt;/text&gt;
&lt;/g&gt;
&lt;!-- daily_outputs_format --&gt;
&lt;g id=&quot;node9&quot; class=&quot;node&quot;&gt;
&lt;title&gt;daily_outputs_format&lt;/title&gt;
&lt;polygon fill=&quot;#ffffff&quot; stroke=&quot;black&quot; points=&quot;311,-594 135,-594 135,-544 317,-544 317,-588 311,-594&quot;/&gt;
&lt;polyline fill=&quot;none&quot; stroke=&quot;black&quot; points=&quot;311,-594 311,-588 &quot;/&gt;
&lt;polyline fill=&quot;none&quot; stroke=&quot;black&quot; points=&quot;317,-588 311,-588 &quot;/&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;143&quot; y=&quot;-579.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-weight=&quot;bold&quot; font-size=&quot;14.00&quot;&gt;daily_outputs_format&lt;/text&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;214.5&quot; y=&quot;-551.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-style=&quot;italic&quot; font-size=&quot;14.00&quot;&gt;flat&lt;/text&gt;
&lt;/g&gt;
&lt;!-- hamilton.enable_power_user_mode --&gt;
&lt;g id=&quot;node10&quot; class=&quot;node&quot;&gt;
&lt;title&gt;hamilton.enable_power_user_mode&lt;/title&gt;
&lt;polygon fill=&quot;#ffffff&quot; stroke=&quot;black&quot; points=&quot;365.5,-662 80.5,-662 80.5,-612 371.5,-612 371.5,-656 365.5,-662&quot;/&gt;
&lt;polyline fill=&quot;none&quot; stroke=&quot;black&quot; points=&quot;365.5,-662 365.5,-656 &quot;/&gt;
&lt;polyline fill=&quot;none&quot; stroke=&quot;black&quot; points=&quot;371.5,-656 365.5,-656 &quot;/&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;88.5&quot; y=&quot;-647.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-weight=&quot;bold&quot; font-size=&quot;14.00&quot;&gt;hamilton.enable_power_user_mode&lt;/text&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;210.5&quot; y=&quot;-619.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-style=&quot;italic&quot; font-size=&quot;14.00&quot;&gt;True&lt;/text&gt;
&lt;/g&gt;
&lt;!-- dates_daily --&gt;
&lt;g id=&quot;node11&quot; class=&quot;node&quot;&gt;
&lt;title&gt;dates_daily&lt;/title&gt;
&lt;path fill=&quot;#b4d8e4&quot; stroke=&quot;black&quot; d=&quot;M645,-1064C645,-1064 544,-1064 544,-1064 538,-1064 532,-1058 532,-1052 532,-1052 532,-1012 532,-1012 532,-1006 538,-1000 544,-1000 544,-1000 645,-1000 645,-1000 651,-1000 657,-1006 657,-1012 657,-1012 657,-1052 657,-1052 657,-1058 651,-1064 645,-1064&quot;/&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;550&quot; y=&quot;-1042.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-weight=&quot;bold&quot; font-size=&quot;14.00&quot;&gt;dates_daily&lt;/text&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;543&quot; y=&quot;-1014.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-style=&quot;italic&quot; font-size=&quot;14.00&quot;&gt;DatetimeIndex&lt;/text&gt;
&lt;/g&gt;
&lt;!-- splash --&gt;
&lt;g id=&quot;node21&quot; class=&quot;node&quot;&gt;
&lt;title&gt;splash&lt;/title&gt;
&lt;path fill=&quot;#b4d8e4&quot; stroke=&quot;red&quot; d=&quot;M798,-777C798,-777 749,-777 749,-777 743,-777 737,-771 737,-765 737,-765 737,-725 737,-725 737,-719 743,-713 749,-713 749,-713 798,-713 798,-713 804,-713 810,-719 810,-725 810,-725 810,-765 810,-765 810,-771 804,-777 798,-777&quot;/&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;748&quot; y=&quot;-755.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-weight=&quot;bold&quot; font-size=&quot;14.00&quot;&gt;splash&lt;/text&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;760.5&quot; y=&quot;-727.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-style=&quot;italic&quot; font-size=&quot;14.00&quot;&gt;dict&lt;/text&gt;
&lt;/g&gt;
&lt;!-- dates_daily&amp;#45;&amp;gt;splash --&gt;
&lt;g id=&quot;edge5&quot; class=&quot;edge&quot;&gt;
&lt;title&gt;dates_daily&amp;#45;&amp;gt;splash&lt;/title&gt;
&lt;path fill=&quot;none&quot; stroke=&quot;black&quot; d=&quot;M657.24,-1021.07C675.78,-1015.01 694.7,-1005.6 708,-991 759.57,-934.39 770.87,-841.57 772.79,-787.59&quot;/&gt;
&lt;polygon fill=&quot;black&quot; stroke=&quot;black&quot; points=&quot;776.3,-787.43 773.06,-777.34 769.3,-787.24 776.3,-787.43&quot;/&gt;
&lt;/g&gt;
&lt;!-- elevation --&gt;
&lt;g id=&quot;node12&quot; class=&quot;node&quot;&gt;
&lt;title&gt;elevation&lt;/title&gt;
&lt;path fill=&quot;#b4d8e4&quot; stroke=&quot;black&quot; d=&quot;M630.5,-982C630.5,-982 558.5,-982 558.5,-982 552.5,-982 546.5,-976 546.5,-970 546.5,-970 546.5,-930 546.5,-930 546.5,-924 552.5,-918 558.5,-918 558.5,-918 630.5,-918 630.5,-918 636.5,-918 642.5,-924 642.5,-930 642.5,-930 642.5,-970 642.5,-970 642.5,-976 636.5,-982 630.5,-982&quot;/&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;557.5&quot; y=&quot;-960.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-weight=&quot;bold&quot; font-size=&quot;14.00&quot;&gt;elevation&lt;/text&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;559&quot; y=&quot;-932.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-style=&quot;italic&quot; font-size=&quot;14.00&quot;&gt;DataArray&lt;/text&gt;
&lt;/g&gt;
&lt;!-- elevation&amp;#45;&amp;gt;splash --&gt;
&lt;g id=&quot;edge9&quot; class=&quot;edge&quot;&gt;
&lt;title&gt;elevation&amp;#45;&amp;gt;splash&lt;/title&gt;
&lt;path fill=&quot;none&quot; stroke=&quot;black&quot; d=&quot;M642.74,-942.11C664.95,-936.26 690.36,-926.2 708,-909 741.64,-876.2 758.13,-823.74 765.94,-787.01&quot;/&gt;
&lt;polygon fill=&quot;black&quot; stroke=&quot;black&quot; points=&quot;769.39,-787.59 767.92,-777.09 762.53,-786.21 769.39,-787.59&quot;/&gt;
&lt;/g&gt;
&lt;!-- split_daily_inputs --&gt;
&lt;g id=&quot;node13&quot; class=&quot;node&quot;&gt;
&lt;title&gt;split_daily_inputs&lt;/title&gt;
&lt;path fill=&quot;#b4d8e4&quot; stroke=&quot;black&quot; d=&quot;M292,-744C292,-744 160,-744 160,-744 154,-744 148,-738 148,-732 148,-732 148,-692 148,-692 148,-686 154,-680 160,-680 160,-680 292,-680 292,-680 298,-680 304,-686 304,-692 304,-692 304,-732 304,-732 304,-738 298,-744 292,-744&quot;/&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;159&quot; y=&quot;-722.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-weight=&quot;bold&quot; font-size=&quot;14.00&quot;&gt;split_daily_inputs&lt;/text&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;213&quot; y=&quot;-694.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-style=&quot;italic&quot; font-size=&quot;14.00&quot;&gt;dict&lt;/text&gt;
&lt;/g&gt;
&lt;!-- sunshine_fraction_daily --&gt;
&lt;g id=&quot;node15&quot; class=&quot;node&quot;&gt;
&lt;title&gt;sunshine_fraction_daily&lt;/title&gt;
&lt;path fill=&quot;#b4d8e4&quot; stroke=&quot;black&quot; d=&quot;M685,-818C685,-818 504,-818 504,-818 498,-818 492,-812 492,-806 492,-806 492,-766 492,-766 492,-760 498,-754 504,-754 504,-754 685,-754 685,-754 691,-754 697,-760 697,-766 697,-766 697,-806 697,-806 697,-812 691,-818 685,-818&quot;/&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;503&quot; y=&quot;-796.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-weight=&quot;bold&quot; font-size=&quot;14.00&quot;&gt;sunshine_fraction_daily&lt;/text&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;559&quot; y=&quot;-768.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-style=&quot;italic&quot; font-size=&quot;14.00&quot;&gt;DataArray&lt;/text&gt;
&lt;/g&gt;
&lt;!-- split_daily_inputs&amp;#45;&amp;gt;sunshine_fraction_daily --&gt;
&lt;g id=&quot;edge1&quot; class=&quot;edge&quot;&gt;
&lt;title&gt;split_daily_inputs&amp;#45;&amp;gt;sunshine_fraction_daily&lt;/title&gt;
&lt;path fill=&quot;none&quot; stroke=&quot;black&quot; d=&quot;M304.28,-727.6C355.89,-738.03 424.6,-751.9 482.1,-763.51&quot;/&gt;
&lt;polygon fill=&quot;black&quot; stroke=&quot;black&quot; points=&quot;481.47,-766.95 491.96,-765.5 482.85,-760.09 481.47,-766.95&quot;/&gt;
&lt;/g&gt;
&lt;!-- temperature_celcius_daily --&gt;
&lt;g id=&quot;node17&quot; class=&quot;node&quot;&gt;
&lt;title&gt;temperature_celcius_daily&lt;/title&gt;
&lt;path fill=&quot;#b4d8e4&quot; stroke=&quot;black&quot; d=&quot;M696,-736C696,-736 493,-736 493,-736 487,-736 481,-730 481,-724 481,-724 481,-684 481,-684 481,-678 487,-672 493,-672 493,-672 696,-672 696,-672 702,-672 708,-678 708,-684 708,-684 708,-724 708,-724 708,-730 702,-736 696,-736&quot;/&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;492&quot; y=&quot;-714.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-weight=&quot;bold&quot; font-size=&quot;14.00&quot;&gt;temperature_celcius_daily&lt;/text&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;559&quot; y=&quot;-686.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-style=&quot;italic&quot; font-size=&quot;14.00&quot;&gt;DataArray&lt;/text&gt;
&lt;/g&gt;
&lt;!-- split_daily_inputs&amp;#45;&amp;gt;temperature_celcius_daily --&gt;
&lt;g id=&quot;edge2&quot; class=&quot;edge&quot;&gt;
&lt;title&gt;split_daily_inputs&amp;#45;&amp;gt;temperature_celcius_daily&lt;/title&gt;
&lt;path fill=&quot;none&quot; stroke=&quot;black&quot; d=&quot;M304.28,-710.31C352.52,-709.26 415.69,-707.88 470.68,-706.68&quot;/&gt;
&lt;polygon fill=&quot;black&quot; stroke=&quot;black&quot; points=&quot;470.93,-710.18 480.85,-706.46 470.77,-703.18 470.93,-710.18&quot;/&gt;
&lt;/g&gt;
&lt;!-- precipitation_mm_daily --&gt;
&lt;g id=&quot;node20&quot; class=&quot;node&quot;&gt;
&lt;title&gt;precipitation_mm_daily&lt;/title&gt;
&lt;path fill=&quot;#b4d8e4&quot; stroke=&quot;red&quot; d=&quot;M684,-654C684,-654 505,-654 505,-654 499,-654 493,-648 493,-642 493,-642 493,-602 493,-602 493,-596 499,-590 505,-590 505,-590 684,-590 684,-590 690,-590 696,-596 696,-602 696,-602 696,-642 696,-642 696,-648 690,-654 684,-654&quot;/&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;504&quot; y=&quot;-632.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-weight=&quot;bold&quot; font-size=&quot;14.00&quot;&gt;precipitation_mm_daily&lt;/text&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;559&quot; y=&quot;-604.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-style=&quot;italic&quot; font-size=&quot;14.00&quot;&gt;DataArray&lt;/text&gt;
&lt;/g&gt;
&lt;!-- split_daily_inputs&amp;#45;&amp;gt;precipitation_mm_daily --&gt;
&lt;g id=&quot;edge4&quot; class=&quot;edge&quot;&gt;
&lt;title&gt;split_daily_inputs&amp;#45;&amp;gt;precipitation_mm_daily&lt;/title&gt;
&lt;path fill=&quot;none&quot; stroke=&quot;black&quot; d=&quot;M304.18,-700.93C347.91,-693.92 403.42,-683.76 452,-671 466.95,-667.07 482.6,-662.33 497.8,-657.35&quot;/&gt;
&lt;polygon fill=&quot;black&quot; stroke=&quot;black&quot; points=&quot;499.33,-660.53 507.72,-654.06 497.12,-653.89 499.33,-660.53&quot;/&gt;
&lt;/g&gt;
&lt;!-- latitude --&gt;
&lt;g id=&quot;node14&quot; class=&quot;node&quot;&gt;
&lt;title&gt;latitude&lt;/title&gt;
&lt;path fill=&quot;#b4d8e4&quot; stroke=&quot;black&quot; d=&quot;M629,-900C629,-900 560,-900 560,-900 554,-900 548,-894 548,-888 548,-888 548,-848 548,-848 548,-842 554,-836 560,-836 560,-836 629,-836 629,-836 635,-836 641,-842 641,-848 641,-848 641,-888 641,-888 641,-894 635,-900 629,-900&quot;/&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;563.5&quot; y=&quot;-878.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-weight=&quot;bold&quot; font-size=&quot;14.00&quot;&gt;latitude&lt;/text&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;559&quot; y=&quot;-850.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-style=&quot;italic&quot; font-size=&quot;14.00&quot;&gt;DataArray&lt;/text&gt;
&lt;/g&gt;
&lt;!-- latitude&amp;#45;&amp;gt;splash --&gt;
&lt;g id=&quot;edge10&quot; class=&quot;edge&quot;&gt;
&lt;title&gt;latitude&amp;#45;&amp;gt;splash&lt;/title&gt;
&lt;path fill=&quot;none&quot; stroke=&quot;black&quot; d=&quot;M641.01,-857.65C662.68,-851.22 688.13,-841.39 708,-827 723.33,-815.9 736.89,-800.2 747.57,-785.5&quot;/&gt;
&lt;polygon fill=&quot;black&quot; stroke=&quot;black&quot; points=&quot;750.65,-787.21 753.52,-777.01 744.92,-783.19 750.65,-787.21&quot;/&gt;
&lt;/g&gt;
&lt;!-- sunshine_fraction_daily&amp;#45;&amp;gt;splash --&gt;
&lt;g id=&quot;edge6&quot; class=&quot;edge&quot;&gt;
&lt;title&gt;sunshine_fraction_daily&amp;#45;&amp;gt;splash&lt;/title&gt;
&lt;path fill=&quot;none&quot; stroke=&quot;black&quot; d=&quot;M697.18,-762.45C707.59,-760.04 717.75,-757.68 727.07,-755.52&quot;/&gt;
&lt;polygon fill=&quot;black&quot; stroke=&quot;black&quot; points=&quot;727.87,-758.93 736.82,-753.26 726.29,-752.11 727.87,-758.93&quot;/&gt;
&lt;/g&gt;
&lt;!-- splash_parameters --&gt;
&lt;g id=&quot;node16&quot; class=&quot;node&quot;&gt;
&lt;title&gt;splash_parameters&lt;/title&gt;
&lt;path fill=&quot;#b4d8e4&quot; stroke=&quot;black&quot; d=&quot;M668,-572C668,-572 521,-572 521,-572 515,-572 509,-566 509,-560 509,-560 509,-520 509,-520 509,-514 515,-508 521,-508 521,-508 668,-508 668,-508 674,-508 680,-514 680,-520 680,-520 680,-560 680,-560 680,-566 674,-572 668,-572&quot;/&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;520&quot; y=&quot;-550.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-weight=&quot;bold&quot; font-size=&quot;14.00&quot;&gt;splash_parameters&lt;/text&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;576.5&quot; y=&quot;-522.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-style=&quot;italic&quot; font-size=&quot;14.00&quot;&gt;tuple&lt;/text&gt;
&lt;/g&gt;
&lt;!-- splash_parameters&amp;#45;&amp;gt;splash --&gt;
&lt;g id=&quot;edge12&quot; class=&quot;edge&quot;&gt;
&lt;title&gt;splash_parameters&amp;#45;&amp;gt;splash&lt;/title&gt;
&lt;path fill=&quot;none&quot; stroke=&quot;black&quot; d=&quot;M680.08,-561.66C690.31,-566.81 699.96,-573.16 708,-581 741.64,-613.8 758.13,-666.26 765.94,-702.99&quot;/&gt;
&lt;polygon fill=&quot;black&quot; stroke=&quot;black&quot; points=&quot;762.53,-703.79 767.92,-712.91 769.39,-702.41 762.53,-703.79&quot;/&gt;
&lt;/g&gt;
&lt;!-- temperature_celcius_daily&amp;#45;&amp;gt;splash --&gt;
&lt;g id=&quot;edge7&quot; class=&quot;edge&quot;&gt;
&lt;title&gt;temperature_celcius_daily&amp;#45;&amp;gt;splash&lt;/title&gt;
&lt;path fill=&quot;none&quot; stroke=&quot;black&quot; d=&quot;M708.1,-730.08C714.65,-731.6 721.01,-733.07 727.02,-734.47&quot;/&gt;
&lt;polygon fill=&quot;black&quot; stroke=&quot;black&quot; points=&quot;726.25,-737.88 736.78,-736.73 727.83,-731.06 726.25,-737.88&quot;/&gt;
&lt;/g&gt;
&lt;!-- soil_moisture_daily --&gt;
&lt;g id=&quot;node18&quot; class=&quot;node&quot;&gt;
&lt;title&gt;soil_moisture_daily&lt;/title&gt;
&lt;path fill=&quot;#b4d8e4&quot; stroke=&quot;red&quot; d=&quot;M997,-777C997,-777 851,-777 851,-777 845,-777 839,-771 839,-765 839,-765 839,-725 839,-725 839,-719 845,-713 851,-713 851,-713 997,-713 997,-713 1003,-713 1009,-719 1009,-725 1009,-725 1009,-765 1009,-765 1009,-771 1003,-777 997,-777&quot;/&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;850&quot; y=&quot;-755.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-weight=&quot;bold&quot; font-size=&quot;14.00&quot;&gt;soil_moisture_daily&lt;/text&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;888.5&quot; y=&quot;-727.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-style=&quot;italic&quot; font-size=&quot;14.00&quot;&gt;DataArray&lt;/text&gt;
&lt;/g&gt;
&lt;!-- max_soil_moisture --&gt;
&lt;g id=&quot;node19&quot; class=&quot;node&quot;&gt;
&lt;title&gt;max_soil_moisture&lt;/title&gt;
&lt;path fill=&quot;#b4d8e4&quot; stroke=&quot;black&quot; d=&quot;M665.5,-490C665.5,-490 523.5,-490 523.5,-490 517.5,-490 511.5,-484 511.5,-478 511.5,-478 511.5,-438 511.5,-438 511.5,-432 517.5,-426 523.5,-426 523.5,-426 665.5,-426 665.5,-426 671.5,-426 677.5,-432 677.5,-438 677.5,-438 677.5,-478 677.5,-478 677.5,-484 671.5,-490 665.5,-490&quot;/&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;522.5&quot; y=&quot;-468.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-weight=&quot;bold&quot; font-size=&quot;14.00&quot;&gt;max_soil_moisture&lt;/text&gt;
&lt;text text-anchor=&quot;start&quot; x=&quot;559&quot; y=&quot;-440.8&quot; font-family=&quot;Helvetica,sans-Serif&quot; font-style=&quot;italic&quot; font-size=&quot;14.00&quot;&gt;DataArray&lt;/text&gt;
&lt;/g&gt;
&lt;!-- max_soil_moisture&amp;#45;&amp;gt;splash --&gt;
&lt;g id=&quot;edge11&quot; class=&quot;edge&quot;&gt;
&lt;title&gt;max_soil_moisture&amp;#45;&amp;gt;splash&lt;/title&gt;
&lt;path fill=&quot;none&quot; stroke=&quot;black&quot; d=&quot;M677.59,-477.08C688.96,-482.63 699.6,-489.78 708,-499 759.57,-555.61 770.87,-648.43 772.79,-702.41&quot;/&gt;
&lt;polygon fill=&quot;black&quot; stroke=&quot;black&quot; points=&quot;769.3,-702.76 773.06,-712.66 776.3,-702.57 769.3,-702.76&quot;/&gt;
&lt;/g&gt;
&lt;!-- precipitation_mm_daily&amp;#45;&amp;gt;splash --&gt;
&lt;g id=&quot;edge8&quot; class=&quot;edge&quot;&gt;
&lt;title&gt;precipitation_mm_daily&amp;#45;&amp;gt;splash&lt;/title&gt;
&lt;path fill=&quot;none&quot; stroke=&quot;red&quot; d=&quot;M694.15,-654.14C698.97,-656.88 703.62,-659.83 708,-663 723.33,-674.1 736.89,-689.8 747.57,-704.5&quot;/&gt;
&lt;polygon fill=&quot;red&quot; stroke=&quot;red&quot; points=&quot;744.92,-706.81 753.52,-712.99 750.65,-702.79 744.92,-706.81&quot;/&gt;
&lt;/g&gt;
&lt;!-- splash&amp;#45;&amp;gt;soil_moisture_daily --&gt;
&lt;g id=&quot;edge3&quot; class=&quot;edge&quot;&gt;
&lt;title&gt;splash&amp;#45;&amp;gt;soil_moisture_daily&lt;/title&gt;
&lt;path fill=&quot;none&quot; stroke=&quot;red&quot; d=&quot;M810.07,-745C815.87,-745 822.11,-745 828.56,-745&quot;/&gt;
&lt;polygon fill=&quot;red&quot; stroke=&quot;red&quot; points=&quot;828.86,-748.5 838.86,-745 828.86,-741.5 828.86,-748.5&quot;/&gt;
&lt;/g&gt;
&lt;/g&gt;
&lt;/svg&gt;
" alt="svg+xml">

## Step 4: Run the pipeline

We run the pipeline by calling `dr.execute()` and naming the outputs we want.

By default, the pipeline saves results to files on disk (the `save_*_outputs` nodes).
Here we instead request the merged output dataset directly as an in-memory object
— useful for exploration and plotting without writing any files.

```python {.marimo}
outputs = dr.execute(["merged_daily_outputs"])
outputs
```

<!-- @output:emfo -->

<pre style="white-space: pre; overflow-x: auto;">{&#x27;merged_daily_outputs&#x27;: &#x27;text/html:&lt;div&gt;&lt;svg style=&quot;position: absolute; &#x27;
                         &#x27;width: 0; height: 0; overflow: hidden&quot;&gt;\n&#x27;
                         &#x27;&lt;defs&gt;\n&#x27;
                         &#x27;&lt;symbol id=&quot;icon-database&quot; viewBox=&quot;0 0 32 32&quot;&gt;\n&#x27;
                         &#x27;&lt;path d=&quot;M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 &#x27;
                         &#x27;7.163 5 16 5s16-2.239 &#x27;
                         &#x27;16-5v-4c0-2.761-7.163-5-16-5z&quot;&gt;&lt;/path&gt;\n&#x27;
                         &#x27;&lt;path d=&quot;M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 &#x27;
                         &#x27;7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 &#x27;
                         &#x27;5z&quot;&gt;&lt;/path&gt;\n&#x27;
                         &#x27;&lt;path d=&quot;M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 &#x27;
                         &#x27;7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 &#x27;
                         &#x27;5z&quot;&gt;&lt;/path&gt;\n&#x27;
                         &#x27;&lt;/symbol&gt;\n&#x27;
                         &#x27;&lt;symbol id=&quot;icon-file-text2&quot; viewBox=&quot;0 0 32 32&quot;&gt;\n&#x27;
                         &#x27;&lt;path d=&quot;M28.681 &#x27;
                         &#x27;7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 &#x27;
                         &#x27;0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 &#x27;
                         &#x27;2.5h23c1.378 0 2.5-1.122 &#x27;
                         &#x27;2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 &#x27;
                         &#x27;5.457c0.959 0.959 1.712 1.825 2.268 &#x27;
                         &#x27;2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 &#x27;
                         &#x27;2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 &#x27;
                         &#x27;0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 &#x27;
                         &#x27;0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 &#x27;
                         &#x27;1h7v19.5z&quot;&gt;&lt;/path&gt;\n&#x27;
                         &#x27;&lt;path d=&quot;M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 &#x27;
                         &#x27;1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z&quot;&gt;&lt;/path&gt;\n&#x27;
                         &#x27;&lt;path d=&quot;M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 &#x27;
                         &#x27;1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z&quot;&gt;&lt;/path&gt;\n&#x27;
                         &#x27;&lt;path d=&quot;M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 &#x27;
                         &#x27;1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z&quot;&gt;&lt;/path&gt;\n&#x27;
                         &#x27;&lt;/symbol&gt;\n&#x27;
                         &#x27;&lt;/defs&gt;\n&#x27;
                         &#x27;&lt;/svg&gt;\n&#x27;
                         &#x27;&lt;style&gt;/* CSS stylesheet for displaying xarray &#x27;
                         &#x27;objects in notebooks */\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;:root {\n&#x27;
                         &#x27;  --xr-font-color0: var(\n&#x27;
                         &#x27;    --jp-content-font-color0,\n&#x27;
                         &#x27;    var(--pst-color-text-base rgba(0, 0, 0, 1))\n&#x27;
                         &#x27;  );\n&#x27;
                         &#x27;  --xr-font-color2: var(\n&#x27;
                         &#x27;    --jp-content-font-color2,\n&#x27;
                         &#x27;    var(--pst-color-text-base, rgba(0, 0, 0, 0.54))\n&#x27;
                         &#x27;  );\n&#x27;
                         &#x27;  --xr-font-color3: var(\n&#x27;
                         &#x27;    --jp-content-font-color3,\n&#x27;
                         &#x27;    var(--pst-color-text-base, rgba(0, 0, 0, 0.38))\n&#x27;
                         &#x27;  );\n&#x27;
                         &#x27;  --xr-border-color: var(\n&#x27;
                         &#x27;    --jp-border-color2,\n&#x27;
                         &#x27;    hsl(from var(--pst-color-on-background, white) h &#x27;
                         &#x27;s calc(l - 10))\n&#x27;
                         &#x27;  );\n&#x27;
                         &#x27;  --xr-disabled-color: var(\n&#x27;
                         &#x27;    --jp-layout-color3,\n&#x27;
                         &#x27;    hsl(from var(--pst-color-on-background, white) h &#x27;
                         &#x27;s calc(l - 40))\n&#x27;
                         &#x27;  );\n&#x27;
                         &#x27;  --xr-background-color: var(\n&#x27;
                         &#x27;    --jp-layout-color0,\n&#x27;
                         &#x27;    var(--pst-color-on-background, white)\n&#x27;
                         &#x27;  );\n&#x27;
                         &#x27;  --xr-background-color-row-even: var(\n&#x27;
                         &#x27;    --jp-layout-color1,\n&#x27;
                         &#x27;    hsl(from var(--pst-color-on-background, white) h &#x27;
                         &#x27;s calc(l - 5))\n&#x27;
                         &#x27;  );\n&#x27;
                         &#x27;  --xr-background-color-row-odd: var(\n&#x27;
                         &#x27;    --jp-layout-color2,\n&#x27;
                         &#x27;    hsl(from var(--pst-color-on-background, white) h &#x27;
                         &#x27;s calc(l - 15))\n&#x27;
                         &#x27;  );\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;html&#91;theme=&quot;dark&quot;&#93;,\n&#x27;
                         &#x27;html&#91;data-theme=&quot;dark&quot;&#93;,\n&#x27;
                         &#x27;body&#91;data-theme=&quot;dark&quot;&#93;,\n&#x27;
                         &#x27;body.vscode-dark {\n&#x27;
                         &#x27;  --xr-font-color0: var(\n&#x27;
                         &#x27;    --jp-content-font-color0,\n&#x27;
                         &#x27;    var(--pst-color-text-base, rgba(255, 255, 255, &#x27;
                         &#x27;1))\n&#x27;
                         &#x27;  );\n&#x27;
                         &#x27;  --xr-font-color2: var(\n&#x27;
                         &#x27;    --jp-content-font-color2,\n&#x27;
                         &#x27;    var(--pst-color-text-base, rgba(255, 255, 255, &#x27;
                         &#x27;0.54))\n&#x27;
                         &#x27;  );\n&#x27;
                         &#x27;  --xr-font-color3: var(\n&#x27;
                         &#x27;    --jp-content-font-color3,\n&#x27;
                         &#x27;    var(--pst-color-text-base, rgba(255, 255, 255, &#x27;
                         &#x27;0.38))\n&#x27;
                         &#x27;  );\n&#x27;
                         &#x27;  --xr-border-color: var(\n&#x27;
                         &#x27;    --jp-border-color2,\n&#x27;
                         &#x27;    hsl(from var(--pst-color-on-background, #111111) &#x27;
                         &#x27;h s calc(l + 10))\n&#x27;
                         &#x27;  );\n&#x27;
                         &#x27;  --xr-disabled-color: var(\n&#x27;
                         &#x27;    --jp-layout-color3,\n&#x27;
                         &#x27;    hsl(from var(--pst-color-on-background, #111111) &#x27;
                         &#x27;h s calc(l + 40))\n&#x27;
                         &#x27;  );\n&#x27;
                         &#x27;  --xr-background-color: var(\n&#x27;
                         &#x27;    --jp-layout-color0,\n&#x27;
                         &#x27;    var(--pst-color-on-background, #111111)\n&#x27;
                         &#x27;  );\n&#x27;
                         &#x27;  --xr-background-color-row-even: var(\n&#x27;
                         &#x27;    --jp-layout-color1,\n&#x27;
                         &#x27;    hsl(from var(--pst-color-on-background, #111111) &#x27;
                         &#x27;h s calc(l + 5))\n&#x27;
                         &#x27;  );\n&#x27;
                         &#x27;  --xr-background-color-row-odd: var(\n&#x27;
                         &#x27;    --jp-layout-color2,\n&#x27;
                         &#x27;    hsl(from var(--pst-color-on-background, #111111) &#x27;
                         &#x27;h s calc(l + 15))\n&#x27;
                         &#x27;  );\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-wrap {\n&#x27;
                         &#x27;  display: block !important;\n&#x27;
                         &#x27;  min-width: 300px;\n&#x27;
                         &#x27;  max-width: 700px;\n&#x27;
                         &#x27;  line-height: 1.6;\n&#x27;
                         &#x27;  padding-bottom: 4px;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-text-repr-fallback {\n&#x27;
                         &#x27;  /* fallback to plain text repr when CSS is not &#x27;
                         &#x27;injected (untrusted notebook) */\n&#x27;
                         &#x27;  display: none;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-header {\n&#x27;
                         &#x27;  padding-top: 6px;\n&#x27;
                         &#x27;  padding-bottom: 6px;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-header {\n&#x27;
                         &#x27;  border-bottom: solid 1px var(--xr-border-color);\n&#x27;
                         &#x27;  margin-bottom: 4px;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-header &gt; div,\n&#x27;
                         &#x27;.xr-header &gt; ul {\n&#x27;
                         &#x27;  display: inline;\n&#x27;
                         &#x27;  margin-top: 0;\n&#x27;
                         &#x27;  margin-bottom: 0;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-obj-type,\n&#x27;
                         &#x27;.xr-obj-name {\n&#x27;
                         &#x27;  margin-left: 2px;\n&#x27;
                         &#x27;  margin-right: 10px;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-obj-type,\n&#x27;
                         &#x27;.xr-group-box-contents &gt; label {\n&#x27;
                         &#x27;  color: var(--xr-font-color2);\n&#x27;
                         &#x27;  display: block;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-sections {\n&#x27;
                         &#x27;  padding-left: 0 !important;\n&#x27;
                         &#x27;  display: grid;\n&#x27;
                         &#x27;  grid-template-columns: 150px auto auto 1fr 0 20px &#x27;
                         &#x27;0 20px;\n&#x27;
                         &#x27;  margin-block-start: 0;\n&#x27;
                         &#x27;  margin-block-end: 0;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-section-item {\n&#x27;
                         &#x27;  display: contents;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-section-item &gt; input,\n&#x27;
                         &#x27;.xr-group-box-contents &gt; input,\n&#x27;
                         &#x27;.xr-array-wrap &gt; input {\n&#x27;
                         &#x27;  display: block;\n&#x27;
                         &#x27;  opacity: 0;\n&#x27;
                         &#x27;  height: 0;\n&#x27;
                         &#x27;  margin: 0;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-section-item &gt; input + label,\n&#x27;
                         &#x27;.xr-var-item &gt; input + label {\n&#x27;
                         &#x27;  color: var(--xr-disabled-color);\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-section-item &gt; input:enabled + label,\n&#x27;
                         &#x27;.xr-var-item &gt; input:enabled + label,\n&#x27;
                         &#x27;.xr-array-wrap &gt; input:enabled + label,\n&#x27;
                         &#x27;.xr-group-box-contents &gt; input:enabled + label {\n&#x27;
                         &#x27;  cursor: pointer;\n&#x27;
                         &#x27;  color: var(--xr-font-color2);\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-section-item &gt; input:focus-visible + label,\n&#x27;
                         &#x27;.xr-var-item &gt; input:focus-visible + label,\n&#x27;
                         &#x27;.xr-array-wrap &gt; input:focus-visible + label,\n&#x27;
                         &#x27;.xr-group-box-contents &gt; input:focus-visible + label &#x27;
                         &#x27;{\n&#x27;
                         &#x27;  outline: auto;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-section-item &gt; input:enabled + label:hover,\n&#x27;
                         &#x27;.xr-var-item &gt; input:enabled + label:hover,\n&#x27;
                         &#x27;.xr-array-wrap &gt; input:enabled + label:hover,\n&#x27;
                         &#x27;.xr-group-box-contents &gt; input:enabled + label:hover &#x27;
                         &#x27;{\n&#x27;
                         &#x27;  color: var(--xr-font-color0);\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-section-summary {\n&#x27;
                         &#x27;  grid-column: 1;\n&#x27;
                         &#x27;  color: var(--xr-font-color2);\n&#x27;
                         &#x27;  font-weight: 500;\n&#x27;
                         &#x27;  white-space: nowrap;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-section-summary &gt; em {\n&#x27;
                         &#x27;  font-weight: normal;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-span-grid {\n&#x27;
                         &#x27;  grid-column-end: -1;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-section-summary &gt; span {\n&#x27;
                         &#x27;  display: inline-block;\n&#x27;
                         &#x27;  padding-left: 0.3em;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-group-box-contents &gt; input:checked + label &gt; &#x27;
                         &#x27;span {\n&#x27;
                         &#x27;  display: inline-block;\n&#x27;
                         &#x27;  padding-left: 0.6em;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-section-summary-in:disabled + label {\n&#x27;
                         &#x27;  color: var(--xr-font-color2);\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-section-summary-in + label:before {\n&#x27;
                         &#x27;  display: inline-block;\n&#x27;
                         &#x27;  content: &quot;►&quot;;\n&#x27;
                         &#x27;  font-size: 11px;\n&#x27;
                         &#x27;  width: 15px;\n&#x27;
                         &#x27;  text-align: center;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-section-summary-in:disabled + label:before {\n&#x27;
                         &#x27;  color: var(--xr-disabled-color);\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-section-summary-in:checked + label:before {\n&#x27;
                         &#x27;  content: &quot;▼&quot;;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-section-summary-in:checked + label &gt; span {\n&#x27;
                         &#x27;  display: none;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-section-summary,\n&#x27;
                         &#x27;.xr-section-inline-details,\n&#x27;
                         &#x27;.xr-group-box-contents &gt; label {\n&#x27;
                         &#x27;  padding-top: 4px;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-section-inline-details {\n&#x27;
                         &#x27;  grid-column: 2 / -1;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-section-details {\n&#x27;
                         &#x27;  grid-column: 1 / -1;\n&#x27;
                         &#x27;  margin-top: 4px;\n&#x27;
                         &#x27;  margin-bottom: 5px;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-section-summary-in ~ .xr-section-details {\n&#x27;
                         &#x27;  display: none;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-section-summary-in:checked ~ .xr-section-details &#x27;
                         &#x27;{\n&#x27;
                         &#x27;  display: contents;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-children {\n&#x27;
                         &#x27;  display: inline-grid;\n&#x27;
                         &#x27;  grid-template-columns: 100%;\n&#x27;
                         &#x27;  grid-column: 1 / -1;\n&#x27;
                         &#x27;  padding-top: 4px;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-group-box {\n&#x27;
                         &#x27;  display: inline-grid;\n&#x27;
                         &#x27;  grid-template-columns: 0px 30px auto;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-group-box-vline {\n&#x27;
                         &#x27;  grid-column-start: 1;\n&#x27;
                         &#x27;  border-right: 0.2em solid;\n&#x27;
                         &#x27;  border-color: var(--xr-border-color);\n&#x27;
                         &#x27;  width: 0px;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-group-box-hline {\n&#x27;
                         &#x27;  grid-column-start: 2;\n&#x27;
                         &#x27;  grid-row-start: 1;\n&#x27;
                         &#x27;  height: 1em;\n&#x27;
                         &#x27;  width: 26px;\n&#x27;
                         &#x27;  border-bottom: 0.2em solid;\n&#x27;
                         &#x27;  border-color: var(--xr-border-color);\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-group-box-contents {\n&#x27;
                         &#x27;  grid-column-start: 3;\n&#x27;
                         &#x27;  padding-bottom: 4px;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-group-box-contents &gt; label::before {\n&#x27;
                         &#x27;  content: &quot;📂&quot;;\n&#x27;
                         &#x27;  padding-right: 0.3em;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-group-box-contents &gt; input:checked + &#x27;
                         &#x27;label::before {\n&#x27;
                         &#x27;  content: &quot;📁&quot;;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-group-box-contents &gt; input:checked + label {\n&#x27;
                         &#x27;  padding-bottom: 0px;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-group-box-contents &gt; input:checked ~ &#x27;
                         &#x27;.xr-sections {\n&#x27;
                         &#x27;  display: none;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-group-box-contents &gt; input + label &gt; span {\n&#x27;
                         &#x27;  display: none;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-group-box-ellipsis {\n&#x27;
                         &#x27;  font-size: 1.4em;\n&#x27;
                         &#x27;  font-weight: 900;\n&#x27;
                         &#x27;  color: var(--xr-font-color2);\n&#x27;
                         &#x27;  letter-spacing: 0.15em;\n&#x27;
                         &#x27;  cursor: default;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-array-wrap {\n&#x27;
                         &#x27;  grid-column: 1 / -1;\n&#x27;
                         &#x27;  display: grid;\n&#x27;
                         &#x27;  grid-template-columns: 20px auto;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-array-wrap &gt; label {\n&#x27;
                         &#x27;  grid-column: 1;\n&#x27;
                         &#x27;  vertical-align: top;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-preview {\n&#x27;
                         &#x27;  color: var(--xr-font-color3);\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-array-preview,\n&#x27;
                         &#x27;.xr-array-data {\n&#x27;
                         &#x27;  padding: 0 5px !important;\n&#x27;
                         &#x27;  grid-column: 2;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-array-data,\n&#x27;
                         &#x27;.xr-array-in:checked ~ .xr-array-preview {\n&#x27;
                         &#x27;  display: none;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-array-in:checked ~ .xr-array-data,\n&#x27;
                         &#x27;.xr-array-preview {\n&#x27;
                         &#x27;  display: inline-block;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-dim-list {\n&#x27;
                         &#x27;  display: inline-block !important;\n&#x27;
                         &#x27;  list-style: none;\n&#x27;
                         &#x27;  padding: 0 !important;\n&#x27;
                         &#x27;  margin: 0;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-dim-list li {\n&#x27;
                         &#x27;  display: inline-block;\n&#x27;
                         &#x27;  padding: 0;\n&#x27;
                         &#x27;  margin: 0;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-dim-list:before {\n&#x27;
                         &#x27;  content: &quot;(&quot;;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-dim-list:after {\n&#x27;
                         &#x27;  content: &quot;)&quot;;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-dim-list li:not(:last-child):after {\n&#x27;
                         &#x27;  content: &quot;,&quot;;\n&#x27;
                         &#x27;  padding-right: 5px;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-has-index {\n&#x27;
                         &#x27;  font-weight: bold;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-var-list,\n&#x27;
                         &#x27;.xr-var-item {\n&#x27;
                         &#x27;  display: contents;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-var-item &gt; div,\n&#x27;
                         &#x27;.xr-var-item label,\n&#x27;
                         &#x27;.xr-var-item &gt; .xr-var-name span {\n&#x27;
                         &#x27;  background-color: &#x27;
                         &#x27;var(--xr-background-color-row-even);\n&#x27;
                         &#x27;  border-color: var(--xr-background-color-row-odd);\n&#x27;
                         &#x27;  margin-bottom: 0;\n&#x27;
                         &#x27;  padding-top: 2px;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-var-item &gt; .xr-var-name:hover span {\n&#x27;
                         &#x27;  padding-right: 5px;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-var-list &gt; li:nth-child(odd) &gt; div,\n&#x27;
                         &#x27;.xr-var-list &gt; li:nth-child(odd) &gt; label,\n&#x27;
                         &#x27;.xr-var-list &gt; li:nth-child(odd) &gt; .xr-var-name span &#x27;
                         &#x27;{\n&#x27;
                         &#x27;  background-color: &#x27;
                         &#x27;var(--xr-background-color-row-odd);\n&#x27;
                         &#x27;  border-color: &#x27;
                         &#x27;var(--xr-background-color-row-even);\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-var-name {\n&#x27;
                         &#x27;  grid-column: 1;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-var-dims {\n&#x27;
                         &#x27;  grid-column: 2;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-var-dtype {\n&#x27;
                         &#x27;  grid-column: 3;\n&#x27;
                         &#x27;  text-align: right;\n&#x27;
                         &#x27;  color: var(--xr-font-color2);\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-var-preview {\n&#x27;
                         &#x27;  grid-column: 4;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-index-preview {\n&#x27;
                         &#x27;  grid-column: 2 / 5;\n&#x27;
                         &#x27;  color: var(--xr-font-color2);\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-var-name,\n&#x27;
                         &#x27;.xr-var-dims,\n&#x27;
                         &#x27;.xr-var-dtype,\n&#x27;
                         &#x27;.xr-preview,\n&#x27;
                         &#x27;.xr-attrs dt {\n&#x27;
                         &#x27;  white-space: nowrap;\n&#x27;
                         &#x27;  overflow: hidden;\n&#x27;
                         &#x27;  text-overflow: ellipsis;\n&#x27;
                         &#x27;  padding-right: 10px;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-var-name:hover,\n&#x27;
                         &#x27;.xr-var-dims:hover,\n&#x27;
                         &#x27;.xr-var-dtype:hover,\n&#x27;
                         &#x27;.xr-attrs dt:hover {\n&#x27;
                         &#x27;  overflow: visible;\n&#x27;
                         &#x27;  width: auto;\n&#x27;
                         &#x27;  z-index: 1;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-var-attrs,\n&#x27;
                         &#x27;.xr-var-data,\n&#x27;
                         &#x27;.xr-index-data {\n&#x27;
                         &#x27;  display: none;\n&#x27;
                         &#x27;  border-top: 2px dotted &#x27;
                         &#x27;var(--xr-background-color);\n&#x27;
                         &#x27;  padding-bottom: 20px !important;\n&#x27;
                         &#x27;  padding-top: 10px !important;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-var-attrs-in + label,\n&#x27;
                         &#x27;.xr-var-data-in + label,\n&#x27;
                         &#x27;.xr-index-data-in + label {\n&#x27;
                         &#x27;  padding: 0 1px;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-var-attrs-in:checked ~ .xr-var-attrs,\n&#x27;
                         &#x27;.xr-var-data-in:checked ~ .xr-var-data,\n&#x27;
                         &#x27;.xr-index-data-in:checked ~ .xr-index-data {\n&#x27;
                         &#x27;  display: block;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-var-data &gt; table {\n&#x27;
                         &#x27;  float: right;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-var-data &gt; pre,\n&#x27;
                         &#x27;.xr-index-data &gt; pre,\n&#x27;
                         &#x27;.xr-var-data &gt; table &gt; tbody &gt; tr {\n&#x27;
                         &#x27;  background-color: transparent !important;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-var-name span,\n&#x27;
                         &#x27;.xr-var-data,\n&#x27;
                         &#x27;.xr-index-name div,\n&#x27;
                         &#x27;.xr-index-data,\n&#x27;
                         &#x27;.xr-attrs {\n&#x27;
                         &#x27;  padding-left: 25px !important;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-attrs,\n&#x27;
                         &#x27;.xr-var-attrs,\n&#x27;
                         &#x27;.xr-var-data,\n&#x27;
                         &#x27;.xr-index-data {\n&#x27;
                         &#x27;  grid-column: 1 / -1;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;dl.xr-attrs {\n&#x27;
                         &#x27;  padding: 0;\n&#x27;
                         &#x27;  margin: 0;\n&#x27;
                         &#x27;  display: grid;\n&#x27;
                         &#x27;  grid-template-columns: 125px auto;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-attrs dt,\n&#x27;
                         &#x27;.xr-attrs dd {\n&#x27;
                         &#x27;  padding: 0;\n&#x27;
                         &#x27;  margin: 0;\n&#x27;
                         &#x27;  float: left;\n&#x27;
                         &#x27;  padding-right: 10px;\n&#x27;
                         &#x27;  width: auto;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-attrs dt {\n&#x27;
                         &#x27;  font-weight: normal;\n&#x27;
                         &#x27;  grid-column: 1;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-attrs dt:hover span {\n&#x27;
                         &#x27;  display: inline-block;\n&#x27;
                         &#x27;  background: var(--xr-background-color);\n&#x27;
                         &#x27;  padding-right: 10px;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-attrs dd {\n&#x27;
                         &#x27;  grid-column: 2;\n&#x27;
                         &#x27;  white-space: pre-wrap;\n&#x27;
                         &#x27;  word-break: break-all;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-icon-database,\n&#x27;
                         &#x27;.xr-icon-file-text2,\n&#x27;
                         &#x27;.xr-no-icon {\n&#x27;
                         &#x27;  display: inline-block;\n&#x27;
                         &#x27;  vertical-align: middle;\n&#x27;
                         &#x27;  width: 1em;\n&#x27;
                         &#x27;  height: 1.5em !important;\n&#x27;
                         &#x27;  stroke-width: 0;\n&#x27;
                         &#x27;  stroke: currentColor;\n&#x27;
                         &#x27;  fill: currentColor;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;\n&#x27;
                         &#x27;.xr-var-attrs-in:checked + label &gt; &#x27;
                         &#x27;.xr-icon-file-text2,\n&#x27;
                         &#x27;.xr-var-data-in:checked + label &gt; &#x27;
                         &#x27;.xr-icon-database,\n&#x27;
                         &#x27;.xr-index-data-in:checked + label &gt; &#x27;
                         &#x27;.xr-icon-database {\n&#x27;
                         &#x27;  color: var(--xr-font-color0);\n&#x27;
                         &#x27;  filter: drop-shadow(1px 1px 5px &#x27;
                         &#x27;var(--xr-font-color2));\n&#x27;
                         &#x27;  stroke-width: 0.8px;\n&#x27;
                         &#x27;}\n&#x27;
                         &#x27;&lt;/style&gt;&lt;pre &#x27;
                         &quot;class=&#x27;xr-text-repr-fallback&#x27;&gt;&amp;lt;xarray.Dataset&amp;gt; &quot;
                         &#x27;Size: 23kB\n&#x27;
                         &#x27;Dimensions:                          (time: 730, &#x27;
                         &#x27;pixel: 1)\n&#x27;
                         &#x27;Coordinates:\n&#x27;
                         &#x27;  * time                             (time) &#x27;
                         &#x27;datetime64&#91;us&#93; 6kB 2020-01-01 ......\n&#x27;
                         &#x27;  * pixel                            (pixel) int64 &#x27;
                         &#x27;8B 0\n&#x27;
                         &#x27;Data variables:\n&#x27;
                         &#x27;    actual_evapotranspiration_daily  (time, pixel) &#x27;
                         &#x27;float64 6kB 0.314 ... 0.2944\n&#x27;
                         &#x27;    soil_moisture_daily              (time, pixel) &#x27;
                         &#x27;float64 6kB 212.4 ... 238.0\n&#x27;
                         &#x27;    runoff_daily                     (time, pixel) &#x27;
                         &#x27;float64 6kB 0.0 0.0 ... 0.0&lt;/pre&gt;&lt;div &#x27;
                         &quot;class=&#x27;xr-wrap&#x27; style=&#x27;display:none&#x27;&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-header&#x27;&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-obj-type&#x27;&gt;xarray.Dataset&lt;/div&gt;&lt;/div&gt;&lt;ul &quot;
                         &quot;class=&#x27;xr-sections&#x27;&gt;&lt;li &quot;
                         &quot;class=&#x27;xr-section-item&#x27;&gt;&lt;input &quot;
                         &quot;id=&#x27;section-6b31b6f4-40a5-43cd-b1c4-2732c3cb30fd&#x27; &quot;
                         &quot;class=&#x27;xr-section-summary-in&#x27; type=&#x27;checkbox&#x27; &quot;
                         &#x27;disabled /&gt;&lt;label &#x27;
                         &quot;for=&#x27;section-6b31b6f4-40a5-43cd-b1c4-2732c3cb30fd&#x27; &quot;
                         &quot;class=&#x27;xr-section-summary&#x27;&gt;Dimensions:&lt;/label&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-section-inline-details&#x27;&gt;&lt;ul &quot;
                         &quot;class=&#x27;xr-dim-list&#x27;&gt;&lt;li&gt;&lt;span &quot;
                         &quot;class=&#x27;xr-has-index&#x27;&gt;time&lt;/span&gt;: 730&lt;/li&gt;&lt;li&gt;&lt;span &quot;
                         &quot;class=&#x27;xr-has-index&#x27;&gt;pixel&lt;/span&gt;: &quot;
                         &#x27;1&lt;/li&gt;&lt;/ul&gt;&lt;/div&gt;&lt;/li&gt;&lt;li &#x27;
                         &quot;class=&#x27;xr-section-item&#x27;&gt;&lt;input &quot;
                         &quot;id=&#x27;section-5be7c86f-9776-4dc5-89c8-17c06d6a0ee4&#x27; &quot;
                         &quot;class=&#x27;xr-section-summary-in&#x27; type=&#x27;checkbox&#x27; &quot;
                         &#x27;checked /&gt;&lt;label &#x27;
                         &quot;for=&#x27;section-5be7c86f-9776-4dc5-89c8-17c06d6a0ee4&#x27; &quot;
                         &quot;class=&#x27;xr-section-summary&#x27; title=&#x27;Expand/collapse &quot;
                         &quot;section&#x27;&gt;Coordinates: &lt;span&gt;(2)&lt;/span&gt;&lt;/label&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-section-inline-details&#x27;&gt;&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-section-details&#x27;&gt;&lt;ul &quot;
                         &quot;class=&#x27;xr-var-list&#x27;&gt;&lt;li class=&#x27;xr-var-item&#x27;&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-name&#x27;&gt;&lt;span &quot;
                         &quot;class=&#x27;xr-has-index&#x27;&gt;time&lt;/span&gt;&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-dims&#x27;&gt;(time)&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-dtype&#x27;&gt;datetime64&#91;us&#93;&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-preview xr-preview&#x27;&gt;2020-01-01 ... &quot;
                         &#x27;2021-12-30&lt;/div&gt;&lt;input &#x27;
                         &quot;id=&#x27;attrs-83b14901-34f6-436e-8316-71ffb0b7b6f3&#x27; &quot;
                         &quot;class=&#x27;xr-var-attrs-in&#x27; type=&#x27;checkbox&#x27; &quot;
                         &#x27;disabled&gt;&lt;label &#x27;
                         &quot;for=&#x27;attrs-83b14901-34f6-436e-8316-71ffb0b7b6f3&#x27; &quot;
                         &quot;title=&#x27;Show/Hide attributes&#x27;&gt;&lt;svg class=&#x27;icon &quot;
                         &quot;xr-icon-file-text2&#x27;&gt;&lt;use &quot;
                         &quot;xlink:href=&#x27;#icon-file-text2&#x27;&gt;&lt;/use&gt;&lt;/svg&gt;&lt;/label&gt;&lt;input &quot;
                         &quot;id=&#x27;data-87ee194d-bf08-4758-8242-af6d6e456b7d&#x27; &quot;
                         &quot;class=&#x27;xr-var-data-in&#x27; type=&#x27;checkbox&#x27;&gt;&lt;label &quot;
                         &quot;for=&#x27;data-87ee194d-bf08-4758-8242-af6d6e456b7d&#x27; &quot;
                         &quot;title=&#x27;Show/Hide data repr&#x27;&gt;&lt;svg class=&#x27;icon &quot;
                         &quot;xr-icon-database&#x27;&gt;&lt;use &quot;
                         &quot;xlink:href=&#x27;#icon-database&#x27;&gt;&lt;/use&gt;&lt;/svg&gt;&lt;/label&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-attrs&#x27;&gt;&lt;dl &quot;
                         &quot;class=&#x27;xr-attrs&#x27;&gt;&lt;/dl&gt;&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-data&#x27;&gt;&lt;pre&gt;array(&#91;&amp;#x27;2020-01-01T00:00:00.000000&amp;#x27;, &quot;
                         &#x27;&amp;#x27;2020-01-02T00:00:00.000000&amp;#x27;,\n&#x27;
                         &#x27;       &amp;#x27;2020-01-03T00:00:00.000000&amp;#x27;, ..., &#x27;
                         &#x27;&amp;#x27;2021-12-28T00:00:00.000000&amp;#x27;,\n&#x27;
                         &#x27;       &amp;#x27;2021-12-29T00:00:00.000000&amp;#x27;, &#x27;
                         &#x27;&amp;#x27;2021-12-30T00:00:00.000000&amp;#x27;&#93;,\n&#x27;
                         &#x27;      shape=(730,), &#x27;
                         &#x27;dtype=&amp;#x27;datetime64&#91;us&#93;&amp;#x27;)&lt;/pre&gt;&lt;/div&gt;&lt;/li&gt;&lt;li &#x27;
                         &quot;class=&#x27;xr-var-item&#x27;&gt;&lt;div class=&#x27;xr-var-name&#x27;&gt;&lt;span &quot;
                         &quot;class=&#x27;xr-has-index&#x27;&gt;pixel&lt;/span&gt;&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-dims&#x27;&gt;(pixel)&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-dtype&#x27;&gt;int64&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-preview xr-preview&#x27;&gt;0&lt;/div&gt;&lt;input &quot;
                         &quot;id=&#x27;attrs-8b7561d9-c4f1-4837-aa94-4542844409d2&#x27; &quot;
                         &quot;class=&#x27;xr-var-attrs-in&#x27; type=&#x27;checkbox&#x27; &quot;
                         &#x27;disabled&gt;&lt;label &#x27;
                         &quot;for=&#x27;attrs-8b7561d9-c4f1-4837-aa94-4542844409d2&#x27; &quot;
                         &quot;title=&#x27;Show/Hide attributes&#x27;&gt;&lt;svg class=&#x27;icon &quot;
                         &quot;xr-icon-file-text2&#x27;&gt;&lt;use &quot;
                         &quot;xlink:href=&#x27;#icon-file-text2&#x27;&gt;&lt;/use&gt;&lt;/svg&gt;&lt;/label&gt;&lt;input &quot;
                         &quot;id=&#x27;data-875d3dd1-9402-4bf0-b998-a4eb3da00b6f&#x27; &quot;
                         &quot;class=&#x27;xr-var-data-in&#x27; type=&#x27;checkbox&#x27;&gt;&lt;label &quot;
                         &quot;for=&#x27;data-875d3dd1-9402-4bf0-b998-a4eb3da00b6f&#x27; &quot;
                         &quot;title=&#x27;Show/Hide data repr&#x27;&gt;&lt;svg class=&#x27;icon &quot;
                         &quot;xr-icon-database&#x27;&gt;&lt;use &quot;
                         &quot;xlink:href=&#x27;#icon-database&#x27;&gt;&lt;/use&gt;&lt;/svg&gt;&lt;/label&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-attrs&#x27;&gt;&lt;dl &quot;
                         &quot;class=&#x27;xr-attrs&#x27;&gt;&lt;/dl&gt;&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-data&#x27;&gt;&lt;pre&gt;array(&#91;0&#93;)&lt;/pre&gt;&lt;/div&gt;&lt;/li&gt;&lt;/ul&gt;&lt;/div&gt;&lt;/li&gt;&lt;li &quot;
                         &quot;class=&#x27;xr-section-item&#x27;&gt;&lt;input &quot;
                         &quot;id=&#x27;section-c7c916dd-7a86-43f7-9650-2129669e7a1c&#x27; &quot;
                         &quot;class=&#x27;xr-section-summary-in&#x27; type=&#x27;checkbox&#x27; &quot;
                         &#x27;checked /&gt;&lt;label &#x27;
                         &quot;for=&#x27;section-c7c916dd-7a86-43f7-9650-2129669e7a1c&#x27; &quot;
                         &quot;class=&#x27;xr-section-summary&#x27; title=&#x27;Expand/collapse &quot;
                         &quot;section&#x27;&gt;Data variables: &quot;
                         &#x27;&lt;span&gt;(3)&lt;/span&gt;&lt;/label&gt;&lt;div &#x27;
                         &quot;class=&#x27;xr-section-inline-details&#x27;&gt;&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-section-details&#x27;&gt;&lt;ul &quot;
                         &quot;class=&#x27;xr-var-list&#x27;&gt;&lt;li class=&#x27;xr-var-item&#x27;&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-name&#x27;&gt;&lt;span&gt;actual_evapotranspiration_daily&lt;/span&gt;&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-dims&#x27;&gt;(time, pixel)&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-dtype&#x27;&gt;float64&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-preview xr-preview&#x27;&gt;0.314 0.2893 ... &quot;
                         &#x27;0.3058 0.2944&lt;/div&gt;&lt;input &#x27;
                         &quot;id=&#x27;attrs-c5ad8196-7d56-4ef3-bb1f-3dbb4c91d7e5&#x27; &quot;
                         &quot;class=&#x27;xr-var-attrs-in&#x27; type=&#x27;checkbox&#x27; &quot;
                         &#x27;disabled&gt;&lt;label &#x27;
                         &quot;for=&#x27;attrs-c5ad8196-7d56-4ef3-bb1f-3dbb4c91d7e5&#x27; &quot;
                         &quot;title=&#x27;Show/Hide attributes&#x27;&gt;&lt;svg class=&#x27;icon &quot;
                         &quot;xr-icon-file-text2&#x27;&gt;&lt;use &quot;
                         &quot;xlink:href=&#x27;#icon-file-text2&#x27;&gt;&lt;/use&gt;&lt;/svg&gt;&lt;/label&gt;&lt;input &quot;
                         &quot;id=&#x27;data-0e10f96d-1838-45c4-9a67-125742bbb97d&#x27; &quot;
                         &quot;class=&#x27;xr-var-data-in&#x27; type=&#x27;checkbox&#x27;&gt;&lt;label &quot;
                         &quot;for=&#x27;data-0e10f96d-1838-45c4-9a67-125742bbb97d&#x27; &quot;
                         &quot;title=&#x27;Show/Hide data repr&#x27;&gt;&lt;svg class=&#x27;icon &quot;
                         &quot;xr-icon-database&#x27;&gt;&lt;use &quot;
                         &quot;xlink:href=&#x27;#icon-database&#x27;&gt;&lt;/use&gt;&lt;/svg&gt;&lt;/label&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-attrs&#x27;&gt;&lt;dl &quot;
                         &quot;class=&#x27;xr-attrs&#x27;&gt;&lt;/dl&gt;&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-data&#x27;&gt;&lt;pre&gt;array(&#91;&#91;0.31395395&#93;,\n&quot;
                         &#x27;       &#91;0.28929518&#93;,\n&#x27;
                         &#x27;       &#91;0.32094665&#93;,\n&#x27;
                         &#x27;       &#91;0.31443491&#93;,\n&#x27;
                         &#x27;       &#91;0.34239275&#93;,\n&#x27;
                         &#x27;       &#91;0.34550369&#93;,\n&#x27;
                         &#x27;       &#91;0.38786462&#93;,\n&#x27;
                         &#x27;       &#91;0.41999003&#93;,\n&#x27;
                         &#x27;       &#91;0.42717711&#93;,\n&#x27;
                         &#x27;       &#91;0.44440676&#93;,\n&#x27;
                         &#x27;       &#91;0.35939374&#93;,\n&#x27;
                         &#x27;       &#91;0.32592932&#93;,\n&#x27;
                         &#x27;       &#91;0.34000767&#93;,\n&#x27;
                         &#x27;       &#91;0.31792515&#93;,\n&#x27;
                         &#x27;       &#91;0.31686562&#93;,\n&#x27;
                         &#x27;       &#91;0.31829467&#93;,\n&#x27;
                         &#x27;       &#91;0.33265723&#93;,\n&#x27;
                         &#x27;       &#91;0.35701819&#93;,\n&#x27;
                         &#x27;       &#91;0.35056228&#93;,\n&#x27;
                         &#x27;       &#91;0.31752367&#93;,\n&#x27;
                         &#x27;...\n&#x27;
                         &#x27;       &#91;0.32583644&#93;,\n&#x27;
                         &#x27;       &#91;0.32800456&#93;,\n&#x27;
                         &#x27;       &#91;0.29824272&#93;,\n&#x27;
                         &#x27;       &#91;0.27012712&#93;,\n&#x27;
                         &#x27;       &#91;0.27779491&#93;,\n&#x27;
                         &#x27;       &#91;0.28297273&#93;,\n&#x27;
                         &#x27;       &#91;0.27164025&#93;,\n&#x27;
                         &#x27;       &#91;0.2869309 &#93;,\n&#x27;
                         &#x27;       &#91;0.29738512&#93;,\n&#x27;
                         &#x27;       &#91;0.28568484&#93;,\n&#x27;
                         &#x27;       &#91;0.29145102&#93;,\n&#x27;
                         &#x27;       &#91;0.27823169&#93;,\n&#x27;
                         &#x27;       &#91;0.25053747&#93;,\n&#x27;
                         &#x27;       &#91;0.31472124&#93;,\n&#x27;
                         &#x27;       &#91;0.31199284&#93;,\n&#x27;
                         &#x27;       &#91;0.26308311&#93;,\n&#x27;
                         &#x27;       &#91;0.28302546&#93;,\n&#x27;
                         &#x27;       &#91;0.29710491&#93;,\n&#x27;
                         &#x27;       &#91;0.3057631 &#93;,\n&#x27;
                         &#x27;       &#91;0.29444755&#93;&#93;)&lt;/pre&gt;&lt;/div&gt;&lt;/li&gt;&lt;li &#x27;
                         &quot;class=&#x27;xr-var-item&#x27;&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-name&#x27;&gt;&lt;span&gt;soil_moisture_daily&lt;/span&gt;&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-dims&#x27;&gt;(time, pixel)&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-dtype&#x27;&gt;float64&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-preview xr-preview&#x27;&gt;212.4 215.1 215.4 &quot;
                         &#x27;... 232.5 238.0&lt;/div&gt;&lt;input &#x27;
                         &quot;id=&#x27;attrs-a992da68-2422-4dab-ab17-fddecd5875f5&#x27; &quot;
                         &quot;class=&#x27;xr-var-attrs-in&#x27; type=&#x27;checkbox&#x27; &quot;
                         &#x27;disabled&gt;&lt;label &#x27;
                         &quot;for=&#x27;attrs-a992da68-2422-4dab-ab17-fddecd5875f5&#x27; &quot;
                         &quot;title=&#x27;Show/Hide attributes&#x27;&gt;&lt;svg class=&#x27;icon &quot;
                         &quot;xr-icon-file-text2&#x27;&gt;&lt;use &quot;
                         &quot;xlink:href=&#x27;#icon-file-text2&#x27;&gt;&lt;/use&gt;&lt;/svg&gt;&lt;/label&gt;&lt;input &quot;
                         &quot;id=&#x27;data-620f522e-61e4-417f-ba15-505cbe43a38c&#x27; &quot;
                         &quot;class=&#x27;xr-var-data-in&#x27; type=&#x27;checkbox&#x27;&gt;&lt;label &quot;
                         &quot;for=&#x27;data-620f522e-61e4-417f-ba15-505cbe43a38c&#x27; &quot;
                         &quot;title=&#x27;Show/Hide data repr&#x27;&gt;&lt;svg class=&#x27;icon &quot;
                         &quot;xr-icon-database&#x27;&gt;&lt;use &quot;
                         &quot;xlink:href=&#x27;#icon-database&#x27;&gt;&lt;/use&gt;&lt;/svg&gt;&lt;/label&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-attrs&#x27;&gt;&lt;dl &quot;
                         &quot;class=&#x27;xr-attrs&#x27;&gt;&lt;/dl&gt;&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-data&#x27;&gt;&lt;pre&gt;array(&#91;&#91;212.42937151&#93;,\n&quot;
                         &#x27;       &#91;215.05458418&#93;,\n&#x27;
                         &#x27;       &#91;215.39787179&#93;,\n&#x27;
                         &#x27;       &#91;216.69619068&#93;,\n&#x27;
                         &#x27;       &#91;217.00471729&#93;,\n&#x27;
                         &#x27;       &#91;236.00704947&#93;,\n&#x27;
                         &#x27;       &#91;236.34398619&#93;,\n&#x27;
                         &#x27;       &#91;236.72217307&#93;,\n&#x27;
                         &#x27;       &#91;243.49392362&#93;,\n&#x27;
                         &#x27;       &#91;243.92442002&#93;,\n&#x27;
                         &#x27;       &#91;247.40464952&#93;,\n&#x27;
                         &#x27;       &#91;247.60189784&#93;,\n&#x27;
                         &#x27;       &#91;251.62609766&#93;,\n&#x27;
                         &#x27;       &#91;251.841253  &#93;,\n&#x27;
                         &#x27;       &#91;253.86081944&#93;,\n&#x27;
                         &#x27;       &#91;260.28031484&#93;,\n&#x27;
                         &#x27;       &#91;260.5042381 &#93;,\n&#x27;
                         &#x27;       &#91;260.69445708&#93;,\n&#x27;
                         &#x27;       &#91;266.74114969&#93;,\n&#x27;
                         &#x27;       &#91;267.01363955&#93;,\n&#x27;
                         &#x27;...\n&#x27;
                         &#x27;       &#91;211.82712465&#93;,\n&#x27;
                         &#x27;       &#91;212.21022551&#93;,\n&#x27;
                         &#x27;       &#91;212.55479028&#93;,\n&#x27;
                         &#x27;       &#91;215.2002562 &#93;,\n&#x27;
                         &#x27;       &#91;216.65723805&#93;,\n&#x27;
                         &#x27;       &#91;216.90242897&#93;,\n&#x27;
                         &#x27;       &#91;217.13295343&#93;,\n&#x27;
                         &#x27;       &#91;217.42614989&#93;,\n&#x27;
                         &#x27;       &#91;222.74777473&#93;,\n&#x27;
                         &#x27;       &#91;224.46663826&#93;,\n&#x27;
                         &#x27;       &#91;224.8138381 &#93;,\n&#x27;
                         &#x27;       &#91;225.22962767&#93;,\n&#x27;
                         &#x27;       &#91;225.60095702&#93;,\n&#x27;
                         &#x27;       &#91;226.10832126&#93;,\n&#x27;
                         &#x27;       &#91;231.10086916&#93;,\n&#x27;
                         &#x27;       &#91;231.3978305 &#93;,\n&#x27;
                         &#x27;       &#91;231.74611458&#93;,\n&#x27;
                         &#x27;       &#91;232.12029734&#93;,\n&#x27;
                         &#x27;       &#91;232.53378091&#93;,\n&#x27;
                         &#x27;       &#91;238.035744  &#93;&#93;)&lt;/pre&gt;&lt;/div&gt;&lt;/li&gt;&lt;li &#x27;
                         &quot;class=&#x27;xr-var-item&#x27;&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-name&#x27;&gt;&lt;span&gt;runoff_daily&lt;/span&gt;&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-dims&#x27;&gt;(time, pixel)&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-dtype&#x27;&gt;float64&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-preview xr-preview&#x27;&gt;0.0 0.0 0.0 0.0 &quot;
                         &#x27;... 0.0 0.0 0.0 0.0&lt;/div&gt;&lt;input &#x27;
                         &quot;id=&#x27;attrs-3b680ee2-15fd-442c-81ae-570af4da5341&#x27; &quot;
                         &quot;class=&#x27;xr-var-attrs-in&#x27; type=&#x27;checkbox&#x27; &quot;
                         &#x27;disabled&gt;&lt;label &#x27;
                         &quot;for=&#x27;attrs-3b680ee2-15fd-442c-81ae-570af4da5341&#x27; &quot;
                         &quot;title=&#x27;Show/Hide attributes&#x27;&gt;&lt;svg class=&#x27;icon &quot;
                         &quot;xr-icon-file-text2&#x27;&gt;&lt;use &quot;
                         &quot;xlink:href=&#x27;#icon-file-text2&#x27;&gt;&lt;/use&gt;&lt;/svg&gt;&lt;/label&gt;&lt;input &quot;
                         &quot;id=&#x27;data-f9e6903a-645e-4dcb-93d1-348c4cae0a8e&#x27; &quot;
                         &quot;class=&#x27;xr-var-data-in&#x27; type=&#x27;checkbox&#x27;&gt;&lt;label &quot;
                         &quot;for=&#x27;data-f9e6903a-645e-4dcb-93d1-348c4cae0a8e&#x27; &quot;
                         &quot;title=&#x27;Show/Hide data repr&#x27;&gt;&lt;svg class=&#x27;icon &quot;
                         &quot;xr-icon-database&#x27;&gt;&lt;use &quot;
                         &quot;xlink:href=&#x27;#icon-database&#x27;&gt;&lt;/use&gt;&lt;/svg&gt;&lt;/label&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-attrs&#x27;&gt;&lt;dl &quot;
                         &quot;class=&#x27;xr-attrs&#x27;&gt;&lt;/dl&gt;&lt;/div&gt;&lt;div &quot;
                         &quot;class=&#x27;xr-var-data&#x27;&gt;&lt;pre&gt;array(&#91;&#91;0.00000000e+00&#93;,\n&quot;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;...\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#91;0.00000000e+00&#93;,\n&#x27;
                         &#x27;       &#x27;
                         &#x27;&#91;0.00000000e+00&#93;&#93;)&lt;/pre&gt;&lt;/div&gt;&lt;/li&gt;&lt;/ul&gt;&lt;/div&gt;&lt;/li&gt;&lt;/ul&gt;&lt;/div&gt;&lt;/div&gt;&#x27;}</pre>

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

<!-- @output:nWHF -->

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA90AAAEiCAYAAADklbFjAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjksIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvJkbTWQAAAAlwSFlzAAAPYQAAD2EBqD+naQAAgylJREFUeJzt3Xd4VGXax/HvzKST3hNIIPRepEa6dOxYUEHsFbDXV1fF1WVX19521wL2DhZUEJGm9N4CBEhIQnpCes+c948JA5EekkzK73Ndc12Zc86c3ONgztzneZ77NhmGYSAiIiIiIiIitc7s6ABEREREREREmiol3SIiIiIiIiJ1REm3iIiIiIiISB1R0i0iIiIiIiJSR5R0i4iIiIiIiNQRJd0iIiIiIiIidURJt4iIiIiIiEgdUdItIiIiIiIiUkeUdIuIiIiIiIjUESXdIiIijVibNm248cYb6+z8N954I23atKmz84uIiDR1SrpFRESkwTl06BBXX301vr6+eHt7c+mll3LgwAFHhyUiInLWnBwdgIiIiNTcnj17MJub1j30goICRo4cSW5uLv/3f/+Hs7Mzr7zyCsOHD2fLli0EBAQ4OkQREZEzpqRbRESkEXN1dXV0CLXu7bffJjY2lnXr1tG/f38AJkyYQPfu3XnppZf4xz/+4eAIRUREzlzTujUuIiLSBDzzzDOYTCZ2797N1Vdfjbe3NwEBAdx7772UlJRUO/bYNd2GYTBy5EiCgoJIT0+3H1NWVkaPHj1o164dhYWF9u2ffPIJffv2xd3dHX9/f6655hoSExPr5T2eyjfffEP//v3tCTdA586dGTVqFF999ZUDIxMRETl7SrpFREQaqKuvvpqSkhJmz57NxIkTef3117n99ttPerzJZOKDDz6gpKSEO++807796aefZufOncyZM4cWLVoA8PzzzzNt2jQ6dOjAyy+/zH333ceSJUsYNmwYOTk5Zx1rQUEBmZmZp33k5uae8jxWq5Vt27bRr1+/4/YNGDCA/fv3k5+ff9bxiYiIOIqml4uIiDRQUVFRfP/99wBMnz4db29v3n77bR566CF69ux50te89NJL3HHHHXz66ae0b9+eF198kXvvvZdhw4YBcPDgQZ5++mmee+45/u///s/+2kmTJtGnTx/efvvtatvPxIwZM/jwww9Pe9zw4cNZtmzZSfdnZ2dTWlpKWFjYcfuObEtOTqZTp05nFZ+IiIijKOkWERFpoKZPn17t+cyZM3n77bf5+eefT5p0A9x+++3MmzePmTNnEhgYSLt27aqtg543bx5Wq5Wrr76azMxM+/bQ0FA6dOjA0qVLzzrpfuSRR5g6deppj/Pz8zvl/uLiYuDEa9Xd3NyqHSMiItIYKOkWERFpoDp06FDtebt27TCbzcTHx5/2te+//z7t2rUjNjaWVatW4e7ubt8XGxuLYRjHnf8IZ2fns461a9eudO3a9axf91dH4iwtLT1u35H17Me+FxERkYZOSbeIiEgjYTKZzvjYZcuW2RPX7du3Ex0dbd9ntVoxmUz88ssvWCyW417r6el51rHl5uae0Qi0i4sL/v7+J93v7++Pq6srKSkpx+07si08PPys4xMREXEUJd0iIiINVGxsLFFRUfbn+/btw2q10qZNm1O+LiUlhZkzZzJ27FhcXFx46KGHGDduHK1btwZsI+aGYRAVFUXHjh1rJdZ77723VtZ0m81mevTowYYNG47bt3btWtq2bYuXl9e5hCoiIlKvlHSLiIg0UG+99RZjx461P3/jjTcAW8/qU7ntttuwWq28//77WCwWunXrxi233MLixYsxmUxMmjSJxx9/nFmzZvHJJ59UG0E3DIPs7GwCAgLOKtbaWtMNcOWVV/LYY4+xYcMGexXzPXv28Pvvv/PQQw+dVVwiIiKOpqRbRESkgYqLi+OSSy5h/PjxrF69mk8++YTrrruOXr16nfQ1c+bM4aeffmLu3Lm0atUKsCXrU6dO5Z133uHuu++mXbt2PPfcczz++OPEx8dz2WWX4eXlRVxcHPPnz+f2228/6+S2ttZ0A9x99928++67XHjhhTz00EM4Ozvz8ssvExISwoMPPlgrv0NERKS+qE+3iIhIA/Xll1/i6urKY489xk8//cSMGTN4//33T3p8UlIS999/PxdffDE33HCDffuUKVO4/PLLeeSRR4iLiwPgscce49tvv8VsNjNr1iweeughfvjhB8aOHcsll1xS5+/tVLy8vFi2bBnDhg3jueee429/+xu9evVi+fLlBAUFOTQ2ERGRs2UyDMNwdBAiIiJy1DPPPMOsWbPIyMggMDDQ0eGIiIjIOdBIt4iIiIiIiEgdUdItIiIiIiIiUkeUdIuIiIiIiIjUEa3pFhEREREREakjGukWERER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37+fsLAw+vbti7OzM0uWLLHv37NnDwkJCURHRzswShGR+vXg2I50CTv5zcP+Uf71GI2IiIiIHKtBr+l+6KGHuPjii2ndujXJyck8/fTTWCwWrr32Wnx8fLjlllt44IEH8Pf3x9vbm5kzZxIdHX3KImoiIk1NKz8Pfrl3KHvT8hn7ygqczCZWPDKSkvJKUvNKGNBGSbeIiIiIozTopDspKYlrr72WrKwsgoKCGDJkCGvWrCEoyNb25pVXXsFsNnPFFVdQWlrKuHHjePvttx0ctYiIY3QM8eKbO6MprzQI93UHoG2QppaLiIiIOFKjKqRWV850AbyIiIiIiIgINNFCaiIiIiIiIiKNiZJuERERERERkTqipFtERERERESkjijpFhEREREREakjSrpFRERERERE6oiSbhEREREREZE60qD7dNeXI13T8vLyHByJiIiIiIiINAZH8sfTdeFW0g3k5+cDEBER4eBIREREREREpDHJz8/Hx8fnpPtNxunS8mbAarWSnJyMl5cXJpPJ0eGcUF5eHhERESQmJp6y8bo4hj6fhk2fT8Onz6jh02fUMOlzadj0+TR8+owavob8GRmGQX5+PuHh4ZjNJ1+5rZFuwGw206pVK0eHcUa8vb0b3D82OUqfT8Omz6fh02fU8Okzapj0uTRs+nwaPn1GDV9D/YxONcJ9hAqpiYiIiIiIiNQRJd0iIiIiIiIidURJdyPh6urK008/jaurq6NDkRPQ59Ow6fNp+PQZNXz6jBomfS4Nmz6fhk+fUcPXFD4jFVITERERERERqSMa6RYRERERERGpI0q6RUREREREROqIkm4RabZMJhPfffedo8MQEZFmRtcfkeZFSXc9mj17Nv3798fLy4vg4GAuu+wy9uzZU+2YkpISpk+fTkBAAJ6enlxxxRWkpaXZ92/dupVrr72WiIgI3N3d6dKlC6+99tpxv2vZsmWcd955uLq60r59e+bOnVvXb69JW716NRaLhQsvvNDRocgp3HjjjVx22WWODkP+IjExkZtvvpnw8HBcXFxo3bo19957L1lZWWf0+mXLlmEymcjJyanbQJu4+roGpaSkcN1119GxY0fMZjP33Xdffby9JkvXn8ZB15+GS9cgx6uv68+8efMYM2YMQUFBeHt7Ex0dzaJFi+rlPZ6Oku56tHz5cqZPn86aNWtYvHgx5eXljB07lsLCQvsx999/Pz/++CNff/01y5cvJzk5mUmTJtn3b9y4keDgYD755BN27tzJE088weOPP86bb75pPyYuLo4LL7yQkSNHsmXLFu677z5uvfXWBvOPrjF6//33mTlzJitWrCA5OfmczlVZWYnVaq2lyEQatgMHDtCvXz9iY2P5/PPP2bdvH//5z39YsmQJ0dHRZGdnOzrEZqO+rkGlpaUEBQXx5JNP0qtXr3p9j02Rrj8iNadrUMNQX9efFStWMGbMGH7++Wc2btzIyJEjufjii9m8eXO9vt8TMsRh0tPTDcBYvny5YRiGkZOTYzg7Oxtff/21/ZiYmBgDMFavXn3S89x9993GyJEj7c8feeQRo1u3btWOmTx5sjFu3LhafgfNQ35+vuHp6Wns3r3bmDx5svH888/b9y1dutQAjAULFhg9evQwXF1djYEDBxrbt2+3HzNnzhzDx8fH+P77740uXboYFovFiIuLc8A7afpuuOEG49JLLzUMwzBat25tvPLKK9X29+rVy3j66aftzwFj/vz59RZfczR+/HijVatWRlFRUbXtKSkphoeHh3HnnXcahmEYJSUlxiOPPGK0atXKcHFxMdq1a2e89957RlxcnAFUe9xwww0OeCdNT11dg441fPhw4957763VuJsTXX8aD11/GiZdgxqm+rj+HNG1a1dj1qxZtRP4OdBItwPl5uYC4O/vD9ju4JSXlzN69Gj7MZ07dyYyMpLVq1ef8jxHzgG2qWjHngNg3LhxpzyHnNxXX31F586d6dSpE1OnTuWDDz7A+EunvYcffpiXXnqJ9evXExQUxMUXX0x5ebl9f1FREf/6179477332LlzJ8HBwfX9NkTqXXZ2NosWLeLuu+/G3d292r7Q0FCmTJnCl19+iWEYTJs2jc8//5zXX3+dmJgY/vvf/+Lp6UlERATffvstAHv27CElJeWES2rk7NXVNUhqj64/IjWna1DDVV/XH6vVSn5+foO4Rjk5OoDmymq1ct999zF48GC6d+8OQGpqKi4uLvj6+lY7NiQkhNTU1BOeZ9WqVXz55Zf89NNP9m2pqamEhIQcd468vDyKi4uP+8Mjp/b+++8zdepUAMaPH09ubi7Lly9nxIgR9mOefvppxowZA8CHH35Iq1atmD9/PldffTUA5eXlvP3225pqKc1KbGwshmHQpUuXE+7v0qULhw8fZv369Xz11VcsXrzYfsFt27at/bgjF8vg4ODj/j5KzdTlNUhqj64/IjWna1DDVJ/Xn3//+98UFBTY/x46kka6HWT69Ons2LGDL774osbn2LFjB5deeilPP/00Y8eOrcXo5Ig9e/awbt06rr32WgCcnJyYPHky77//frXjoqOj7T/7+/vTqVMnYmJi7NtcXFzo2bNn/QQt0sD8dWTur+Lj47FYLAwfPryeIhJdgxo+XX9EaoeuQQ1LfV1/PvvsM2bNmsVXX33VIGb4KOl2gBkzZrBgwQKWLl1Kq1at7NtDQ0MpKys7rjpiWloaoaGh1bbt2rWLUaNGcfvtt/Pkk09W2xcaGlqt2t+Rc3h7e2uU+yy9//77VFRUEB4ejpOTE05OTrzzzjt8++239qkxZ8Ld3R2TyVSHkcpfmc3m4y60x065lLrXvn17TCZTtQTgWDExMfj5+envUj2r62uQ1A5dfxovXX8aBl2DGp76uv588cUX3HrrrXz11VfHLbl1FCXd9cgwDGbMmMH8+fP5/fffiYqKqra/b9++ODs7s2TJEvu2PXv2kJCQUO1O9s6dOxk5ciQ33HADzz///HG/Jzo6uto5ABYvXlztHHJ6FRUVfPTRR7z00kts2bLF/ti6dSvh4eF8/vnn9mPXrFlj//nw4cPs3bv3pNOZpH4EBQWRkpJif56Xl0dcXJwDI2p+AgICGDNmDG+//TbFxcXV9qWmpvLpp58yefJkevTogdVqZfny5Sc8j4uLC2CrvCw1V1/XIDl3uv40brr+NAy6BjUc9Xn9+fzzz7npppv4/PPPG1arRYeUb2um7rrrLsPHx8dYtmyZkZKSYn8cW1HxzjvvNCIjI43ff//d2LBhgxEdHW1ER0fb92/fvt0ICgoypk6dWu0c6enp9mMOHDhgeHh4GA8//LARExNjvPXWW4bFYjEWLlxYr++3sZs/f77h4uJi5OTkHLfvkUceMfr162evHtutWzfjt99+M7Zv325ccsklRmRkpFFaWmoYxtHqsVL3jq0e+9hjjxmhoaHGihUrjG3bthmXXXaZ4enpqeqx9Wzv3r1GYGCgMXToUGP58uVGQkKC8csvvxjdu3c3OnToYGRlZRmGYRg33nijERERYcyfP984cOCAsXTpUuPLL780DMMwkpKSDJPJZMydO9dIT0838vPzHfmWGq36ugYZhmFs3rzZ2Lx5s9G3b1/juuuuMzZv3mzs3Lmz3t5rY6frT+Oj60/DpGtQw1Bf159PP/3UcHJyMt56661qx5zob2l9U9Jdj/hLy4Ejjzlz5tiPKS4uNu6++27Dz8/P8PDwMC6//HIjJSXFvv/pp58+4Tlat25d7XctXbrU6N27t+Hi4mK0bdu22u+QM3PRRRcZEydOPOG+tWvXGoDx2muvGYDx448/Gt26dTNcXFyMAQMGGFu3brUfqy899ef66683rrjiCsMwDCM3N9eYPHmy4e3tbURERBhz585VyxYHiY+PN2644QYjJCTEcHZ2NiIiIoyZM2camZmZ9mOKi4uN+++/3wgLCzNcXFyM9u3bGx988IF9/7PPPmuEhoYaJpNJ7VpqqD6vQWdyjJycrj+Nj64/DZeuQY5XX9ef4cOHn/CYhvCZmQzjNNUFROSkli1bxsiRIzl8+LAqWjYA48ePp3379rz55puODkVEpE7p+tOw6PojIqeiNd0i0ugdPnyYBQsWsGzZsgZTMENERJo+XX9E5EyoT7eINHo333wz69ev58EHH+TSSy91dDgiItJM6PojImdC08tFRERERERE6oiml4uIiIiIiIjUESXdIiIiIiIiInVESbeIiIiIiIhIHVHSLSIiIiIiIlJHlHSLiIiIiIiI1BEl3SIiIs3UsmXLMJlM5OTkODoUERGRJkstw0RERJqJESNG0Lt3b1599VUAysrKyM7OJiQkBJPJ5NjgREREmignRwcgIiIijuHi4kJoaKijwxAREWnSNL1cRESkGbjxxhtZvnw5r732GiaTCZPJxNy5c6tNL587dy6+vr4sWLCATp064eHhwZVXXklRUREffvghbdq0wc/Pj3vuuYfKykr7uUtLS3nooYdo2bIlLVq0YODAgSxbtswxb1RERKSB0Ui3iIhIM/Daa6+xd+9eunfvzrPPPgvAzp07jzuuqKiI119/nS+++IL8/HwmTZrE5Zdfjq+vLz///DMHDhzgiiuuYPDgwUyePBmAGTNmsGvXLr744gvCw8OZP38+48ePZ/v27XTo0KFe36eIiEhDo6RbRESkGfDx8cHFxQUPDw/7lPLdu3cfd1x5eTnvvPMO7dq1A+DKK6/k448/Ji0tDU9PT7p27crIkSNZunQpkydPJiEhgTlz5pCQkEB4eDgADz30EAsXLmTOnDn84x//qL83KSIi0gAp6RYRERE7Dw8Pe8INEBISQps2bfD09Ky2LT09HYDt27dTWVlJx44dq52ntLSUgICA+glaRESkAVPSLSIiInbOzs7VnptMphNus1qtABQUFGCxWNi4cSMWi6Xacccm6iIiIs2Vkm4REZFmwsXFpVoBtNrQp08fKisrSU9PZ+jQobV6bhERkaZA1ctFRESaiTZt2rB27Vri4+PJzMy0j1afi44dOzJlyhSmTZvGvHnziIuLY926dcyePZuffvqpFqIWERFp3JR0i4iINBMPPfQQFouFrl27EhQUREJCQq2cd86cOUybNo0HH3yQTp06cdlll7F+/XoiIyNr5fwiIiKNmckwDMPRQYiIiIiIiIg0RRrpFhEREREREakjSrpFRERERERE6oiSbhEREREREZE6oqRbREREREREpI4o6RYRERERERGpI0q6RUREREREROqIkm4RERERERGROqKkW0RERERERKSOKOkWERERERERqSNKukVERERERETqiJJuERERERERkTqipFtERERERESkjvw/2FQolGo5vaAAAAAASUVORK5CYII=" alt="png">

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