# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "satterc==0.8.0",
#   "ukceh-catena",
#   "marimo",
#   "matplotlib==3.10.9",
#   "xarray-annotated",
# ]
#
# [tool.uv.sources]
# satterc = { path = ".." }
# ukceh-catena = { git = "https://github.com/NERC-CEH/catena" }
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Getting started with SatTerC

    This notebook walks through running a SatTerC pipeline step by step.
    It is aimed at users who are new to SatTerC, and assumes only basic familiarity with Python.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    /// admonition | Running this notebook yourself

    ### Option A — standalone, using `uv` (recommended)

    [uv](https://docs.astral.sh/uv/) is a fast Python package manager and installer.
    From the repository root, run:

    ```bash
    uv run recipes/my_first_pipeline.py
    ```

    `uv` will read the dependency list embedded at the top of this file, install everything
    it needs into a temporary isolated environment, and open the notebook in your browser.
    You do not need to install SatTerC separately.

    ### Option B — using an existing Python environment

    If SatTerC is already installed in a Python environment (for example, the project
    development environment), activate that environment, move to the repository root,
    and run:

    ```bash
    marimo run recipes/my_first_pipeline.py
    ```
    ///
    """)
    return


@app.cell
def _():
    import tempfile
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    from catena import build_driver, run
    from catena.config import Config

    from satterc.scaffold.data_gen import generate_synthetic_data

    return (
        Config,
        Path,
        build_driver,
        generate_synthetic_data,
        mo,
        plt,
        run,
        tempfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: Configure the pipeline

    A SatTerC pipeline is described by a configuration file written in
    [TOML](https://toml.io/en/) — a simple, human-readable format. The config schema
    is [catena](https://github.com/NERC-CEH/catena)'s; SatTerC supplies the models.
    Every section activates a pipeline component.
    `[splash]` loads the SPLASH water-balance module, which SatTerC registers with catena so the section name alone is enough to find it.
    `[inputs.daily]` loads daily climate data from the given path, and `[outputs.daily]` saves the named variables to disk when the pipeline finishes.

    Node names are `{var}{suffix}`, with the suffix defaulting to the section label,
    so `temperature` under `[inputs.daily]` becomes the node `temperature_daily` —
    which is what SPLASH's parameter is called. Static variables are consumed under
    bare names, so that section sets `suffix = ""`.

    `latitude` appears in neither input section. The inputs are georeferenced, so
    catena computes each pixel's latitude from their coordinate reference system.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import textwrap

    config_toml = textwrap.dedent("""\
    [splash]

    [inputs.daily]
    path = "daily.nc"
    vars = [
      "precipitation",
      "sunshine_fraction",
      "temperature",
    ]

    [inputs.static]
    path = "static.nc"
    suffix = ""
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
    """)

    mo.md("```toml\n" + config_toml + "```")
    return (config_toml,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The same config as a dictionary

    TOML is only a serialisation format. `Config.loads` parses the string into an
    ordinary Python dictionary and everything downstream works from that, so the
    dictionary below is what catena actually sees.
    """)
    return


@app.cell
def _(config_toml):
    import tomllib

    config_dict = tomllib.loads(config_toml)
    config_dict
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Config` takes that dictionary directly — `Config.loads(config_toml)` is
    defined as `Config(tomllib.loads(config_toml))` and nothing more. So you can
    build a config in Python if you want to: one per parameter value for a sweep,
    or a pipeline driven by another program that already holds the settings as
    data.

    It is worth knowing what you trade away. A TOML file is one artefact you can
    version, diff, review and pass to `satterc run`; a dictionary assembled over
    fifty lines of Python is none of those. Write the file by default, and reach
    for the dictionary when you are generating configs rather than authoring them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: Generate synthetic input data

    SatTerC reads input data from NetCDF or Zarr files, detecting the format from
    the file extension listed in the configuration. SatTerC's built-in synthetic
    data generator writes either, so it is easy to produce realistic stand-in
    inputs.

    Since we do not have real data to hand, we will use the generator to produce
    NetCDF inputs. The generated data covers one virtual site over a two-year
    period.

    > **If you have real data**, skip ahead to the *Using your own data* section at the bottom
    > of this notebook before running the pipeline.

    ### A note on units

    Every model input in SatTerC declares the units it expects, and catena
    checks the data against that declaration before the model sees it. The check
    reads a `units` attribute on the array: the generator labels `temperature` as
    `degC`, and NetCDF keeps the label.

    A missing label is an error, and so is a different unit, even one that could
    be converted: `degF` where `degC` is declared fails rather than being converted
    for you. Convert your data before it enters the pipeline.
    """)
    return


@app.cell
def _(Config, Path, config_toml, generate_synthetic_data, tempfile):
    _tmpdir = Path(tempfile.mkdtemp())

    # Parse the embedded config string
    parsed_config = Config.loads(config_toml).parse()

    # Redirect every path in the config into a temporary directory: the inputs we
    # are about to generate, and the results the run will write.
    parsed_config.input_specs["daily"].path = str(_tmpdir / "daily.nc")
    parsed_config.input_specs["static"].path = str(_tmpdir / "static.nc")
    parsed_config.output_specs["daily"].path = str(_tmpdir / "results" / "daily.nc")

    # Generate synthetic data — this may take a few seconds
    generate_synthetic_data(config=parsed_config, grid=(1, 1), n_days=730, seed=42)

    print(f"Synthetic data written to: {_tmpdir}")
    return (parsed_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: Build the pipeline

    SatTerC represents a pipeline as a Directed Acyclic Graph (DAG) — a network of nodes
    where each node is a computation, and edges show which computations depend on which others.

    Building the pipeline means constructing this graph from the modules and configuration
    you specified. Below we visualise the portion of the graph running from the daily
    precipitation input through to the soil moisture output.
    """)
    return


@app.cell
def _(build_driver, parsed_config):
    dr = build_driver(
        modules=parsed_config.modules,
        config=parsed_config.driver_config,
        node_specs=parsed_config.node_specs,
    )
    return (dr,)


@app.cell
def _(dr):
    dr.visualize_path_between(
        "precipitation_daily",
        "soil_moisture_daily",
        show_legend=False,
        graphviz_kwargs={"graph_attr": {"ratio": "compress", "size": "10,15"}},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4: Run the pipeline

    `run` does the whole thing in one call: it loads the inputs, builds the DAG, computes the variables `[outputs.daily]` names, and writes them to the path that section gives.

    What comes back is a `RunReport`.
    `report.outputs` holds one Dataset per output section, keyed by section name, so the results are in memory to explore and plot as well as on disk.
    `report.written` says where each file went and how large it is.

    Note that `run` takes the config, not the driver we built in step 3.
    That step was to see the graph; running the pipeline does not need it.
    """)
    return


@app.cell
def _(parsed_config, run):
    report = run(parsed_config)

    report.outputs["daily"].info()
    return (report,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5: Inspect the results

    Let us plot the simulated soil moisture over the two-year period.
    Soil moisture rises after precipitation events and falls during dry periods —
    a clear seasonal signal should be visible.
    """)
    return


@app.cell
def _(plt, report):
    # Written outputs are back on the inputs' (y, x) grid; this one has one cell.
    soil_moisture = report.outputs["daily"]["soil_moisture"].isel(y=0, x=0)

    fig, ax = plt.subplots(figsize=(10, 3))
    soil_moisture.plot(ax=ax)
    ax.set_ylabel("Soil moisture (mm)")
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using your own data

    To run the pipeline on real data instead of synthetic data, follow these steps.

    ### 1. Save the config to a file

    Run the cell below. It will write the embedded config to a file called
    `my_pipeline.toml` in your current working directory.
    """)
    return


@app.cell
def _(Path):
    _output_path = Path("my_pipeline.toml")

    # Uncomment this line!
    # _output_path.write_text(config_toml.strip())
    # print(f"Config written to: {_output_path.resolve()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Prepare your data files

    Write each input section to a NetCDF file with xarray, with a `units`
    attribute on every variable. For a single site with no coordinate reference
    system, use a `pixel` dimension of length one, and supply `latitude` yourself:

    ```python
    import pandas as pd
    import xarray as xr

    time = pd.date_range("2020-01-01", periods=731, freq="D")
    daily = xr.Dataset(
        {
            "precipitation": (("time", "pixel"), precip, {"units": "mm d-1"}),
            "sunshine_fraction": (("time", "pixel"), sunshine, {"units": "1"}),
            "temperature": (("time", "pixel"), temp, {"units": "degC"}),
        },
        coords={"time": time, "pixel": [0]},
    )
    daily.to_netcdf("daily.nc")

    static = xr.Dataset(
        {
            "elevation": ("pixel", [150.0], {"units": "m"}),
            "latitude": ("pixel", [51.5], {"units": "degrees_north"}),
            "max_soil_moisture": ("pixel", [200.0], {"units": "mm"}),
        },
        coords={"pixel": [0]},
    )
    static.to_netcdf("static.nc")
    ```

    The model docs list the unit each input is declared in.

    ### 3. Edit the config file

    Open `my_pipeline.toml` in a text editor.
    Under each `[inputs.*]` section, change the `path` value to point to your real file.
    Paths can be absolute or relative to the location of the config file. For example:

    ```toml
    [inputs.daily]
    path = "/data/my-site/daily.nc"
    vars = [
      "precipitation",
      "sunshine_fraction",
      "temperature",
    ]

    [inputs.static]
    path = "/data/my-site/static.nc"
    suffix = ""
    vars = [
      "elevation",
      "latitude",
      "max_soil_moisture",
    ]
    ```

    `latitude` is listed here because these files have no coordinate reference
    system for catena to compute it from.

    ### 4. Load the config from the file

    Replace the config and data-generation cells in this notebook with:

    ```python
    from catena import load_config

    parsed_config = load_config("my_pipeline.toml")
    ```

    `load_config` reads the TOML file and resolves all paths relative to the file's location.

    ### 5. Remove the data generation cell

    You no longer need to generate synthetic data — delete that cell.
    The pipeline will load your real files directly.
    """)
    return


if __name__ == "__main__":
    app.run()
