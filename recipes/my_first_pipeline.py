# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "satterc==0.7.0",
#   "conduit",
#   "marimo",
#   "matplotlib==3.10.9",
#   "xarray-annotated",
# ]
#
# [tool.uv.sources]
# satterc = { path = ".." }
# conduit = { git = "https://github.com/NERC-CEH/conduit", rev = "develop" }
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
    ///
    """)
    return


@app.cell
def _():
    import tempfile
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    from conduit import build_driver, run
    from conduit.config import Config
    from xarray_annotated.units import set_policy

    from satterc.scaffold.data_gen import generate_synthetic_data

    # CSV and JSON files have nowhere to record a unit, so the inputs below
    # arrive unlabelled and conduit warns once per input that it cannot check
    # them. See "A note on units" below.
    set_policy(on_missing="ignore")
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
    is [conduit](https://github.com/NERC-CEH/conduit)'s; SatTerC supplies the models.
    Every section activates a pipeline component.
    `[splash]` loads the SPLASH water-balance module, which SatTerC registers with conduit so the section name alone is enough to find it.
    `[inputs.daily]` loads daily climate data from the given path, and `[outputs.daily]` saves the named variables to disk when the pipeline finishes.

    Node names are `{var}{suffix}`, with the suffix defaulting to the section label,
    so `temperature` under `[inputs.daily]` becomes the node `temperature_daily` —
    which is what SPLASH's parameter is called. Static variables are consumed under
    bare names, so that section sets `suffix = "\"`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import textwrap

    config_toml = textwrap.dedent("""\
    [splash]

    [inputs.daily]
    path = "daily.csv"
    vars = [
      "precipitation",
      "sunshine_fraction",
      "temperature",
    ]

    [inputs.static]
    path = "static.json"
    suffix = ""
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
    """)

    mo.md("```toml\n" + config_toml + "```")
    return (config_toml,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The same config as a dictionary

    TOML is only a serialisation format. `Config.loads` parses the string into an
    ordinary Python dictionary and everything downstream works from that, so the
    dictionary below is what conduit actually sees.
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

    SatTerC reads input data from files — the format is detected automatically from
    the file extension listed in the configuration. SatTerC's built-in synthetic
    data generator supports NetCDF, Zarr, CSV, Parquet, and JSON output, making
    it easy to produce realistic stand-in inputs in whatever format you need.

    Since we do not have real data to hand, we will use the generator to produce
    single-site CSV and JSON inputs. The generated data covers one virtual site
    over a two-year period.

    > **If you have real data**, skip ahead to the *Using your own data* section at the bottom
    > of this notebook before running the pipeline.

    ### A note on units

    Every model input in SatTerC declares the units it expects, and conduit
    checks the data against that declaration before the model sees it. The check
    reads a `units` attribute on the array, which CSV and JSON have no way to
    store: the generator labels `temperature` as `degC`, and the label is lost
    the moment the column is written out.

    conduit warns once per unlabelled input by default. The import cell above
    turns those warnings off with `set_policy(on_missing="ignore")`, which
    affects only the unlabelled case — a label that contradicts the declaration
    is still an error. Write the inputs to NetCDF or Zarr instead and the labels
    survive the round trip, so the check does its job with nothing to configure.
    """)
    return


@app.cell
def _(Config, Path, config_toml, generate_synthetic_data, tempfile):
    _tmpdir = Path(tempfile.mkdtemp())

    # Parse the embedded config string
    parsed_config = Config.loads(config_toml).parse()

    # Redirect every path in the config into a temporary directory: the inputs we
    # are about to generate, and the results the run will write.
    parsed_config.input_specs["daily"].path = str(_tmpdir / "daily.csv")
    parsed_config.input_specs["static"].path = str(_tmpdir / "static.json")
    # conduit will not create a missing output directory; it fails rather than
    # guess. The config asks for `results/daily.csv`, so make `results/` first.
    (_tmpdir / "results").mkdir()
    parsed_config.output_specs["daily"].path = str(_tmpdir / "results" / "daily.csv")

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
    soil_moisture = report.outputs["daily"]["soil_moisture"].isel(pixel=0)

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

    **Daily CSV** — one row per day, first column a parseable date:

    ```csv
    time,precipitation,sunshine_fraction,temperature
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
      "precipitation",
      "sunshine_fraction",
      "temperature",
    ]

    [inputs.static]
    path = "/data/my-site/static.json"
    suffix = "\"
    vars = [
      "elevation",
      "latitude",
      "max_soil_moisture",
    ]
    ```

    ### 4. Load the config from the file

    Replace the config and data-generation cells in this notebook with:

    ```python
    from conduit import load_config

    parsed_config = load_config("my_pipeline.toml")
    ```

    `load_config` reads the TOML file and resolves all paths relative to the file's location.

    ### 5. Remove the data generation cell

    You no longer need to generate synthetic data — delete that cell.
    The pipeline will load your real CSV and JSON files directly.
    """)
    return


if __name__ == "__main__":
    app.run()
