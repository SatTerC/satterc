# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "matplotlib",
# ]
# ///
import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Example notebook

    This notebook is intended to show how to run a pipeline from within a Python environment, rather than using the CLI command. We will keep the computed outputs, which are `xarray.Dataset` objects, in memory rather than saving them to netcdf files.
    """)
    return


@app.cell
def _():
    import tempfile
    import tomllib
    from pathlib import Path

    import marimo as mo  # required for Markdown etc.
    import matplotlib.pyplot as plt

    from satterc import build_driver
    from satterc.config import Config
    from satterc.setup_utils.data_gen import generate_synthetic_data

    return (
        Config,
        Path,
        build_driver,
        generate_synthetic_data,
        mo,
        plt,
        tempfile,
        tomllib,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pipeline configuration

    The pipeline configuration is defined in a [TOML](https://toml.io/en/) file.

    `satterc` provides a loader / parser for pipeline configurations, which takes a path to a config file and returns a dict with three items:

    1. `modules`: a list of Python modules containing the nodes (functions) which will be used to construct the pipeline.
    2. `driver_config`: a dictionary of additional config options that is applied to the driver at _build time_ (not run time).
    3. `targets`: a list of node (function) names that are the end-points of the pipeline. By default these will be the nodes that save output data to the disk, but this can be customised (as we will see later).
    """)
    return


@app.cell
def _(Config, tomllib):
    _config_toml = """
modules = [
  "models.pmodel",
  "models.splash",
  "models.sgam",
  "models.rothc",
  "inputs.daily",
  "inputs.weekly",
  "inputs.monthly",
  "inputs.static",
  "resample",
  "outputs.daily",
  "outputs.weekly",
  "outputs.monthly",
]

[extra_config]
n_years_spinup = 1

[models.pmodel]
method_kphio = "sandoval"
method_optchi = "lavergne20_c3"

[inputs.daily]
path = "daily.nc"
vars = [
  "precipitation_mm",
  "sunshine_fraction",
  "temperature_celcius",
  "lai",
  "gpp",
]

[inputs.weekly]
path = "weekly.nc"
vars = [
  "co2_ppm",
  "fapar",
  "ppfd_umol_m2_s1",
  "pressure_pa",
  "vpd_pa",
]

[inputs.monthly]
path = "monthly.nc"
vars = [
  "dummy_variable",
]

[inputs.static]
path = "static.nc"
vars = [
  "elevation",
  "plant_type",
  "max_soil_moisture",
  "clay_content",
  "soil_depth",
  "organic_carbon_stocks",
  "root_pool_init",
  "leaf_pool_init",
  "stem_pool_init",
]

[resample]
daily_to_weekly = [
  "temperature_celcius",
  "precipitation_mm",
  "soil_moisture",
  "aridity_index",
]

daily_to_monthly = [
  "temperature_celcius",
  "precipitation_mm",
  "actual_evapotranspiration",
]

weekly_to_monthly = [
  "litter_pool",
]

[outputs.daily]
path = "results/daily.nc"
vars = [
  "actual_evapotranspiration",
  "soil_moisture",
  "runoff",
]

[outputs.weekly]
path = "results/weekly.nc"
vars = [
  "gpp",
  "leaf_pool",
  "stem_pool",
  "root_pool",
  "litter_pool",
  "leaf_area_index",
]

[outputs.monthly]
path = "results/monthly.nc"
vars = [
  "decomposable_plant_material",
  "resistant_plant_material",
  "microbial_biomass",
  "humified_organic_matter",
  "soil_organic_carbon",
]
"""

    # Parse config directly from the TOML string (no file needed)
    parsed_config = Config(tomllib.loads(_config_toml)).parse()

    # `parsed_config` contains (1) `modules`, (2) `driver_config`, (3) `targets`
    parsed_config
    return (parsed_config,)


@app.cell
def _(Path, generate_synthetic_data, parsed_config, tempfile):
    # Generate synthetic input data into a temporary directory
    _tmpdir = Path(tempfile.mkdtemp())

    parsed_config["driver_config"]["daily_inputs_path"] = str(_tmpdir / "daily.nc")
    parsed_config["driver_config"]["weekly_inputs_path"] = str(_tmpdir / "weekly.nc")
    parsed_config["driver_config"]["monthly_inputs_path"] = str(_tmpdir / "monthly.nc")
    parsed_config["driver_config"]["static_inputs_path"] = str(_tmpdir / "static.nc")

    generate_synthetic_data(config=parsed_config, grid=(4, 4), n_days=730, seed=42)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Building the pipeline

    Building the pipeline really means building the 'driver'.

    `satterc` provides a function `build_driver` which takes a list of modules and a driver configuration, and returns a built driver.
    Notice that we do not pass `targets` during build stage; we are only required to supply `targets` when actually executing the pipeline.

    Once we have constructed the driver, we can inspect it in various ways, including visualising the DAG.
    Since the entire pipeline is very large, we can focus on visualising sub-DAGs between two given nodes.
    """)
    return


@app.cell
def _(build_driver, parsed_config):
    # Build the driver object
    dr = build_driver(
        modules=parsed_config["modules"],
        config=parsed_config["driver_config"],
    )

    # This produces a visualisation of the entire DAG, which is too large..
    # dr
    return (dr,)


@app.cell
def _(dr):
    # Here we restrict the visualisation to a sub-DAG between sgam and rothc
    # NOTE: I need to figure out how to filter out the config inputs!
    dr.visualize_path_between(
        "sgam",
        "rothc",
        # strict_path_visualization=True,
        show_legend=False,
        # Make the graph smaller to fit the screen
        graphviz_kwargs={
            "graph_attr": {
                "ratio": "compress",
                "size": "10,15",  # Width and height in inches
            }
        },
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running the pipeline

    To run the pipeline we can call the `execute` method of the driver object, providing

    1. `final_vars`: A list of 'target' nodes to be computed (e.g. `parsed_config[targets]`).
    2. Optionally, `overrides`: overrides for any nodes in the DAG.\*
    3. Optionally, `inputs`: extra run-time inputs.

    Recall that `parsed_config['targets']` are the 'save' nodes which save the merged outputs to the disk.
    These are the default `final_vars` when you run the pipeline using the `satterc` command-line interface (CLI).

    Since we are in a notebook, we might prefer to stop the pipeline just before saving, and instead have the driver return the `xarray.Dataset` objects.
    We will demonstrate that here by requesting computed outputs at daily, weekly and monthly resolution.

    \* The `overrides` option will be useful later on when we want to run the DAG repeatedly with different parameter values, without rebuilding it from scratch each time.
    """)
    return


@app.cell
def _(dr):
    _outputs = dr.execute(
        ["merged_daily_outputs", "merged_weekly_outputs", "merged_monthly_outputs"]
    )

    _outputs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparing input LAI with modelled LAI

    We can compare input and modelled LAI by requesting those two nodes as `final_vars`.
    """)
    return


@app.cell
def _(dr, plt):
    _outputs = dr.execute(["lai_daily", "leaf_area_index_weekly"])

    input_lai, modelled_lai = _outputs["lai_daily"], _outputs["leaf_area_index_weekly"]

    n_pixels = input_lai.sizes["pixel"]
    fig, axes = plt.subplots(n_pixels, 1, figsize=(10, 3 * n_pixels), sharex=True)
    for i, ax in enumerate(axes):
        input_lai[:, i].plot(ax=ax, label="Input LAI")
        modelled_lai[:, i].plot(ax=ax, label="Modelled LAI")
        ax.legend()
    fig.tight_layout()

    fig
    return


if __name__ == "__main__":
    app.run()
