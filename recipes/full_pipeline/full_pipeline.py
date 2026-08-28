# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.9",
#     "satterc==0.7.0",
#     "conduit",
# ]
#
# [tool.uv.sources]
# satterc = { path = ".." }
# conduit = { git = "https://github.com/NERC-CEH/conduit", rev = "develop" }
# ///

import marimo

__generated_with = "0.23.5"
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
    from conduit import build_driver, get_final_vars, get_outputs, load_inputs
    from conduit.config import Config

    from satterc.scaffold.data_gen import generate_synthetic_data

    return (
        Config,
        Path,
        build_driver,
        generate_synthetic_data,
        get_final_vars,
        get_outputs,
        load_inputs,
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

    `conduit` provides a loader / parser for pipeline configurations, which takes a path to a config file and returns a `ParsedConfig`. The attributes we use here are:

    1. `modules`: a list of Python modules containing the nodes (functions) which will be used to construct the pipeline.
    2. `driver_config`: a dictionary of additional config options that is applied to the driver at _build time_ (not run time).
    3. `node_specs`: the parsed `[[node]]` and `[[resample]]` entries, from which conduit generates a module of nodes.
    4. `input_specs`: a mapping from section label to `IOSpec` (path, vars) — consumed by `load_inputs()`.
    5. `output_specs`: a mapping from section label to `IOSpec` — consumed by `get_outputs()` and `save_outputs()`.
    """)
    return


@app.cell
def _(Config, tomllib):
    _config_toml = """

    [splash]
    _import_path = "satterc.models.splash"

    [pmodel]
    _import_path = "satterc.models.pmodel"
    method_kphio = "sandoval"
    method_optchi = "lavergne20_c3"

    [sgam]
    _import_path = "satterc.models.sgam"

    [rothc]
    _import_path = "satterc.models.rothc"
    n_years_spinup = 1
    equilibrium_threshold = 0.0001


    [inputs.daily]
    path = "daily.nc"
    vars = [
      "precipitation",
      "sunshine_fraction",
      "temperature",
      "lai",
      "gpp",
    ]

    [inputs.weekly]
    path = "weekly.nc"
    vars = [
      "co2",
      "fapar",
      "ppfd",
      "pressure",
      "vpd",
    ]

    [inputs.monthly]
    path = "monthly.nc"
    vars = [
      "dummy_variable",
    ]

    [inputs.static]
    path = "static.nc"
    suffix = ""
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

    # pyrealm's `aridity_index` is climatological and oriented PET/P: both terms
    # accumulate over the whole record, giving one value per pixel rather than a
    # time series. SPLASH computes PET on its way to AET and exposes it.
    [[node]]
    name = "aridity_index"
    inputs = ["potential_evapotranspiration_daily", "precipitation_daily"]
    expression = "potential_evapotranspiration_daily.sum('time') / precipitation_daily.sum('time')"
    units = "1"  # ratio of two mm totals -> dimensionless

    # Both the P-model (as pyrealm's `theta`) and SGAM (against the PFT's
    # `wilting_point`/`field_capacity`) want volumetric water content (m3 m-3).
    # SPLASH reports a depth of water in mm held in a bucket of capacity
    # `max_soil_moisture`, so dividing gives relative saturation (0-1) and
    # multiplying by a representative mineral-soil porosity (0.45) turns that
    # into a volume fraction.
    [[node]]
    name = "volumetric_water_content_weekly"
    inputs = ["soil_moisture_weekly", "max_soil_moisture"]
    expression = "soil_moisture_weekly / max_soil_moisture * 0.45"
    units = "m3 m-3"
    freq = "7D"

    # satterc.models.pmodel is an honest wrapper. It reports GPP and LUE in
    # pyrealm's own units, which are not the ones SGAM consumes. These two nodes
    # are the bridge, kept here rather than inside the model node so that a
    # reader can see the conversion factor.
    #
    # pyrealm's GPP is an instantaneous flux in ug C m-2 s-1. SGAM wants a daily
    # rate in g C m-2 d-1, so 86400 s d-1 x 1e-6 g ug-1 = 0.0864.
    [[node]]
    name = "gpp_weekly"
    inputs = ["gpp_flux_weekly"]
    expression = "gpp_flux_weekly * 0.0864"
    units = "g m-2 d-1"
    freq = "7D"

    # pyrealm defines LUE against PPFD, a *photon* flux, so its denominator is
    # moles of photons. SGAM wants carbon per MJ of absorbed PAR, and 4.57 mol
    # MJ-1 is the standard photon content of PAR over the 400-700 nm band.
    [[node]]
    name = "lue_weekly"
    inputs = ["lue_photon_weekly"]
    expression = "lue_photon_weekly * 4.57"
    units = "g MJ-1"
    freq = "7D"

    # iWUE needs no bridge: pyrealm reports umol mol-1 and SGAM consumes it.

    [[node]]
    name = "leaf_area_index_weekly"
    inputs = ["leaf_pool_weekly", "pft_params"]
    expression = 'leaf_pool_weekly / pft_params["leaf_carbon_area"]'
    units = "m2 m-2"  # leaf carbon per ground area / leaf carbon per leaf area

    # PET, not AET: RothC's water balance is rainfall minus evaporative *demand*.
    # AET is already suppressed by the dryness RothC is trying to compute, so
    # feeding it in double-counts the limitation and holds the soil too wet.
    # SPLASH PET is a daily rate (mm d-1); RothC wants a monthly total (mm).
    # Summing the daily rate over the month integrates it (daily Δt = 1 day, so
    # Σ mm d-1 is numerically the monthly mm total); units = "mm" relabels the
    # rate as the resulting total.
    [[node]]
    name = "potential_evapotranspiration_monthly"
    inputs = ["potential_evapotranspiration_daily"]
    expression = "potential_evapotranspiration_daily.resample(time='1ME').sum()"
    units = "mm"

    # Precipitation is likewise a daily rate (mm d-1); aggregate to a monthly
    # total (mm) for RothC the same way. (Done as a derive rather than a plain
    # [[resample]] because that would feed the mm d-1 rate straight into RothC's
    # mm input and the resample output name would collide with this one.)
    [[node]]
    name = "precipitation_monthly"
    inputs = ["precipitation_daily"]
    expression = "precipitation_daily.resample(time='1ME').sum()"
    units = "mm"

    # Carbon entering the soil each month = the litter produced that month. SGAM's
    # litter_pool is an accumulate-only stock (no decomposition; that is RothC's
    # job), so the monthly litterfall is its *increment*: diff the weekly pool and
    # sum within each month. Using the increment (rather than summing turnover_*)
    # also captures litter from disturbance events, which the turnover outputs omit.
    # SGAM reports g m-2; RothC wants t ha-1, so convert with pint (factor 100).
    [[node]]
    name = "soil_carbon_input_monthly"
    inputs = ["litter_pool_weekly"]
    expression = "litter_pool_weekly.diff('time').resample(time='1ME').sum().assign_attrs(units='g m-2').pint.quantify().pint.to('t ha-1').pint.dequantify()"
    units = "t ha-1"

    [[node]]
    name = "inert_organic_matter"
    inputs = ["organic_carbon_stocks"]
    expression = "0.049 * organic_carbon_stocks**1.139"  # Falloon IOM (t ha-1)
    units = "t ha-1"

    [[resample]]
    vars = [
      "temperature",
      "precipitation",
      "soil_moisture",
    ]
    from = "daily"
    to = "weekly"
    freq = "7D"

    [[resample]]
    vars = [
      "temperature",
    ]
    from = "daily"
    to = "monthly"
    freq = "1ME"

    [[resample]]
    vars = ["disturbances"]
    from = "daily"
    to = "weekly"
    freq = "7D"
    aggfunc = "max"

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
      "heterotrophic_respiration",
    ]
    """

    # Parse config directly from the TOML string (no file needed)
    parsed_config = Config(tomllib.loads(_config_toml)).parse()

    # `parsed_config` contains (1) `modules`, (2) `driver_config`, (3) `input_specs`, (4) `output_specs`
    parsed_config
    return (parsed_config,)


@app.cell
def _(Path, generate_synthetic_data, load_inputs, parsed_config, tempfile):
    # Generate synthetic input data into a temporary directory
    _tmpdir = Path(tempfile.mkdtemp())

    parsed_config.input_specs["daily"].path = str(_tmpdir / "daily.nc")
    parsed_config.input_specs["weekly"].path = str(_tmpdir / "weekly.nc")
    parsed_config.input_specs["monthly"].path = str(_tmpdir / "monthly.nc")
    parsed_config.input_specs["static"].path = str(_tmpdir / "static.nc")

    generate_synthetic_data(config=parsed_config, grid=(4, 4), n_days=730, seed=42)

    inputs = load_inputs(parsed_config.input_specs)
    return (inputs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Building the pipeline

    Building the pipeline really means building the 'driver'.

    `conduit` provides a function `build_driver` which takes a list of modules, a driver configuration and the parsed node specs, and returns a built driver.
    Building is also when conduit checks the whole graph's declared contracts — units, dimensions and temporal frequency — so a mis-wired pipeline fails here rather than part-way through a long run.
    Notice that we do not pass `targets` during build stage; we are only required to supply `targets` when actually executing the pipeline.

    Once we have constructed the driver, we can inspect it in various ways, including visualising the DAG.
    Since the entire pipeline is very large, we can focus on visualising sub-DAGs between two given nodes.
    """)
    return


@app.cell
def _(build_driver, parsed_config):
    # Build the driver object
    dr = build_driver(
        modules=parsed_config.modules,
        config=parsed_config.driver_config,
        node_specs=parsed_config.node_specs,
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

    To run the pipeline we call `dr.execute()`, providing:

    1. `final_vars`: the output variable names to compute.
    2. `inputs`: the raw input DataArrays returned by `load_inputs()`.
    3. Optionally, `overrides`: overrides for any computed node in the DAG.\*

    We collect the output variable names from `parsed_config.output_specs` and merge them
    into per-frequency Datasets using `get_outputs()`.

    \* The `overrides` option will be useful later on when we want to run the DAG repeatedly with different parameter values, without rebuilding it from scratch each time.
    """)
    return


@app.cell
def _(dr, get_final_vars, get_outputs, inputs, parsed_config):
    _target_vars = get_final_vars(parsed_config.output_specs)
    _results = dr.execute(_target_vars, inputs=inputs)
    _output_datasets = get_outputs(_results, parsed_config.output_specs)
    _output_datasets["monthly"].info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparing input LAI with modelled LAI

    We can compare input and modelled LAI by requesting those two nodes as `final_vars`.
    """)
    return


@app.cell
def _(dr, inputs, plt):
    _outputs = dr.execute(["lai_daily", "leaf_area_index_weekly"], inputs=inputs)

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
