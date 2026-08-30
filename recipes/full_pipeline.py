# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.9",
#     "satterc==0.8.0",
#     "conduit",
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
    # Wiring four models into one pipeline

    This recipe builds a pipeline out of all four of SatTerC's models at once: SPLASH for the water balance, the P-model for photosynthesis, SGAM for plant carbon allocation, and RothC for soil carbon.

    The models are the easy part.
    None of the four knows the others exist, and none of them imports another.
    Each declares its inputs by parameter name and its outputs by return value, and conduit matches those names when it builds the DAG.
    So the pipeline is not really four models; it is four models plus the joins between them, and the joins are where the work is.

    This notebook takes its time over those joins.
    Two models connect only when the name *and* the units line up, and between these four they mostly do not, so most of the configuration below exists to convert one model's output into the next one's input.
    We will draw each join as its own small graph before running anything.
    """)
    return


@app.cell
def _():
    import tempfile
    import tomllib
    from pathlib import Path

    import marimo as mo  # required for Markdown etc.
    import matplotlib.pyplot as plt
    from conduit import build_driver, run
    from conduit.config import Config
    from conduit.graph import _node_maps as node_maps
    from conduit.graph import relabel_with_units

    from satterc.scaffold.data_gen import generate_synthetic_data

    return (
        Config,
        Path,
        build_driver,
        generate_synthetic_data,
        mo,
        node_maps,
        plt,
        relabel_with_units,
        run,
        tempfile,
        tomllib,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What each model needs and produces

    SPLASH is a bucket water balance.
    Give it precipitation, temperature and sunshine fraction, and it returns soil moisture, runoff, and both potential and actual evapotranspiration.

    The P-model computes photosynthesis from light, CO2, vapour pressure deficit and the water available to the plant.
    It reports gross primary productivity, light use efficiency and intrinsic water use efficiency.

    SGAM takes that carbon and allocates it to leaf, stem and root pools, and puts what the plant sheds into a litter pool.

    RothC decomposes soil carbon.
    Its input is the carbon arriving from litter; its outputs are the soil organic carbon pools and heterotrophic respiration.

    Read down that list and the chain is obvious: water limits photosynthesis, photosynthesis feeds allocation, allocation feeds the soil.
    Read it again looking at the *names* and the units, and the chain stops being obvious, because SPLASH does not produce anything called what the P-model asks for.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The configuration

    Four `[<model>]` sections activate the modules.
    Everything else in the file exists to connect them:

    - `[[node]]` defines a new node from an expression over existing ones, which is how a bridge between two models is written.
    - `[[resample]]` changes the frequency of a variable, which is how a daily model feeds a weekly one.
    - `[inputs.*]` and `[outputs.*]` say what is read from and written to disk.

    The comments are worth reading; each bridge records why its conversion factor is what it is.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import textwrap

    config_toml = textwrap.dedent("""\
    [splash]

    [pmodel]
    method_kphio = "sandoval"
    method_optchi = "lavergne20_c3"

    [sgam]

    [rothc]
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
    """)

    mo.md("```toml\n" + config_toml + "```")
    return (config_toml,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Building the driver

    Parsing the config gives a `ParsedConfig`, and `build_driver` turns that into a Hamilton driver.
    Building is also when conduit checks the whole graph's contracts, comparing the units, dimensions and temporal frequency each node declares against what its dependencies actually supply.
    A mis-wired pipeline therefore fails here, in under a second, rather than part-way through a long run.
    That check is what makes the bridges below safe to get wrong: a missing unit conversion is a build error, not a quietly incorrect carbon budget.

    We generate synthetic inputs into a temporary directory rather than reading real data, so the notebook runs anywhere.
    """)
    return


@app.cell
def _(Config, config_toml, tomllib):
    parsed_config = Config(tomllib.loads(config_toml)).parse()
    parsed_config
    return (parsed_config,)


@app.cell
def _(Path, generate_synthetic_data, parsed_config, tempfile):
    _tmpdir = Path(tempfile.mkdtemp())

    parsed_config.input_specs["daily"].path = str(_tmpdir / "daily.nc")
    parsed_config.input_specs["weekly"].path = str(_tmpdir / "weekly.nc")
    parsed_config.input_specs["static"].path = str(_tmpdir / "static.nc")

    # conduit will not create a missing output directory; it fails rather than
    # guess. The config asks for `results/*.nc`, so make `results/` first.
    (_tmpdir / "results").mkdir()
    for _label in ("daily", "weekly", "monthly"):
        parsed_config.output_specs[_label].path = str(
            _tmpdir / "results" / f"{_label}.nc"
        )

    generate_synthetic_data(config=parsed_config, grid=(4, 4), n_days=730, seed=42)

    # The same object under a second name. `run` reads the input files off the
    # config, so it must not run until this cell has redirected the paths and
    # written the data; taking `pipeline_config` is what puts that edge in the
    # graph.
    pipeline_config = parsed_config
    return (pipeline_config,)


@app.cell
def _(build_driver, parsed_config):
    dr = build_driver(
        modules=parsed_config.modules,
        config=parsed_config.driver_config,
        node_specs=parsed_config.node_specs,
    )
    return (dr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Looking at one join at a time

    The whole DAG runs to over a hundred nodes, which is far too many to look at.
    `visualize_path_between` draws only the sub-graph connecting two nodes, and each model contributes a node under its own name, so `show_path("pmodel", "sgam")` draws exactly the wiring between those two models.

    Three settings do the tidying:

    - `strict_path_visualization=True` keeps the nodes on the path and drops their unrelated parents.
    - Hamilton draws every driver-config value as a floating `shape=note` box. They are unconnected to anything, so we filter them out of the rendered source.
    - The style we pass is *merged* into Hamilton's own defaults rather than replacing them. Hamilton already does this for nested attribute dicts such as `graph_attr`, and `show_path` does the same for its per-call arguments, so changing the orientation of one graph does not silently discard the shared style.
    """)
    return


@app.cell
def _(dr, node_maps, relabel_with_units):
    #: Shared appearance for every sub-graph drawn below.
    GRAPH_STYLE = {
        "graph_attr": {"rankdir": "LR", "ranksep": "0.5", "bgcolor": "transparent"},
        "node_attr": {"fontsize": "11"},
    }

    def show_path(upstream, downstream, **graph_attr):
        """Draw the sub-DAG connecting two nodes, using the shared style.

        Keyword arguments update ``GRAPH_STYLE["graph_attr"]`` rather than
        replacing it, and Hamilton in turn merges the result into its own
        defaults, so each level adds to the one below it.
        """
        style = {key: dict(value) for key, value in GRAPH_STYLE.items()}
        style["graph_attr"].update(graph_attr)

        graph = dr.visualize_path_between(
            upstream,
            downstream,
            output_file_path=None,
            strict_path_visualization=True,
            show_legend=False,
            graphviz_kwargs=style,
        )
        unit_map, _ = node_maps(dr)
        relabel_with_units(graph, unit_map)
        # Driver-config values are drawn as unconnected notes; drop them.
        graph.body = [line for line in graph.body if "shape=note" not in line]
        return graph

    return (show_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SPLASH to the P-model: water

    Two quantities cross, and neither crosses unchanged.

    SPLASH tracks soil moisture as a depth of water held in a bucket, in mm.
    The P-model wants volumetric water content, in m3 m-3.
    `volumetric_water_content_weekly` divides by the bucket's capacity to get relative saturation, then multiplies by 0.45, a representative porosity for mineral soil.
    The frequency changes too: SPLASH runs daily and the P-model weekly, so a `[[resample]]` sits in between.

    The aridity index is a different shape of problem.
    pyrealm's is climatological, potential evapotranspiration over precipitation with both terms accumulated across the whole record, so it is one number per pixel rather than a time series.
    SPLASH computes PET on its way to actual evapotranspiration and exposes it, so the bridge is a sum and a divide.
    """)
    return


@app.cell
def _(show_path):
    show_path("splash", "pmodel")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The P-model to SGAM: carbon, and the unit conversions

    This is the join worth studying.
    Three quantities cross and two of them are in the wrong units, because `satterc.models.pmodel` is an honest wrapper: it reports what pyrealm computes, in pyrealm's units, rather than quietly converting on the way out.

    **Gross primary productivity.**
    pyrealm reports an instantaneous flux in ug C m-2 s-1.
    SGAM wants a daily rate in g C m-2 d-1.
    That is 86400 s d-1 multiplied by 1e-6 g ug-1, so the factor is 0.0864.

    **Light use efficiency.**
    pyrealm defines LUE against PPFD, which is a *photon* flux, so the denominator is moles of photons.
    SGAM wants carbon per MJ of absorbed PAR.
    The conversion is 4.57 mol MJ-1, the standard photon content of PAR over the 400-700 nm band.

    **Intrinsic water use efficiency** needs no bridge at all.
    pyrealm reports umol mol-1 and SGAM consumes umol mol-1, so it goes straight across, which is what the direct `pmodel` to `sgam` edge in the graph below is.

    Look at the node names in the graph: the P-model produces `gpp_flux_weekly` and `lue_photon_weekly`, while SGAM consumes `gpp_weekly` and `lue_weekly`.
    The unconverted quantities are deliberately named so that they cannot satisfy SGAM's inputs by accident.
    Wiring them together directly would take an edit, and the contract check would reject the units even then.
    """)
    return


@app.cell
def _(show_path):
    show_path("pmodel", "sgam")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SGAM to RothC: from a stock to a flux

    One quantity crosses, and converting it is more than arithmetic.

    SGAM's `litter_pool` accumulates and never decreases, because decomposition is RothC's job rather than SGAM's.
    So the carbon entering the soil during a month is the pool's *increment*, not its level: difference the weekly pool, then sum those differences within each month.
    Taking the increment also picks up litter from disturbance events, which the individual turnover outputs leave out.

    Only then is there a unit conversion, from SGAM's g m-2 to RothC's t ha-1.
    The bridge does it with pint rather than writing the factor of 100 by hand, so the declared units carry the conversion.
    """)
    return


@app.cell
def _(show_path):
    show_path("sgam", "rothc")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SPLASH to RothC, and the whole chain at once

    The path from the first model to the last passes through both of the others, so this graph is the pipeline end to end.
    Drawn left to right like the others it comes out 2020 pixels wide and unreadable in a page column, so this one is drawn top to bottom instead.
    The per-call `rankdir` overrides the shared style's without discarding the rest of it.

    Two more bridges appear here, and both are about aggregating a daily rate into a monthly total.
    Summing a rate in mm d-1 over a month integrates it, because the daily timestep is one day, and `units = "mm"` relabels the result as the total it now is.

    The choice of *which* water flux to send RothC matters more than the arithmetic.
    It is potential evapotranspiration, not actual.
    RothC's water balance is rainfall minus evaporative demand, and actual evapotranspiration has already been suppressed by exactly the dryness RothC is trying to compute, so feeding it in would double-count the limitation and hold the modelled soil too wet.
    """)
    return


@app.cell
def _(show_path):
    show_path("splash", "rothc", rankdir="TB")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running it end to end

    `run` takes the config and does the rest: loads the inputs, builds the DAG, computes every variable the `[outputs.*]` sections name, and writes the three NetCDF files.

    It hands back a `RunReport`.
    `report.outputs` is one Dataset per output section, keyed by section name, and that is what the plots below read — so the files on disk are incidental here.
    `report.written` records where each one went.
    """)
    return


@app.cell
def _(pipeline_config, run):
    report = run(pipeline_config)
    outputs = report.outputs

    {written.label: str(written.path) for written in report.written}
    return (outputs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Diagnostics

    One panel per model, averaged over the sixteen pixels, in the order the carbon travels.
    Each panel runs at its model's own frequency: SPLASH daily, the P-model and SGAM weekly, RothC monthly.

    What to look for is whether the chain behaves: soil moisture should cycle seasonally, productivity should track it, the plant pools should accumulate, and the soil pools should respond to litter arriving from above.
    """)
    return


@app.cell
def _(outputs, plt):
    _daily = outputs["daily"].mean(("y", "x"))
    _weekly = outputs["weekly"].mean(("y", "x"))
    _monthly = outputs["monthly"].mean(("y", "x"))

    _fig, _axes = plt.subplots(4, 1, figsize=(10, 12))

    _daily["soil_moisture"].plot(ax=_axes[0], color="tab:blue")
    _axes[0].set_title("SPLASH: soil moisture")

    _weekly["gpp"].plot(ax=_axes[1], color="tab:green")
    _axes[1].set_title("P-model: gross primary productivity")

    for _pool in ["leaf_pool", "stem_pool", "root_pool", "litter_pool"]:
        _weekly[_pool].plot(ax=_axes[2], label=_pool)
    _axes[2].set_title("SGAM: plant carbon pools")
    _axes[2].legend(ncols=4, fontsize="small")

    _monthly["soil_organic_carbon"].plot(ax=_axes[3], color="tab:brown")
    _axes[3].set_title("RothC: soil organic carbon (bars: heterotrophic respiration)")

    _twin = _axes[3].twinx()
    _twin.bar(
        _monthly["time"],
        _monthly["heterotrophic_respiration"],
        width=20,
        color="tab:orange",
        alpha=0.35,
    )
    _twin.set_ylabel("heterotrophic respiration [t ha-1]")

    for _ax in _axes:
        _ax.set_xlabel("")

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where the joins live, and why

    Every conversion in this notebook is a `[[node]]` in the config rather than a line of Python inside a model wrapper.
    That is a deliberate choice and it costs something: the config is long, and an expression in a TOML string gets no help from a type checker.

    What it buys is that the factor of 0.0864 is visible to whoever reads the pipeline.
    A wrapper that quietly converted pyrealm's GPP on the way out would be shorter and would leave the next person with no way to see the assumption, let alone change it without editing SatTerC.
    """)
    return


if __name__ == "__main__":
    app.run()
