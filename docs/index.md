---
title: Home
icon: lucide/house
---

# SatTerC: Satellite to Terrestrial Carbon

Composable models of the terrestrial carbon and water cycles, for
[conduit][conduit] pipelines.

/// admonition | Pre-alpha: not ready for use
    type: warning

SatTerC is an unfinished research code with no users outside the core
collaboration. Results are unvalidated, large parts are untested against
anything real, and the models, config schema and CLI change without notice or
deprecation. So, probably, will the name.
///

## What is SatTerC?

SatTerC packages models of the terrestrial carbon and water cycles as modules
you can compose into a pipeline described by a TOML config file. Four are
implemented so far: SPLASH, the P-model, SGAM and RothC. Each is an ordinary
Python module of plain functions whose names are DAG node names, with units and
frequency contracts declared on their signatures.

The framework underneath is conduit, which owns everything generic: parsing the
config, building the DAG, validating contracts across the whole graph before
anything runs, loading and saving data, caching, blocking, and spatial
subsetting. SatTerC adds the science, plus a config generator and a
synthetic-data generator for getting started without real data.

Keep that split in mind while reading these docs. **For anything about the
config schema, the CLI, data formats, or how the DAG works, go to [conduit's
documentation][conduit-docs].** What is documented here is what SatTerC itself
provides.

## Where this is going

Four models is a starting point, not the intended scope. Most of what we want
next is the rest of what [pyrealm][pyrealm] implements: subdaily and two-leaf
variants of the P-model, C3/C4 competition, carbon isotopes, phenology, and the
T-model for tree growth and canopy structure. Others will come from elsewhere,
or be written here as SGAM was. Hydrology beyond SPLASH's bucket is on the list
too.

Alongside those, the ecosystem carbon models of the DALEC and CARDAMOM family.
They are satellite-driven and built from coupled carbon pools, which is close to
what SatTerC already does: SGAM and RothC are pool models, and they run off the
same kind of drivers.

Further out, the version we are aiming at lets you assemble a land surface model
of the kind CLM or JULES provides one process at a time, choosing the
representation of each rather than taking a whole model as given. That is a long
way off. What exists today is four models and the machinery to wire them
together.

## Navigating these docs

<div class="grid cards" markdown>

- **[Guides](guides/installation.md)** — how-to guides. Start with [installation](guides/installation.md), then the [quickstart](guides/quickstart.md), which goes from an empty directory to a NetCDF file of results without needing any data of your own.
- **[Models](models/index.md)** — a page per model: [SPLASH](models/splash.md) for the water balance, the [P-model](models/pmodel.md) for photosynthesis, [SGAM](models/sgam.md) for vegetation carbon pools, [RothC](models/rothc.md) for soil carbon. Each one gives the DAG, the theory, and tables of inputs, outputs and settings generated from the code.
- **[Recipes](recipes/my_first_pipeline.md)** — complete pipelines as executable [marimo](https://marimo.io) notebooks. [My first pipeline](recipes/my_first_pipeline.md) is the step-by-step introduction; [soil moisture](recipes/soil_moisture.md) and [PFT parameters](recipes/pft_parameters.md) calibrate parameters against observations; [full pipeline](recipes/full_pipeline.md) chains all four models from Python rather than the CLI.
- **[Reference](reference/index.md)** — the [CLI](reference/cli.md) commands that are satterc's own, the [Python API](reference/python-api.md), and every module's signatures. Config schema and data formats are conduit's, and the [overview](reference/index.md) says where to find them.

</div>

## See also

**Upstream**

SatTerC is a thin layer over other people's work:

- [conduit][conduit] — the pipeline framework: config, DAG, contracts, I/O.
- [Hamilton](https://github.com/dagworks-inc/hamilton) — the DAG engine underneath conduit.
- [pyrealm][pyrealm] — the SPLASH and P-model implementations.
- [RothC-Py](https://github.com/Rothamsted-Models/RothC_Py) — the Rothamsted carbon model.
- [xarray](https://docs.xarray.dev/) — labelled N-D arrays.
- [Typer](https://typer.tiangolo.com/) — the CLI.

SGAM is the exception: it is implemented here.

## Acknowledgements

<!-- TODO: add funding bodies and grant numbers, and contributors. Example:

This work has been supported by:

- NC-International

-->

[pyrealm]: https://github.com/ImperialCollegeLondon/pyrealm
[conduit]: https://github.com/NERC-CEH/conduit
[conduit-docs]: https://github.com/NERC-CEH/conduit/tree/main/docs
