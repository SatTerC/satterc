---
title: Home
icon: lucide/house
---

# SatTerC: Satellite to Terrestrial Carbon

Composable models of the terrestrial carbon and water cycles, for [conduit][conduit] pipelines.

SatTerC is an experimental research project which aims to be a flexible framework for composing models of the terrestrial carbon and water cycles, driven by Earth Observation data, into a Directed Acyclic Graph (DAG).
To achieve this, SatTerC builds on [conduit][conduit] --- an opinionated integration of [Hamilton][hamilton] (the DAG engine) and [xarray][xarray] (the array library). 

This documentation mainly focuses on the science, and the bare minimum technical know-how to assemble and run a pipeline.
If you're interested in learning more about the underlying framework, go to [conduit's documentation][conduit-docs].

!!! warning "Alpha status"

    `satterc` is an early-stage project under active development. Things will change without warning.


## Navigating these docs

<div class="grid cards" markdown>

- **[Guides](guides/installation.md)** — how-to guides. Start with [installation](guides/installation.md), then the [quickstart](guides/quickstart.md), which goes from an empty directory to a NetCDF file of results without needing any data of your own.
- **[Models](models/index.md)** — a page per model: [SPLASH](models/splash.md) for the water balance, the [P-model](models/pmodel.md) for photosynthesis, [SGAM](models/sgam.md) for vegetation carbon pools, [RothC](models/rothc.md) for soil carbon. Each one gives the DAG, the theory, and tables of inputs, outputs and settings generated from the code.
- **[Recipes](recipes/my_first_pipeline.md)** — complete pipelines as executable [marimo](https://marimo.io) notebooks. [My first pipeline](recipes/my_first_pipeline.md) is the step-by-step introduction; [soil moisture](recipes/soil_moisture.md) and [PFT parameters](recipes/pft_parameters.md) calibrate parameters against observations; [full pipeline](recipes/full_pipeline.md) chains all four models from Python rather than the CLI.
- **[Reference](reference/index.md)** — the [CLI](reference/cli.md) commands that are satterc's own, the [Python API](reference/python-api.md), and every module's signatures. Config schema and data formats are conduit's, and the [overview](reference/index.md) says where to find them.

</div>

## See also

SatTerC builds on the following excellent libraries:

- [conduit][conduit] — the pipeline framework. An opinionated integration of [Hamilton][hamilton] and [xarray][xarray].
- [pyrealm][pyrealm] — the SPLASH and P-model implementations.
- [RothC-Py](https://github.com/Rothamsted-Models/RothC_Py) — the Rothamsted carbon model.
- [Typer](https://typer.tiangolo.com/) — the CLI.

## Roadmap

This is an early prototype.
We definitely want to bring in more models, e.g. from PyRealm.
We are also interested in substituting the DAG for a state machine (directed graph that can contain cycles) for prognostic modelling.

## Acknowledgements

<!-- TODO: add funding bodies and grant numbers, and contributors. Example:

This work has been supported by:

- NC-International

-->

[pyrealm]: https://github.com/ImperialCollegeLondon/pyrealm
[conduit]: https://github.com/NERC-CEH/conduit
[conduit-docs]: https://github.com/NERC-CEH/conduit/tree/main/docs
[hamilton]: https://github.com/dagworks-inc/hamilton
[xarray]: https://docs.xarray.dev/
