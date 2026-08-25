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

## Installation

See the [installation guide](getting_started/installation.md).

## Quickstart

The [quickstart](getting_started/quickstart.md) runs a one-model pipeline on
synthetic data, from an empty directory to a NetCDF file of results.

## Learn more

### Here

- [Quickstart](getting_started/quickstart.md) — run your first pipeline
- [Models](models/index.md) — the four model modules, their inputs and outputs
- [Examples](examples/my_first_pipeline.md) — interactive notebooks
  (run `just export-all` to regenerate)

### In conduit's documentation

- [Configuration reference][conduit-config] — every config section
- [Bring your own module][conduit-byom] — the conventions a model module follows
- [Contracts][conduit-contracts] — what the units/schema/frequency checks catch
- [Run and visualise][conduit-run] — the `run` and `graph` commands
- [Scale up][conduit-scale] — caching, blocking, and parallel subset runs

## Acknowledgements

### Funding

<!-- TODO: add funding bodies and grant numbers. Example:

- **[Funding Body Name]** — Grant number XXXXXXX
- **[Institution Name]** — Project title, grant period

-->

### Contributors

<!-- TODO: add contributors. Example:

- **Joe Marsh Rossney** — Lead developer
- **Name** — Role/contribution

-->

### Software dependencies

SatTerC builds on the following open-source projects:

- [conduit][conduit] — the pipeline framework: config, DAG, contracts, I/O
- [Hamilton](https://github.com/dagworks-inc/hamilton) — DAG-based dataflow framework
- [pyrealm](https://github.com/ImperialCollegeLondon/pyrealm) — SPLASH and P-Model implementations
- [RothC-Py](https://github.com/Rothamsted-Models/RothC_Py) — Rothamsted Carbon Model
- [xarray](https://docs.xarray.dev/) — N-D labeled arrays and datasets
- [Typer](https://typer.tiangolo.com/) — CLI framework

[pyrealm]: https://github.com/ImperialCollegeLondon/pyrealm
[conduit]: https://github.com/NERC-CEH/conduit
[conduit-docs]: https://github.com/NERC-CEH/conduit/tree/develop/docs
[conduit-config]: https://github.com/NERC-CEH/conduit/blob/develop/docs/reference/configuration.md
[conduit-byom]: https://github.com/NERC-CEH/conduit/blob/develop/docs/guides/bring-your-own-module.md
[conduit-contracts]: https://github.com/NERC-CEH/conduit/blob/develop/docs/concepts/contracts.md
[conduit-run]: https://github.com/NERC-CEH/conduit/blob/develop/docs/guides/run-and-visualise.md
[conduit-scale]: https://github.com/NERC-CEH/conduit/blob/develop/docs/guides/scale-up.md
