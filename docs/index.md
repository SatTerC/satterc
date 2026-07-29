---
title: Home
icon: lucide/house
---

# SatTerC: Satellite to Terrestrial Carbon

Carbon-cycle model modules for [conduit][conduit] pipelines.

## What is SatTerC?

SatTerC packages four terrestrial carbon models — SPLASH, the P-model, SGAM and
RothC — as modules you can compose into a pipeline described by a TOML config
file. Each is an ordinary Python module of plain functions whose names are DAG
node names, with units and frequency contracts declared on their signatures.

The framework underneath is **conduit**, which owns everything generic: parsing
the config, building the DAG, validating contracts across the whole graph before
anything runs, loading and saving data, caching, blocking, and spatial
subsetting. SatTerC adds the science, plus two conveniences — a config generator
and a synthetic-data generator — for getting started without real data.

That split is worth keeping in mind while reading these docs: **for anything
about the config schema, the CLI, data formats, or how the DAG works, go to
[conduit's documentation][conduit-docs].** What is documented here is what
SatTerC itself provides.

Key features:

- **Composable models** — SPLASH, P-Model, SGAM and RothC can be mixed and matched
- **Declared contracts** — units, dimensions and temporal frequency are checked
  across the whole graph before any compute
- **Scaffolding** — `satterc setup` writes a starting config; `satterc data-gen`
  fills it with synthetic data
- **Extensible** — add your own models as Python modules

## Installation

See the [Installation guide](getting_started/installation.md).

## Quick Start

Get a pipeline running in five minutes — see the
[Quickstart guide](getting_started/quickstart.md).

## Learn More

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

SatTerC is supported by the following grants and institutions:

<!-- Add your funding bodies here. Example:

- **[Funding Body Name]** — Grant number XXXXXXX
- **[Institution Name]** — Project title, grant period

-->

### Contributors

The following people have contributed to the development of SatTerC:

<!-- Add contributors here. Example:

- **Joe Marsh Rossney** — Lead developer
- **Name** — Role/contribution
- **Name** — Role/contribution

-->

### Software Dependencies

SatTerC builds on the following open-source projects:

- [conduit][conduit] — the pipeline framework: config, DAG, contracts, I/O
- [Hamilton](https://github.com/dagworks-inc/hamilton) — DAG-based dataflow framework
- [pyrealm](https://github.com/ImperialCollegeLondon/pyrealm) — SPLASH and P-Model implementations
- [RothC-Py](https://github.com/Rothamsted-Models/RothC_Py) — Rothamsted Carbon Model
- [xarray](https://docs.xarray.dev/) — N-D labeled arrays and datasets
- [Typer](https://typer.tiangolo.com/) — CLI framework

[conduit]: https://github.com/NERC-CEH/conduit
[conduit-docs]: https://github.com/NERC-CEH/conduit/tree/develop/docs
[conduit-config]: https://github.com/NERC-CEH/conduit/blob/develop/docs/reference/configuration.md
[conduit-byom]: https://github.com/NERC-CEH/conduit/blob/develop/docs/guides/bring-your-own-module.md
[conduit-contracts]: https://github.com/NERC-CEH/conduit/blob/develop/docs/concepts/contracts.md
[conduit-run]: https://github.com/NERC-CEH/conduit/blob/develop/docs/guides/run-and-visualise.md
[conduit-scale]: https://github.com/NERC-CEH/conduit/blob/develop/docs/guides/scale-up.md
