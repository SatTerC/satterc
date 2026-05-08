---
title: Home
icon: lucide/house
---

# SatTerC: Satellite to Terrestrial Carbon modelling using DAGs

A Python framework for data-driven terrestrial carbon modelling based on Directed Acyclic Graphs.

## Installation

See the [Installation guide](getting_started/installation.md) for detailed instructions.

## Quick Start

Get a pipeline running in 5 minutes — see the [Quickstart guide](getting_started/quickstart.md).

## What is SatTerC?

SatTerC lets you compose terrestrial carbon models into pipelines described by a simple TOML configuration file.
Models are connected as a Directed Acyclic Graph (DAG), meaning you declare **what** you want computed and SatTerC figures out **how** to compute it.

Key features:

- **Composable models** — SPLASH, P-Model, SGAM, and RothC can be mixed and matched
- **Automatic dependency resolution** — the DAG engine determines execution order
- **Multiple data formats** — NetCDF, Zarr, CSV, Parquet, JSON
- **CLI and Python API** — run from the terminal or embed in notebooks
- **Extensible** — add your own models as Python modules

## Learn More

- [Quickstart](getting_started/quickstart.md) — run your first pipeline
- [Concepts](getting_started/concepts.md) — how DAGs work
- [Models](models/index.md) — built-in model overview
- [Configuration](usage/config.md) — TOML config reference
- [CLI](usage/cli.md) — command-line interface
- [examples](#) — interactive notebooks (run `just export-all` to generate)

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

- [Hamilton](https://github.com/dagworks-inc/hamilton) — DAG-based dataflow framework
- [pyrealm](https://github.com/ImperialCollegeLondon/pyrealm) — SPLASH and P-Model implementations
- [RothC-Py](https://github.com/Rothamsted-Models/RothC_Py) — Rothamsted Carbon Model
- [xarray](https://docs.xarray.dev/) — N-D labeled arrays and datasets
- [Typer](https://typer.tiangolo.com/) — CLI framework
