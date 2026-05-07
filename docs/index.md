---
title: Home
icon: lucide/house
---

# SatTerC: Satellite to Terrestrial Carbon modelling using DAGs

A Python framework for data-driven terrestrial carbon modelling based on Directed Acyclic Graphs.

## Installation

See the [Installation guide](Getting%20Started/installation.md) for detailed instructions.

## Quick Start

Get a pipeline running in 5 minutes — see the [Quickstart guide](Getting%20Started/quickstart.md).

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

- [Quickstart](Getting%20Started/quickstart.md) — run your first pipeline
- [Concepts](Getting%20Started/concepts.md) — how DAGs work
- [Models](Models/index.md) — built-in model overview
- [Configuration](Usage/config.md) — TOML config reference
- [CLI](Usage/cli.md) — command-line interface
- [Examples](#) — interactive notebooks (run `just export-all` to generate)
