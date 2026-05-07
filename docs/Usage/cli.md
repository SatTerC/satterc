---
title: CLI Reference
icon: lucide/terminal
---

# Command-Line Interface

SatTerC provides a `satterc` command for running pipelines from the terminal.

## Overview

| Command | Description |
|---------|-------------|
| [`satterc setup`](#setup) | Generate a configuration file interactively |
| [`satterc run`](#run) | Execute a pipeline from a config file |
| [`satterc graph`](#graph) | Visualise a pipeline as a graph |
| [`satterc data-gen`](#data-gen-generate) | Generate synthetic input data for testing |

Get help for any command with `-h`:

```sh
satterc -h
satterc run -h
```

## `setup`

Generate a configuration file through interactive prompts or with defaults.

```sh
satterc setup [OPTIONS]
```

### Options

| Flag | Short | Description |
|------|-------|-------------|
| `--models` | `-m` | Space-separated list of built-in models to include (e.g., `splash pmodel`) |
| `--output` | `-o` | Output path for the generated config file (default: `config.toml`) |
| `--defaults` | `-d` | Use default input/output paths without prompting (requires `--models`) |

### Examples

Interactive mode — walks through model selection and path configuration:

```sh
satterc setup
```

Non-interactive mode with defaults:

```sh
satterc setup --models splash pmodel sgam --defaults
```

Custom output path:

```sh
satterc setup --models splash --output my_pipeline.toml
```

## `run`

Execute a pipeline defined in a configuration file.

```sh
satterc run <CONFIG_FILE> [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `CONFIG_FILE` | Path to a TOML configuration file |

### Options

| Flag | Description |
|------|-------------|
| `--allow-overrides` | Allow later modules to override earlier ones |

### Example

```sh
satterc run config.toml
```

This reads the config, builds the DAG, executes all required nodes, and writes output files as specified in the `[outputs.*]` sections.

## `graph`

Visualise a pipeline as a directed graph.

```sh
satterc graph <CONFIG_FILE> [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `CONFIG_FILE` | Path to a TOML configuration file |

### Options

| Flag | Description |
|------|-------------|
| `--output`, `-o` | Name of output file (default: `pipeline`) |
| `--allow-overrides` | Allow later modules to override earlier ones |
| `--png` | Convert to PNG format |
| `--pdf` | Convert to PDF format |

### Example

```sh
satterc graph config.toml --pdf
```

This produces `pipeline.pdf` (and `pipeline.dot`).
The graph is colour-coded by temporal frequency:

| Colour | Frequency |
|--------|-----------|
| Aquamarine | Static inputs |
| Orange | Daily |
| Yellow | Weekly |
| Brown | Monthly |

/// admonition | Note
    type: note

Requires [graphviz](https://graphviz.org/) to be installed.
///

## `data-gen generate`

Generate synthetic input data for testing pipelines.

```sh
satterc data-gen generate <CONFIG_FILE> [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `CONFIG_FILE` | Path to a TOML configuration file |

### Options

| Flag | Short | Description |
|------|-------|-------------|
| `--grid`, `-g` | Grid dimensions as `n_lat,n_lon` (default: `1,1`) |
| `--duration`, `-d` | Time duration: `2y`, `6m`, `30d` (default: `2y`) |
| `--seed`, `-s` | Random seed for reproducibility (default: `42`) |

### Examples

Generate data for a single site over 2 years:

```sh
satterc data-gen generate config.toml
```

Generate data for a 4x4 grid over 6 months:

```sh
satterc data-gen generate config.toml --grid 4,4 --duration 6m
```
