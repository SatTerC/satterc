# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (requires uv and Python 3.13)
uv sync --group dev

# Run tests
just test        # or: uv run pytest
uv run pytest tests/path/to/test_file.py::test_name  # single test

# Lint and format
just lint        # ruff format src/ && ruff check src/

# Run pipeline
just run         # satterc run config.toml

# Generate synthetic input data then run
just synth

# Visualize the DAG
just viz         # outputs pipeline.pdf (requires graphviz)
```

## Architecture

SatTerC is a pipeline framework for terrestrial carbon modelling, built on top of [Hamilton](https://hamilton.dagworks.io) (a DAG-based dataflow library).

### Entry points

- CLI: `satterc.cli:main` — commands: `run`, `graph`, `setup`, `data-gen`, `version`
- Core: `load_config()` → `build_driver()` → `dr.execute(targets)`

### Config → Driver → Pipeline

1. **`config.py`** parses a TOML file into `(modules, driver_config, targets)`. The TOML declares which pipeline modules to activate, file paths for inputs/outputs, and model parameters.

2. **`driver.py`** passes those modules to Hamilton's `Driver`, which introspects function signatures to build a DAG automatically. `driver_config` is passed as kwargs; `targets` are the output node names to materialise.

3. **`pipeline/`** contains the Hamilton modules — each `.py` file is a collection of functions that become DAG nodes:
   - `inputs/` — loaders for daily/weekly/monthly/static NetCDF inputs; stacks (x, y) spatial dims into a single `pixel` dimension
   - `models/` — wrappers around external Earth-system models: `splash` (water balance), `pmodel` (GPP via pyrealm), `sgam` (carbon allocation), `rothc` (soil carbon decomposition; installed from git)
   - `resample.py` — temporal aggregation (daily→weekly/monthly, weekly→monthly) using xarray
   - `outputs.py` — merges result variables, unstacks `pixel` back to (x, y), writes NetCDF

### Key utilities (`utils/`)

- **`xarray_io` decorator** (`xarray_io.py`): bridges xarray DataArrays to plain NumPy/JAX arrays so model functions can remain array-agnostic. Strips coords on input, reattaches them on output.
- **`FixedResolve` decorator** (`resolve.py`): generates Hamilton-compatible functions dynamically from config — used to inject variable-name-specific nodes at config load time.
- **`extract_fields` decorator**: unpacks dict-valued model outputs into separate named DAG nodes (Hamilton's `extract_fields` pattern).

### Data flow

```
config.toml
  → load_config()           # parse modules, driver_config, targets
  → build_driver(modules)   # Hamilton builds DAG from function signatures
  → dr.execute(targets)
      inputs: load NetCDF → stack spatial dims → extract variables
      models: SPLASH → P-Model → SGAM → RothC
      resample: daily/weekly → target frequencies
      outputs: merge → unstack spatial dims → save NetCDF
```

### Testing

Tests live in `tests/`. Fixtures in `conftest.py` generate synthetic xarray datasets at session scope. Tests check shapes, value ranges, and metadata consistency rather than exact numerical values.
