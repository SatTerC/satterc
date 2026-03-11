# `satterc`

This is a work in progress - expect **very** sharp edges.

## Developer instructions

This project uses **[uv](https://docs.astral.sh/uv/)** for dependency management and packaging.

### Prerequisites

* Python 3.13
* `uv` installed (see [docs](https://docs.astral.sh/uv/getting-started/installation/))

### Setup for Development

1. **Clone the repository:**

```bash
git clone https://github.com/SatTerC/satterc.git
cd satterc
```


2. **Create a virtual environment and install dependencies:**

```bash
uv sync
```


3. **Activate the environment:**

```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

(Or prefix all commands with `uv run`.)


### Pre-commit actions

1. **Run the tests:**

```bash
uv run pytest
```

2. **After making changes, run `ruff`:**

```bash
ruff format
ruff check
```

### Building the docs

Build the docs with 

```bash
zensical build
```

Next, open `site/index.html` in your browser.

See [zensical.org](https://zensical.org/) for more details.


### Useful short-cuts

The awesome [`just`](https://github.com/casey/just) is a development dependency that will be installed when you run `uv sync`.

You can run the following commands anywhere in the repository:

```bash
just test  # run the tests (pytest)
just lint  # run the formatter and linter (ruff)
just docs  # build the docs (zensical)
```

## CLI use

Installing `satterc` will installing the `satterc` command.
You can explore the documentation using the `-h` or `--help` flags, e.g.

```bash
satterc -h  # help for the base command
satterc graph -h  # help for the 'graph' subcommand
```

### Generate a visualisation of the DAG

```bash
satterc graph config.toml --pdf  # or --png
```

> [!NOTE]
> This requires graphviz to be installed. E.g. `sudo apt install graphviz` (Ubuntu) or `brew install graphviz` (MacOS).

### Run

```bash
mkdir outputs
satterc run config.toml
```

This will produce three netcdf files in `outputs/`, for daily, weekly and monthly output data.

## Building config files

To do.

## Use in Python sessions and notebooks

To do.


