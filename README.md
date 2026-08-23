# `satterc`

> [!WARNING]
> **Pre-alpha. Not ready for use.**
> SatTerC is an unfinished research code with no users outside the core collaboration.
> Results are unvalidated, large parts are untested against anything real, and the models, config schema and CLI change without notice or deprecation.
> These docs are a skeleton to be filled in, so treat gaps as gaps rather than as things that work and went undocumented.
> If you want to use this for something, get in touch first.

Carbon-cycle model modules (SPLASH, the P-model, SGAM and RothC) for [conduit](https://github.com/NERC-CEH/conduit) pipelines.

Usage instructions are in the [documentation](https://SatTerC.github.io/satterc), which is at the same stage as the code.

## Developer instructions

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and packaging.

### Prerequisites

* Python 3.13
* `uv` installed (see [docs](https://docs.astral.sh/uv/getting-started/installation/))

### Setup for development

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


### Pre-commit hooks

This project uses [pre-commit](https://pre-commit.com/) to run linting and tests automatically before each commit.

**First-time setup:**

```bash
uv run pre-commit install
```

After this, `just lint` and `just test` will run automatically before every `git commit`.

To run hooks manually without committing:

```bash
uv run pre-commit run --all-files
```

### Building the docs

Build the docs with

```bash
zensical build
```

Next, open `site/index.html` in your browser.

See [zensical.org](https://zensical.org/) for more details.


### Useful short-cuts

[`just`](https://github.com/casey/just) is a development dependency, installed when you run `uv sync`.

You can run the following commands anywhere in the repository:

```bash
just test        # run the test suite (pytest)
just lint        # format and lint code with ruff, check examples with marimo
just docs        # build the docs (zensical)
just export <x>  # export a notebook example to docs (e.g. just export my_first_pipeline)
just export-all  # export all example notebooks
```

## CLI use

Installing `satterc` installs the `satterc` command.
You can explore the documentation using the `-h` or `--help` flags, e.g.

```bash
satterc -h  # help for the base command
satterc graph -h  # help for the 'graph' subcommand
```

`run`, `graph` and `gridded` are conduit's commands, mounted into the `satterc` app so there is a single entry point.
`setup`, `data-gen` and `version` are satterc-specific.

### Generate a visualisation of the DAG

```bash
satterc graph config.toml --pdf  # or --png
```

> [!NOTE]
> This requires graphviz to be installed. E.g. `sudo apt install graphviz` (Ubuntu) or `brew install graphviz` (MacOS).

### Run

```bash
mkdir outputs
satterc run config.toml --dry-run  # validate config, inputs and contracts first
satterc run config.toml
```

This produces three netcdf files in `outputs/`, for daily, weekly and monthly output data.
