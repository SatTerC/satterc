# `satterc`

Composable models of the terrestrial carbon and water cycles, for [conduit](https://github.com/NERC-CEH/conduit) pipelines.

> [!WARNING]
> **Alpha status.** `satterc` is an early-stage project under active development. Things will change without warning.

Usage instructions are in the [documentation](https://SatTerC.github.io/satterc).

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
just lint        # format and lint code with ruff, check recipe notebooks with marimo
just docs        # build the docs (zensical)
just export <x>  # export a recipe notebook to docs (e.g. just export my_first_pipeline)
just export-all  # export all recipe notebooks
```

