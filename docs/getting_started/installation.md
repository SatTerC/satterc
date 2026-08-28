---
title: Installation
icon: lucide/download
---

# Installation

SatTerC is not on PyPI and will not be until it is worth installing. Until
then, install it from GitHub.

/// admonition | Pin a commit
    type: warning

SatTerC is pre-alpha and the API changes without deprecation, so installing from
the default branch means a later reinstall can break your pipeline. Pin a commit
(`git+https://github.com/satterc/satterc@<sha>`) if you need the same behaviour
twice.
///

## Prerequisites

- **Python 3.13** or later

## Install into an existing project

=== "pip"

    ```sh
    pip install git+https://github.com/satterc/satterc
    ```

=== "uv"

    ```sh
    uv add git+https://github.com/satterc/satterc
    ```

This installs the `satterc` package and the `satterc` CLI command into your
environment, and pulls in [conduit][conduit], which provides the config parser,
the DAG, contract validation, I/O and the `run`, `graph` and `gridded` commands.
conduit's geospatial support comes with it, since SatTerC's gridded pipelines
need it.

The base install does **not** include the ecological models themselves or the
DAG visualisation support. Install those via the extras below.

## Optional features (extras)

SatTerC groups its optional dependencies into installable extras:

| Extra | Installs | Needed for |
| --- | --- | --- |
| `models` | `pyrealm`, `rothc-py`, `sgam` | the built-in P-model, SPLASH, SGAM and RothC models |
| `viz` | `conduit[viz]` | rendering the DAG with `satterc graph` |
| `all` | everything above | installing every optional feature at once |

Append the extra(s) in square brackets:

=== "pip"

    ```sh
    pip install "satterc[models] @ git+https://github.com/satterc/satterc"
    pip install "satterc[all] @ git+https://github.com/satterc/satterc"
    ```

=== "uv"

    ```sh
    uv add "satterc[models] @ git+https://github.com/satterc/satterc"
    uv add "satterc[all] @ git+https://github.com/satterc/satterc"
    ```

## Install for development

```sh
git clone https://github.com/SatTerC/satterc.git
cd satterc
uv sync
source .venv/bin/activate
```

`uv sync` installs every optional extra (`models` and `viz`) along with the
development tooling, so you don't need to request them explicitly.

## System dependencies

### Graphviz (for pipeline visualisation)

The `viz` extra installs the Python `graphviz` bindings, but `satterc graph` also
needs the Graphviz system binaries:

```sh
# Ubuntu/Debian
sudo apt install graphviz

# macOS
brew install graphviz
```

## Verify installation

```sh
satterc --version
```

This prints the installed SatTerC version and, beneath it, the version of
conduit it is running on.

[conduit]: https://github.com/NERC-CEH/conduit
