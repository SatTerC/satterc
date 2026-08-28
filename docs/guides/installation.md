---
title: Installation
icon: lucide/download
---

# Installation

SatTerC is not on PyPI and will not be until it is worth installing.
Until then, install it from GitHub.

/// admonition | Pin a commit
    type: warning

SatTerC is pre-alpha and the API changes without deprecation, so installing from the default branch means a later reinstall can break your pipeline.
Pin a commit (`git+https://github.com/satterc/satterc@<sha>`) if you need the same behaviour twice.
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

This installs the `satterc` package and the `satterc` CLI command into your environment, along with everything needed to run the built-in models.
That is [conduit][conduit], which provides the config parser, the DAG, contract validation, I/O and the `run`, `graph` and `gridded` commands, plus its geospatial and visualisation support, and the model libraries `pyrealm`, `rothc-py` and `sgam`.

There are no extras to choose from.
SatTerC is pre-alpha and used by a handful of people, so a single install that always works is worth more than a smaller one.

## Install for development

```sh
git clone https://github.com/SatTerC/satterc.git
cd satterc
uv sync
source .venv/bin/activate
```

`uv sync` adds the development tooling on top of the runtime dependencies.

## System dependencies

### Graphviz (for pipeline visualisation)

The Python `graphviz` bindings come with the install, but `satterc graph` also needs the Graphviz system binaries:

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

This prints the installed SatTerC version and, beneath it, the version of conduit it is running on.

[conduit]: https://github.com/NERC-CEH/conduit
