---
title: Installation
icon: lucide/download
---

# Installation

SatTerC is currently only available from GitHub.

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

This installs the `satterc` package and the `satterc` CLI command into your environment.

## Install for development

```sh
git clone https://github.com/SatTerC/satterc.git
cd satterc
uv sync
source .venv/bin/activate
```

## Optional dependencies

### Graphviz (for pipeline visualization)

Required for `satterc graph`:

```sh
# Ubuntu/Debian
sudo apt install graphviz

# macOS
brew install graphviz
```

### PyYAML (for YAML static inputs)

Required if you use `.yaml` or `.yml` files for static inputs:

```sh
uv add pyyaml
```

## Verify installation

```sh
satterc --version
```
