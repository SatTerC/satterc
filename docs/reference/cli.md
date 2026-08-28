---
title: CLI
icon: lucide/square-terminal
---

# CLI

The `satterc` command is mostly [conduit][conduit-cli]'s.
`run`, `graph` and `gridded` are mounted unchanged, so that a pipeline is driven from one entry point rather than two.
They are documented below as they appear under `satterc`, but the behaviour is conduit's and so is the explanation of it: see [conduit's CLI reference][conduit-cli] for what the options mean, and [scale up][conduit-scale] for the workflow `gridded` belongs to.

Only `setup` and `data-gen` are satterc's own.
Both exist to get a pipeline running before there is any real data to run it on: `setup` writes the config, `data-gen` fills the input files it names.

## `satterc setup`

::: mkdocs-typer2
    :module: satterc.cli
    :name: satterc
    :command: setup
    :termynal: true
    :width: 80
    :prompt: ❯

## `satterc data-gen`

::: mkdocs-typer2
    :module: satterc.cli
    :name: satterc
    :command: data-gen
    :termynal: true
    :width: 80
    :prompt: ❯

## `satterc run`

::: mkdocs-typer2
    :module: satterc.cli
    :name: satterc
    :command: run
    :termynal: true
    :width: 80
    :prompt: ❯

## `satterc graph`

::: mkdocs-typer2
    :module: satterc.cli
    :name: satterc
    :command: graph
    :termynal: true
    :width: 80
    :prompt: ❯

## `satterc gridded`

Parallel-Zarr commands for gridded pipelines, needing the `geo` extra.
They bracket a set of independent `[subset]` runs: create the shared store once, run the subsets concurrently, then stitch the parts back together.

### `satterc gridded create-store`

::: mkdocs-typer2
    :module: satterc.cli
    :name: satterc
    :command: gridded create-store
    :termynal: true
    :width: 80
    :prompt: ❯

### `satterc gridded merge`

::: mkdocs-typer2
    :module: satterc.cli
    :name: satterc
    :command: gridded merge
    :termynal: true
    :width: 80
    :prompt: ❯

[conduit-cli]: https://nerc-ceh.github.io/conduit/reference/cli.html
[conduit-scale]: https://nerc-ceh.github.io/conduit/guides/run/scale-up.html
