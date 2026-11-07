---
title: CLI
icon: lucide/square-terminal
---

# CLI

The `satterc` command is mostly [catena][catena-cli]'s.
`run`, `graph` and `gridded` are mounted unchanged, so that a pipeline is driven from one entry point rather than two.
They are documented below as they appear under `satterc`, but the behaviour is catena's and so is the explanation of it: see [catena's CLI reference][catena-cli] for what the options mean, and [scale up][catena-scale] for the workflow `gridded` belongs to.

Only `setup` and `data-gen` are satterc's own.
Both exist to get a pipeline running before there is any real data to run it on: `setup` writes the config, `data-gen` fills the input files it names.

## `satterc setup`

/// admonition | Experimental
    type: warning

`setup` is experimental, and both the command and the configs it generates may change.
Treat its output as a starting point rather than a complete config.

It scaffolds the inputs for each model's default settings only.
Changing a setting can bring in inputs the generated config does not load, such as a P-model `method_*` key, and those have to be added by hand; see [P-model inputs](../models/pmodel.md#inputs).
`satterc run --dry-run` reports any that are missing.
This limitation is tracked in [#56](https://github.com/SatTerC/satterc/issues/56).
///

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

Parallel-Zarr commands for gridded pipelines.
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

[catena-cli]: https://nerc-ceh.github.io/catena/reference/cli.html
[catena-scale]: https://nerc-ceh.github.io/catena/guides/run/scale-up.html
