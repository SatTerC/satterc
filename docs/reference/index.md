---
title: Overview
icon: lucide/book-marked
---

# Reference

Two halves of a satterc pipeline are documented in two places, and which one you need depends on what you are looking up.

Anything generic belongs to conduit: the config schema, the file formats, the DAG, the contract checks, the `run` and `graph` commands.
Anything about the carbon and water cycle models, the config generator, or the synthetic-data generator is satterc's, and is documented here.

## Here

<div class="grid cards" markdown>

- **[CLI](cli.md)** — `satterc setup`, which writes a starting config, and `satterc data-gen`, which fills its input files with synthetic data. The other subcommands are conduit's, mounted unchanged.
- **[Python API](python-api.md)** — the model modules, the scaffolding behind `setup` and `data-gen`, and every module's signatures.

</div>

## In conduit's documentation

<div class="grid cards" markdown>

- **[Configuration][conduit-config]** — every TOML section and key: inputs, outputs, nodes, blocking, subsetting, caching.
- **[Data formats][conduit-formats]** — which file formats can be read and written, and what each one supports.
- **[Python API][conduit-api]** and **[CLI][conduit-cli]** — building and running the pipeline itself.

</div>

[conduit-config]: https://nerc-ceh.github.io/conduit/reference/configuration.html
[conduit-formats]: https://nerc-ceh.github.io/conduit/reference/data-formats.html
[conduit-api]: https://nerc-ceh.github.io/conduit/reference/python-api.html
[conduit-cli]: https://nerc-ceh.github.io/conduit/reference/cli.html
