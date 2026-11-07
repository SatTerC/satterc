---
title: Overview
icon: lucide/book-marked
---

# Reference

Two halves of a satterc pipeline are documented in two places, and which one you need depends on what you are looking up.

Anything generic belongs to catena: the config schema, the file formats, the DAG, the contract checks, the `run` and `graph` commands.
Anything about the carbon and water cycle models, the config generator, or the synthetic-data generator is satterc's, and is documented here.

## Here

<div class="grid cards" markdown>

- **[CLI](cli.md)** — `satterc setup`, which writes a starting config, and `satterc data-gen`, which fills its input files with synthetic data. The other subcommands are catena's, mounted unchanged.
- **[Python API](python-api.md)** — the model modules, the scaffolding behind `setup` and `data-gen`, and every module's signatures.

</div>

## In catena's documentation

<div class="grid cards" markdown>

- **[Configuration][catena-config]** — every TOML section and key: inputs, outputs, nodes, blocking, subsetting, caching.
- **[Data formats][catena-formats]** — which file formats can be read and written, and what each one supports.
- **[Python API][catena-api]** and **[CLI][catena-cli]** — building and running the pipeline itself.

</div>

[catena-config]: https://nerc-ceh.github.io/catena/reference/configuration.html
[catena-formats]: https://nerc-ceh.github.io/catena/reference/data-formats.html
[catena-api]: https://nerc-ceh.github.io/catena/reference/python-api.html
[catena-cli]: https://nerc-ceh.github.io/catena/reference/cli.html
