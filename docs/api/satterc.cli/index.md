---
title: Overview
icon: lucide/terminal
---

# CLI Reference

The `satterc` command-line interface is mostly [conduit][conduit-cli]'s: `run`,
`graph` and `gridded` are mounted unchanged, so that a satterc user has one entry
point rather than two. See conduit's documentation for those.

Only the commands below are satterc's own, and only they are documented here.

## Commands

| Module | Description |
|--------|-------------|
| [`satterc.cli.setup`](setup.md) | Generate a configuration file interactively |
| [`satterc.cli.data_gen`](data_gen.md) | Generate synthetic input data for testing |
| [`satterc.cli.version`](version.md) | Show the installed satterc and conduit versions |

[conduit-cli]: https://github.com/NERC-CEH/conduit/blob/develop/docs/guides/run-and-visualise.md
