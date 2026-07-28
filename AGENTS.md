# AGENTS.md

SatTerC is in **alpha** with no users outside the core collaboration. Backwards
compatibility is *not* a constraint: prefer the cleanest design and make breaking
changes (config schema, public APIs, behaviour) freely rather than adding
compatibility shims.

## Architecture / what lives where

SatTerC is a thin domain package on top of [**conduit**](https://github.com/NERC-CEH/conduit), which owns the framework:
config parsing, the DAG, contract validation, I/O, caching, blocking, subsetting,
the gridded/Zarr layer, and the `run` / `graph` / `gridded` CLI commands.
conduit's **`develop` branch is the source of truth** — read it (or the local
checkout, if present) rather than guessing at its API.

## Gotchas

- All common tasks run via `just` (see `justfile`) — no need to invoke `pytest`/`uv run` directly.
- Pre-commit hooks only run `uv-lock`, `pyright`, and `ruff` — not the full test suite.
- Marimo example notebooks pin `satterc==<version>` in inline `# dependencies` — update when bumping the package version, then re-export with `just export-all`.
- Plain config files in `examples/` (`config.toml`, `graphviz.toml`) are **not** loaded by any code or tooling — they are user-facing references only, so nothing will catch a mistake in them. Check with `satterc run --dry-run`.
- Documentation uses **zensical** (markdown, mkdocstrings-material-like), **not** Sphinx/rst.
- Generic behaviour belongs upstream. If a change would be useful to a pipeline
  that is not about carbon, it probably belongs in conduit, not here.

## Config schema

The schema is conduit's; there is no satterc-specific config code.


## Block size is not numerically free

`[blocking]` and `[subset]` partition the `pixel` dimension, and SPLASH's results
depend slightly on that partition: `estimate_initial_soil_moisture` iterates to a
convergence tolerance evaluated over the whole block, so a different block size
stops at a different iteration. The effect is ~1e-4 relative on `soil_moisture`
and `actual_evapotranspiration`.
Therefore, don't expect bit-identical output when comparing runs with different
`block_size`, different `[subset]` ranges, or blocked against unblocked. Do
expect it for the same partition.
