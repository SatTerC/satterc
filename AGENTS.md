# AGENTS.md

SatTerC is in **alpha** with no users outside the core collaboration. Backwards
compatibility is *not* a constraint: prefer the cleanest design and make breaking
changes (config schema, public APIs, behaviour) freely rather than adding
compatibility shims.

## Gotchas

- All common tasks run via `just` (see `justfile`) — don't invoke `pytest`/`uv run` directly.
- Pre-commit hooks only run `uv-lock`, `pyright`, and `ruff` — not the full test suite.
- Marimo example notebooks pin `satterc==<version>` in inline `# dependencies` — update when bumping the package version, then re-export with `just export-all`.
- Plain config files in `examples/` (`config.toml`, `graphviz.toml`) are **not** loaded by any code or tooling — they are user-facing references only.
- Documentation uses **zensical** (markdown, mkdocstrings-material-like), **not** Sphinx/rst.
