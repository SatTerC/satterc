# AGENTS.md

SatTerC is in **alpha** with no users outside the core collaboration. Backwards
compatibility is *not* a constraint: prefer the cleanest design and make breaking
changes (config schema, public APIs, behaviour) freely rather than adding
compatibility shims.

## Architecture / what lives where

SatTerC is a thin domain package on top of
[**conduit**](https://github.com/NERC-CEH/conduit), which owns the framework:
config parsing, the DAG, contract validation, I/O, caching, blocking, subsetting,
the gridded/Zarr layer, and the `run` / `graph` / `gridded` CLI commands.
conduit's **`develop` branch is the source of truth** — read it (or the local
checkout, if present) rather than guessing at its API.

SatTerC contains only:

- `satterc/models/` — the four model wrappers (SPLASH, P-model, SGAM, RothC),
  plus `_time.py` holding the frequency conventions they declare.
- `satterc/setup_utils/` — the config generator and the synthetic-data generator.
- `satterc/cli/` — `setup`, `data-gen` and `version`; conduit's sub-apps are
  mounted alongside them in `cli/__init__.py`.

## Gotchas

- All common tasks run via `just` (see `justfile`) — don't invoke `pytest`/`uv run` directly.
- Pre-commit hooks only run `uv-lock`, `pyright`, and `ruff` — not the full test suite.
- Marimo example notebooks pin `satterc==<version>` in inline `# dependencies` — update when bumping the package version, then re-export with `just export-all`.
- Plain config files in `examples/` (`config.toml`, `graphviz.toml`) are **not** loaded by any code or tooling — they are user-facing references only, so nothing will catch a mistake in them. Check with `satterc run --dry-run`.
- Documentation uses **zensical** (markdown, mkdocstrings-material-like), **not** Sphinx/rst.
- Generic behaviour belongs upstream. If a change would be useful to a pipeline
  that is not about carbon, it probably belongs in conduit, not here.

## Config schema

The schema is conduit's; there is no satterc-specific config code left. Points
that regularly trip people up:

- Models are ordinary external-module sections addressed by `_import_path`
  (`[rothc]` + `_import_path = "satterc.models.rothc"`). Nested `[models.rothc]`
  does not parse.
- Section labels are **inert** — no frequency is inferred from `daily` /
  `weekly` / `monthly`. Node names are `{var}{suffix}`, with the suffix
  defaulting to `_<label>`. `[inputs.static]` must set `suffix = ""`, because
  static variables are consumed under bare names.
- `[[resample]]` takes `from` / `to` (node-name suffixes) plus a **required**
  pandas `freq`. There is no direction table and no inferred offset.
- There is no `[grid]` (gridding is detected from the inputs' CRS), no
  `[graphviz]` (styling is a `graph --style` file), and no `[units]` — contract
  policy is `[annotations]`.
- `[subset]` uses `start` / `stop`, not `pixel_start` / `pixel_end`.

## Model modules

- There are no `dates_daily` / `dates_weekly` / `dates_monthly` nodes. A model
  that needs a calendar reads it off one of its own time-bearing inputs via
  `satterc.models._time.time_index`. This is why RothC's bridge nodes
  (`plant_cover_monthly` and friends), which otherwise consume only static data,
  take `temperature_monthly`.
- Frequencies are declared, not named: `Freq` markers from
  `satterc.models._time` (`DAILY`, `WEEKLY`, `MONTHLY`) on the signatures. They
  drive conduit's build-time contract check *and* the frequency clustering in
  `satterc graph`, so a model gaining a new frequency-suffixed input needs one.
- Decoration order follows conduit's: `declare_units` outermost, since it is the
  only decorator that converts.

## Block size is not numerically free

`[blocking]` and `[subset]` partition the `pixel` dimension, and SPLASH's results
depend slightly on that partition: `estimate_initial_soil_moisture` iterates to a
convergence tolerance evaluated over the whole block, so a different block size
stops at a different iteration. The effect is ~1e-4 relative on `soil_moisture`
and `actual_evapotranspiration`, and propagates downstream to RothC's pools.
Anything not fed by SPLASH (SGAM, the P-model) is unaffected and matches exactly.

This is long-standing behaviour of the model wrapper, not of conduit's blocking —
it reproduces to the digit on satterc v0.5.0. The consequences:

- Don't expect bit-identical output when comparing runs with different
  `block_size`, different `[subset]` ranges, or blocked against unblocked. Do
  expect it for the same partition.
- Regression tests that compare against stored reference output must pin the
  partition.
- A published result should record its block/subset layout, since that layout is
  part of what produced the numbers.
