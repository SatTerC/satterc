---
title: Overview
icon: lucide/package
---

# Built-in models

SatTerC ships four models so far, which can be composed into pipelines. Each is an
ordinary [conduit module][conduit-byom], a set of plain functions whose names are
DAG node names, referenced from a config by its dotted `_import_path`:

```toml
[splash]
_import_path = "satterc.models.splash"
soil_moisture_init_max_iter = 10
```

/// admonition | What the tests cover
    type: warning

The test suite checks the wiring: that each wrapper passes the right arrays to
the underlying implementation, that units and frequencies match what is
declared, and that a pipeline runs end to end on synthetic data. It does not
check the science. No output on these pages has been compared against a
reference run or against observations.
///

| Model | Description | Temporal resolution |
|-------|-------------|---------------------|
| [SPLASH](splash.md) | Semi-empirical water-balance model that computes actual evapotranspiration, soil moisture, and runoff from daily climate data | Daily |
| [P-model](pmodel.md) | Optimality-based photosynthesis model that computes gross primary production (GPP), light-use efficiency and intrinsic water-use efficiency from environmental drivers | Weekly |
| [SGAM](sgam.md) | Simplified Growth and Allocation Model: tracks carbon pools (leaf, stem, root, litter) over time | Weekly |
| [RothC](rothc.md) | Soil carbon decomposition model that simulates the turnover of organic matter in soil, producing soil organic carbon stocks | Monthly |

## What comes next

These four cover a water balance, photosynthesis, vegetation growth and soil
carbon, which is enough for a carbon pipeline and not much else. The models we
want next are mostly the rest of [pyrealm][pyrealm]: subdaily and two-leaf
P-model variants, C3/C4 competition, carbon isotopes, phenology, and the T-model
for tree growth and canopy structure. Beyond that, the ecosystem carbon models of the
DALEC and CARDAMOM family, which are satellite-driven and pool-based and so sit
close to what SGAM and RothC already do, then a real hydrology in place of
SPLASH's bucket.

Adding a model means writing a conduit module and declaring its contracts, so
the list is limited by the work rather than by anything in the design. If you
want a particular model in here, say so on the issue tracker.

## Naming and declared frequencies

The `_daily` / `_weekly` / `_monthly` / static suffixes on node names are a
SatTerC convention, not framework behaviour. conduit treats a config section's
label as inert and infers no frequency from it. The frequency is carried by a
contract declared on each model's signature:

| Suffix | Declared offset | `satterc.temporal` |
|--------|-----------------|------------------------|
| `_daily` | `D` | `DAILY` |
| `_weekly` | `7D` (unanchored, any weekday) | `WEEKLY` |
| `_monthly` | `1ME` (month end) | `MONTHLY` |

Because those declarations are on the functions, conduit can check the whole
graph before any compute: a weekly series wired into a daily parameter, or a
resample landing on the wrong offset, fails at build time rather than producing
quiet nonsense. They are also what `satterc graph` uses to group and colour nodes
by frequency.

Each model reads the calendar it needs off one of its own time-bearing inputs.
There is no `dates_daily` / `dates_weekly` / `dates_monthly` node to supply. The
time axis lives on the data.

## Model chains

The chains we run:

- **SPLASH alone** — water balance only (evapotranspiration, soil moisture, runoff)
- **SPLASH → P-model** — adds GPP and light-use efficiency
- **SPLASH → P-model → SGAM** — vegetation dynamics with carbon pools
- **SPLASH → P-model → SGAM → RothC** — the above plus soil carbon

Chaining models takes more than listing them. A downstream model usually wants
its inputs at a different frequency, or in different units, from what the
upstream one produces. `examples/config.toml` in the repository is a worked
four-model pipeline showing the `[[resample]]` and `[[node]]` entries that
connect them.

See conduit's [configuration reference][conduit-config] for every config section.

[pyrealm]: https://github.com/ImperialCollegeLondon/pyrealm
[conduit-byom]: https://github.com/NERC-CEH/conduit/blob/develop/docs/guides/bring-your-own-module.md
[conduit-config]: https://github.com/NERC-CEH/conduit/blob/develop/docs/reference/configuration.md
