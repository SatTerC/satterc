---
title: Overview
icon: lucide/package
---

# Built-in Models

SatTerC ships with four models that can be composed into pipelines. Each is an
ordinary [conduit module][conduit-byom] — plain functions whose names are DAG
node names — and is referenced from a config by its dotted `_import_path`:

```toml
[splash]
_import_path = "satterc.models.splash"
soil_moisture_init_max_iter = 10
```

| Model | Description | Temporal Resolution |
|-------|-------------|---------------------|
| [SPLASH](splash.md) | Semi-empirical water-balance model that computes actual evapotranspiration, soil moisture, and runoff from daily climate data | Daily |
| [P-Model](pmodel.md) | Optimal photosynthesis model that computes gross primary production (GPP), light-use efficiency, and leaf area index from environmental drivers | Weekly |
| [SGAM](sgam.md) | Simple Global Assimilation Model — a vegetation dynamics model that tracks carbon pools (leaf, stem, root, litter) over time | Weekly |
| [RothC](rothc.md) | Soil carbon decomposition model that simulates the turnover of organic matter in soil, producing soil organic carbon stocks | Monthly |

## Naming and declared frequencies

The `_daily` / `_weekly` / `_monthly` / static suffixes on node names are a
SatTerC convention, not framework behaviour: conduit treats a config section's
label as inert and infers no frequency from it. What actually carries the
frequency is a contract declared on each model's signature:

| Suffix | Declared offset | `satterc.frequencies` |
|--------|-----------------|------------------------|
| `_daily` | `D` | `DAILY` |
| `_weekly` | `7D` (unanchored — any weekday) | `WEEKLY` |
| `_monthly` | `1ME` (month end) | `MONTHLY` |

Because those declarations are on the functions, conduit can check the whole
graph before any compute: a weekly series wired into a daily parameter, or a
resample landing on the wrong offset, fails at build time rather than producing
quiet nonsense. They are also what `satterc graph` uses to group and colour nodes
by frequency.

Each model reads the calendar it needs off one of its own time-bearing inputs.
There is no `dates_daily` / `dates_weekly` / `dates_monthly` node to supply — the
time axis lives on the data.

## Typical Model Chains

Models are designed to be composed. Common configurations include:

- **SPLASH alone** — water balance only (evapotranspiration, soil moisture, runoff)
- **SPLASH → P-Model** — adds GPP and LAI estimation
- **SPLASH → P-Model → SGAM** — full vegetation dynamics with carbon pools
- **SPLASH → P-Model → SGAM → RothC** — complete terrestrial carbon cycle including soil carbon

Chaining models takes more than listing them: a downstream model generally wants
its inputs at a different frequency, or in different units, from what the
upstream one produces. `examples/config.toml` in the repository is a worked
four-model pipeline showing the `[[resample]]` and `[[node]]` entries that
connect them.

See conduit's [configuration reference][conduit-config] for every config section.

[conduit-byom]: https://github.com/NERC-CEH/conduit/blob/develop/docs/guides/bring-your-own-module.md
[conduit-config]: https://github.com/NERC-CEH/conduit/blob/develop/docs/reference/configuration.md
