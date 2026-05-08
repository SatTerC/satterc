---
title: Overview
icon: lucide/package
---

# Built-in Models

SatTerC ships with four built-in models that can be composed into pipelines.
Each model is a module in the DAG that computes specific variables from its inputs.

| Model | Description | Temporal Resolution |
|-------|-------------|---------------------|
| [SPLASH](splash.md) | Semi-empirical water-balance model that computes actual evapotranspiration, soil moisture, and runoff from daily climate data | Daily |
| [P-Model](pmodel.md) | Optimal photosynthesis model that computes gross primary production (GPP), light-use efficiency, and leaf area index from environmental drivers | Weekly |
| [SGAM](sgam.md) | Simple Global Assimilation Model — a vegetation dynamics model that tracks carbon pools (leaf, stem, root, litter) over time | Weekly |
| [RothC](rothc.md) | Soil carbon decomposition model that simulates the turnover of organic matter in soil, producing soil organic carbon stocks | Monthly |

## Typical Model Chains

Models are designed to be composed. Common configurations include:

- **SPLASH alone** — water balance only (evapotranspiration, soil moisture, runoff)
- **SPLASH → P-Model** — adds GPP and LAI estimation
- **SPLASH → P-Model → SGAM** — full vegetation dynamics with carbon pools
- **SPLASH → P-Model → SGAM → RothC** — complete terrestrial carbon cycle including soil carbon

See the [Configuration](../usage/config.md) reference for how to select and configure models in your pipeline.
