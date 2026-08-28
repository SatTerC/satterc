---
title: Overview
icon: lucide/package
---

# Built-in models

`satterc` currently comes with four models, which can be composed into a pipeline.
Each model is called through a single Python function, which becomes a node in the DAG.

```toml
[splash]
_import_path = "satterc.models.splash"
soil_moisture_init_max_iter = 10
```


| Model | Description | Temporal resolution |
|-------|-------------|---------------------|
| [SPLASH](splash.md) | Semi-empirical water-balance model that computes actual evapotranspiration, soil moisture, and runoff from daily climate data | Daily |
| [P-model](pmodel.md) | Optimality-based photosynthesis model that computes gross primary production (GPP), light-use efficiency and intrinsic water-use efficiency from environmental drivers | Weekly |
| [SGAM](sgam.md) | Simplified Growth and Allocation Model: tracks carbon pools (leaf, stem, root, litter) over time | Weekly |
| [RothC](rothc.md) | Soil carbon decomposition model that simulates the turnover of organic matter in soil, producing soil organic carbon stocks | Monthly |


