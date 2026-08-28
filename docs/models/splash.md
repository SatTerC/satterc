---
title: Splash
icon: lucide/droplet
---

# SPLASH model


## Overview

SPLASH (Simple Process-led Algorithms for Simulating Habitats) is a water balance model that estimates soil moisture, actual evapotranspiration (AET) and surface water runoff for sites.[^davis2017]

The model takes an initial estimate of soil moisture, then uses time series of precipitation, temperature and cloud cover to track how the daily water balance changes with incoming precipitation, condensation and AET. Soil moisture, temperature and downwelling solar radiation at the site all feed into AET and condensation, which is why the site's elevation and latitude are required inputs.

We wrap the [NumPy-based `pyrealm` implementation](https://github.com/ImperialCollegeLondon/pyrealm) of SPLASH.
The [pyrealm SPLASH documentation](https://pyrealm.readthedocs.io/en/latest/users/splash.html) is the authoritative source for the model theory.

The DAG for a pipeline running SPLASH alone, with the dashed cluster grouping nodes by declared frequency (`D` for daily):

<div class="model-graph">
--8<-- "docs/models/_graphs/splash.svg"
</div>

## Theory

The daily water balance equation is:

$$
W_{n[t]} = W_{n[t-1]} + P_{[t]} + C_{[t]} - \textrm{AET}_{[t]},
$$

where:

- $W_{n[t]}$ – current soil moisture (mm)
- $W_{n[t-1]}$ – previous day's soil moisture (mm)
- $P_{[t]}$ – precipitation (mm·d⁻¹)
- $C_{[t]}$ – condensation (mm·d⁻¹)
- $\textrm{AET}_{[t]}$ – actual evapotranspiration (mm·d⁻¹)

Runoff is whatever that balance leaves above the maximum soil moisture capacity ($W_m$), taken before the cap is applied:

$$
R_{[t]} = \max\left(W_{n[t]} - W_m,\; 0\right), \qquad W_{n[t]} \leftarrow \min\left(W_{n[t]}, W_m\right)
$$

The maximum soil moisture capacity defaults to 150 mm but can be set on a per-site basis.

### Initial soil moisture estimation

SPLASH estimates initial soil moisture by iterating over a full year of climate data until the difference between year-start and year-end soil moisture falls below a threshold, so the run starts from a quasi-equilibrium state.
An input period shorter than a year cannot equilibrate and fails.

That tolerance is evaluated over the whole block, so results shift by ~1e-4 relative when the `[blocking]` block size or the `[subset]` range changes.

## Usage

### Quickstart

```sh
satterc setup --models splash --defaults
satterc data-gen generate config.toml --grid 1 1 --duration 2y --seed 42
satterc run config.toml
```

The [quickstart](../getting_started/quickstart.md) walks through what each of those does, and how to point the config at your own data instead of the synthetic set.
For calibrating `max_soil_moisture` against observations, see the [soil moisture example](../examples/soil_moisture.md).

### Configuration

```toml
[splash]
_import_path = "satterc.models.splash"
soil_moisture_init_max_iter = 10
soil_moisture_init_max_diff = 1.0
```

::: satterc.models.splash.splash_config
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_signature: false
      show_docstring_description: false
      show_docstring_returns: false
      docstring_section_style: spacy

### Inputs

::: satterc.models.splash.splash
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_signature: false
      show_docstring_description: false
      show_docstring_returns: false
      docstring_section_style: spacy

### Outputs

::: satterc.models.splash.SplashOut
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_docstring_description: false
      docstring_section_style: spacy
      show_bases: false
      members: false

### Python API

See the [API documentation](../api/satterc.models/splash.md) for full function signatures and parameter details.

## References

[^davis2017]: Davis, T. W., Prentice, I. C., Stocker, B. D., Thomas, R. T., Whitley, R. J., Wang, H., Evans, B. J., Gallego-Sala, A. V., Sykes, M. T., and Cramer, W.: Simple process-led algorithms for simulating habitats (SPLASH v.1.0): robust indices of radiation, evapotranspiration and plant-available moisture, Geosci. Model Dev., 10, 689–708, https://doi.org/10.5194/gmd-10-689-2017, 2017.
