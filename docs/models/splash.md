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

The calculated soil moisture is capped at the maximum soil moisture capacity ($W_m$), with excess water allocated to surface water runoff:

$$
\text{if } W_{n[t]} > W_m: \quad W_{n[t]} = W_m, \quad R_{[t]} = W_{n[t]} - W_m
$$

The maximum soil moisture capacity defaults to 150 mm but can be set on a per-site basis.

### Initial soil moisture estimation

SPLASH estimates initial soil moisture by iterating over a full year of climate data until the difference between year-start and year-end soil moisture falls below a threshold, so the run starts from a quasi-equilibrium state. Because that tolerance is evaluated over the whole block, results shift by ~1e-4 relative when the block size or subset changes.

## Usage

### Configuration

SPLASH is configured in the TOML config file:

```toml
[splash]
_import_path = "satterc.models.splash"
soil_moisture_init_max_iter = 10
soil_moisture_init_max_diff = 1.0
```

Both settings are optional, and both are top-level keys of the `[splash]` section.

::: satterc.models.splash.splash_params
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_signature: false
      show_docstring_description: false
      show_docstring_returns: false
      docstring_section_style: spacy

### Inputs

The three climate inputs are daily `DataArray`s; the three site inputs are static.
`splash_params` carries the settings above.

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
