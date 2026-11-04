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

If any valid pixel in a block has not converged within `soil_moisture_init_max_iter` iterations, the run stops with an error.

To start from a measured state, or to continue an earlier run, set `soil_moisture_init_method = "prescribed"` and supply `initial_soil_moisture`.
That skips the spin-up, so the year minimum and the dependence on block size no longer apply.

### Missing values

Soil moisture carries over from one day to the next, so a missing value stays in that pixel's results:

- A missing value in a pixel's first year makes the whole pixel NaN, because the spin-up cannot finish.
- A missing value after the first year makes the pixel NaN from that day on.

Neither case raises an error while the block has another valid pixel.
A block in which every pixel is NaN stops the run with an error naming the cause, so a block of masked pixels needs leaving out.
Whether a gap raises can therefore depend on the block size.

## Usage

### Quickstart

```sh
satterc setup --models splash --defaults
satterc data-gen config.toml --grid 1 1 --duration 2y --seed 42
satterc run config.toml
```

The [quickstart](../guides/quickstart.md) walks through what each of those does, and how to point the config at your own data instead of the synthetic set.
For calibrating `max_soil_moisture` against observations, see the [soil moisture recipe](../recipes/soil_moisture.md).

### Configuration

```toml
[splash]
soil_moisture_init_method = "equilibrate"
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

#### Always required

::: satterc.models.splash.splash
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_signature: false
      show_docstring_description: false
      show_docstring_returns: false
      docstring_section_style: spacy
      extra:
        hide_parameters: [soil_moisture_init_inputs, splash_config]

#### Initial state, for `prescribed` initialisation

Required when `soil_moisture_init_method` is `prescribed`.

::: satterc.models.splash.soil_moisture_init_inputs__prescribed
    options:
      show_root_full_path: false
      separate_signature: false
      heading_level: 4
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

See the [API documentation](../reference/modules/satterc.models/splash.md) for full function signatures and parameter details.

## References

[^davis2017]: Davis, T. W., Prentice, I. C., Stocker, B. D., Thomas, R. T., Whitley, R. J., Wang, H., Evans, B. J., Gallego-Sala, A. V., Sykes, M. T., and Cramer, W.: Simple process-led algorithms for simulating habitats (SPLASH v.1.0): robust indices of radiation, evapotranspiration and plant-available moisture, Geosci. Model Dev., 10, 689–708, https://doi.org/10.5194/gmd-10-689-2017, 2017.
