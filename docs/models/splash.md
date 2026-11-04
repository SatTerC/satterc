---
title: Splash
icon: lucide/droplet
---

# SPLASH model


## Overview

SPLASH (Simple Process-led Algorithms for Simulating Habitats) is a water balance model that estimates soil moisture, actual evapotranspiration (AET) and surface water runoff for sites.[^davis2017]

The model takes an initial estimate of soil moisture, then uses time series of precipitation, temperature and sunshine fraction to track how the daily water balance changes with incoming precipitation, condensation and AET. Soil moisture, temperature and downwelling solar radiation at the site all feed into AET and condensation, which is why the site's elevation and latitude are required inputs.

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

The capacity also limits how fast the soil can supply water for evaporation, with supply rate $S_w = S_c W_{n[t-1]} / W_m$, where $S_c$ is 1.05 mm·h⁻¹.
$W_m$ comes from the required `max_soil_moisture` input, so it can vary per pixel.

/// admonition | Results differ from pyrealm 2.0.0 called directly
    type: note

pyrealm 2.0.0 uses the configured capacity for runoff but a fixed 150 mm for the supply rate.
This wrapper applies the configured capacity to both, as the equation above and Davis et al. (2017) specify, so its results differ from a direct pyrealm call wherever capacity is not 150 mm.
///

### Initial soil moisture estimation

SPLASH estimates initial soil moisture by iterating over a full year of climate data until the difference between year-start and year-end soil moisture falls below a threshold, so the run starts from a quasi-equilibrium state.
An input period shorter than a year cannot equilibrate and fails.

That tolerance is evaluated over the whole block, so results shift by ~1e-4 relative when the `[blocking]` block size or the `[subset]` range changes.

If any valid pixel in a block has not converged within `soil_moisture_init_max_iter` iterations, the run stops with an error.

To start from a measured state, or to continue an earlier run, set `soil_moisture_init_method = "prescribed"` and supply `initial_soil_moisture`.
That skips the spin-up, so the year minimum and the dependence on block size no longer apply.

### Missing values

Each pixel's outcome comes from its own inputs, so it does not depend on the block size:

| Pixel | Example | Outcome |
|---|---|---|
| Masked | All climate inputs NaN, or a NaN in `elevation`, `latitude`, `max_soil_moisture` or `initial_soil_moisture` | NaN outputs, with no error or warning |
| Gap | Some climate values NaN, including one variable with no data while the others have some | The run stops with an error naming the pixels |
| Model failure | Complete inputs, but pyrealm returns NaN | NaN outputs, with a warning naming the pixels |

A gap stops the run because soil moisture carries over from one day to the next, so everything after it would be lost.
Fill gaps in a `[[node]]` before SPLASH, or mask the pixel.

A misconfigured input that makes every climate variable NaN is indistinguishable from a masked block, so that run finishes with NaN outputs and no error.

Model failures come from pyrealm 2.0.0, whose AET calculation can return NaN at high latitudes.

The input dates must also be consecutive days, and a missing date stops the run with an error, because pyrealm steps through days by position.

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
