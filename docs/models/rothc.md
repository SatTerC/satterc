---
title: RothC
icon: lucide/worm
---

# Rothamsted Carbon model (RothC)


## Overview

RothC simulates the turnover of soil organic carbon (SOC) in non-waterlogged soils.[^jenkinson1990] It splits SOC into five distinct compartments – four active and one inert – and accounts for soil type (clay content), temperature, moisture, and plant cover to calculate decay rates.

The five pools are:

| Pool | Name | Description |
|------|------|-------------|
| DPM | Decomposable Plant Material | Easily decomposed organic matter |
| RPM | Resistant Plant Material | More slowly decomposed organic matter |
| BIO | Microbial Biomass | Living microbial biomass |
| HUM | Humified Organic Matter | Stable humus |
| IOM | Inert Organic Matter | Chemically inert, does not decompose |

We forked the Python implementation from [Rothamsted Research](https://github.com/Rothamsted-Models/RothC_Py/), repackaged it for pip installation and made it ~20× faster. Our fork is available at [github.com/SatTerC/RothC_Py](https://github.com/SatTerC/RothC_Py).

The DAG for a pipeline running RothC alone, with the dashed cluster grouping nodes by declared frequency (`1ME` for monthly):

<div class="model-graph">
--8<-- "docs/models/_graphs/rothc.svg"
</div>

## Theory

Carbon turnover follows first-order kinetics, where each active pool $k$ evolves according to:

$$
\frac{dC_k}{dt} = I_k - k_k \cdot C_k
$$

where $C_k$ is the carbon content of pool $k$, $I_k$ is the carbon input to that pool, and $k_k$ is the decomposition rate constant, modified by temperature, moisture, and soil cover factors.

The decomposition rates are scaled by:

- **Temperature rate modifier** – increases with temperature
- **Moisture rate modifier** – depends on the ratio of rainfall to evaporation
- **Soil cover factor** – reduces decomposition when soil is covered by vegetation

### Spin-up

RothC requires initial pool sizes to begin simulation. A spin-up phase runs the model over repeated climate cycles until the pools reach equilibrium, which fixes the initial conditions.

### Evapotranspiration, not open-pan evaporation

RothC's own water-balance driver is open-pan evaporation, which the underlying RothC_Py scales by 0.75 to get evapotranspiration.
SatTerC instead supplies potential evapotranspiration directly, since SPLASH computes it (Priestley-Taylor) on its way to AET.
There is nothing to convert, so `evap_factor` defaults to 1.0.

PET, not AET, is the right driver: RothC's water balance is rainfall minus evaporative *demand*, and AET is already suppressed by the soil dryness RothC is itself computing, so feeding AET in double-counts the water limitation and holds the soil systematically too wet.

For full model details, see:

- [SatTerC RothC_Py documentation](https://satterc.github.io/RothC_Py/science.html)
- [Original model description paper](https://github.com/Rothamsted-Models/RothC_Py/blob/main/RothC_description.pdf) (Coleman, Prout and Milne, Rothamsted Research)

## Usage

### Quickstart

RothC needs potential evapotranspiration, which [SPLASH](splash.md) produces, so run the two together:

```sh
satterc setup --models splash --models rothc --defaults
satterc data-gen generate config.toml --grid 1 1 --duration 2y --seed 42
satterc run config.toml
```

The [quickstart](../getting_started/quickstart.md) walks through what each of those does, and how to point the config at your own data instead of the synthetic set.

### Configuration

```toml
[rothc]
_import_path = "satterc.models.rothc"
n_years_spinup = 1
```

::: satterc.models.rothc.rothc_config
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_signature: false
      show_docstring_description: false
      show_docstring_returns: false
      docstring_section_style: spacy

The per-PFT DPM/RPM ratios are settings of the `dpm_rpm_ratio_monthly` bridge node below, and are configured as keys of the same `[rothc]` section:

::: satterc.models.rothc.dpm_rpm_ratio_config
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_signature: false
      show_docstring_description: false
      show_docstring_returns: false
      docstring_section_style: spacy

### Inputs

::: satterc.models.rothc.rothc
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_signature: false
      show_docstring_description: false
      show_docstring_returns: false
      docstring_section_style: spacy

### Bridge nodes

Three of those monthly inputs are produced by this module rather than loaded.
Each builds a monthly series out of per-pixel static data, which leaves it nothing to get a calendar from — so each also takes `temperature_monthly`, read purely for its time coordinate.
If you supply your own version of one of these nodes, it needs a time-bearing input for the same reason.

::: satterc.models.rothc.plant_cover_monthly
    options:
      show_root_heading: true
      show_root_full_path: false
      show_root_toc_entry: false
      show_signature: false
      separate_signature: false
      show_docstring_returns: false
      heading_level: 4
      docstring_section_style: spacy

::: satterc.models.rothc.dpm_rpm_ratio_monthly
    options:
      show_root_heading: true
      show_root_full_path: false
      show_root_toc_entry: false
      show_signature: false
      separate_signature: false
      show_docstring_returns: false
      heading_level: 4
      docstring_section_style: spacy

::: satterc.models.rothc.farmyard_manure_input_monthly
    options:
      show_root_heading: true
      show_root_full_path: false
      show_root_toc_entry: false
      show_signature: false
      separate_signature: false
      show_docstring_returns: false
      heading_level: 4
      docstring_section_style: spacy

### Outputs

::: satterc.models.rothc.RothCOut
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_docstring_description: false
      docstring_section_style: spacy
      show_bases: false
      members: false

### Python API

See the [API documentation](../api/satterc.models/rothc.md) for full function signatures and parameter details.

## References

[^jenkinson1990]: Jenkinson, D. S.: The Turnover of Organic Carbon and Nitrogen in Soil, Philosophical Transactions of the Royal Society of London, Series B: Biological Sciences, 329, 361–368, 1990.

[^jenkinson1987]: Jenkinson, D. S., et al.: Modelling the turnover of organic matter in long-term experiments at Rothamsted, INTECOL Bulletin, 15, 1–8, 1987.

[^jenkinson1977]: Jenkinson, D. S., and Rayner, J. H.: Turnover of soil organic matter in some of the Rothamsted classical experiments, Soil Science, 123, 298–305, 1977.

[^bolinder2007]: Bolinder, M. A., et al.: An approach for estimating net primary productivity and annual carbon inputs to soil for common agricultural crops in Canada, Agriculture, Ecosystems & Environment, 118, 29–42, 2007.

[^farina2013]: Farina, R., et al.: Modification of the RothC model for simulations of soil organic C dynamics in dryland regions, Geoderma, 200, 18–30, 2013.

[^giongo2020]: Giongo, V., et al.: Optimizing multifunctional agroecosystems in irrigated dryland agriculture to restore soil carbon – Experiments and modelling, Science of the Total Environment, 725, 2020.
