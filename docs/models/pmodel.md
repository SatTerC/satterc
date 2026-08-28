---
title: P-model
icon: lucide/leaf
---

# P-model


## Overview

The P-model is an optimality-based light use efficiency model for simulating ecosystem gross primary production (GPP).[^stocker2020]
Rather than assigning fixed trait values to plant functional types (PFTs), it predicts how plants continuously acclimate to their environment based on eco-evolutionary optimality theory – the idea that plants adapt to maximise resource use efficiency.[^prentice2014]
For a broader overview of the model's challenges and future directions, see the University of Reading LEMONTREE blog.[^lemontree2025]

The model rests on three hypotheses:

**Least-Cost Hypothesis** – Plants balance water loss during transpiration against carbon fixation capacity to minimise total cost.
This predicts how the ratio of leaf-internal to ambient CO₂ (χ) varies with environment: χ decreases in dry air (conserving water), increases in warm conditions (water transport becomes cheaper due to lower viscosity), and decreases at high altitude (lower O₂ makes photosynthesis cheaper). [^prentice2014]

**Coordination Hypothesis** – Plants adjust their photosynthetic capacity (Vcmax) to fully utilise available light, avoiding waste.
This yields a simple proportionality between photosynthetic capacity (at growth temperature) and incident light intensity.

**Cost-Benefit Hypothesis** – Leaves balance the cost of maintaining maximum electron transport rate (Jmax) against the benefit of electron-transport-limited photosynthesis, predicting the Jmax:Vcmax ratio. [^wang2017]

Together the three hypotheses let the P-model predict GPP without biome-specific calibration, from environmental drivers (light, temperature, CO₂, vapour pressure deficit, soil moisture) and fAPAR alone.

We wrap the [NumPy-based `pyrealm` implementation](https://github.com/ImperialCollegeLondon/pyrealm) of the P-model.
The [pyrealm P-model documentation](https://pyrealm.readthedocs.io/en/latest/users/pmodel/module_overview.html) is the authoritative source for the model theory.

The DAG for a pipeline running the P-model alone, with the dashed clusters grouping nodes by declared frequency (`7D` for weekly, `D` for daily):

<div class="model-graph">
--8<-- "docs/models/_graphs/pmodel.svg"
</div>

## Theory

The P-model builds on the Farquhar–von Caemmerer–Berry (FvCB) model of C3 photosynthesis, which describes the instantaneous response of photosynthesis to environmental conditions.
The FvCB model expresses the net CO₂ assimilation rate $A$ as the minimum of two limiting rates:

$$A = \min(W_c, W_j) - R_d$$

where $W_c$ is the Rubisco-limited rate, $W_j$ is the electron-transport-limited rate, and $R_d$ is dark respiration.

### Key quantities

The FvCB model depends on several quantities:

- **$\Gamma^*$** – the photorespiratory CO₂ compensation point, which varies predictably with temperature and O₂ partial pressure
- **$K$** – the effective Michaelis–Menten coefficient of Rubisco, also temperature- and O₂-dependent
- **$\phi_0$** – the maximum light-use efficiency of photosynthesis (theoretical maximum ≈ 0.125, i.e., at least 8 photons per fixed carbon atom)

Three quantities are treated as unknowns that plants optimise:

- **$V_{\text{cmax}}$** – maximum CO₂ fixation capacity (Rubisco activity)
- **$J_{\text{max}}$** – maximum electron transport capacity
- **$\chi = c_i / c_a$** – the ratio of leaf-internal to ambient CO₂

### Optimal $\chi$ (least-cost hypothesis)

The least-cost hypothesis yields an expression for optimal $\chi$ that depends on temperature, vapour pressure deficit $D$, atmospheric pressure $p$, and the relative viscosity of water $\eta^*$:

$$\chi = \frac{\Gamma^*}{c_a} + \left(1 - \frac{\Gamma^*}{c_a}\right) \frac{\xi}{\xi + \sqrt{D \cdot \eta^*}}$$

where $\xi$ is a cost factor that depends on $p$ and $\eta^*$.
This formulation captures the observed responses: $\chi$ decreases with increasing VPD, increases with temperature (via $\eta^*$), and decreases with altitude (via $p$).

### Light use efficiency

Under the coordination hypothesis, $V_{\text{cmax}}$ is set so that $W_c = W_j$ at typical light levels.
Combined with the optimal $\chi$, this leads to a prediction of light use efficiency (LUE, denoted $\text{LUE}$ or $m$):

$$\text{LUE} = \frac{A}{\text{PPFD} \cdot f_{\text{APAR}}}$$

where PPFD is photosynthetic photon flux density and $f_{\text{APAR}}$ is the fraction of absorbed PAR.
GPP is then:

$$\text{GPP} = \text{LUE} \cdot \text{PPFD} \cdot f_{\text{APAR}}$$

### Soil moisture stress

The P-model includes a soil moisture stress factor $\beta(\theta)$ that reduces GPP under dry conditions.
The implementation in `pyrealm` uses the soil moisture parameter $\theta$ (volumetric water content) to modulate both $\chi$ and LUE.

## Usage

### Quickstart

```sh
satterc setup --models pmodel --defaults
satterc data-gen config.toml --grid 1 1 --duration 2y --seed 42
satterc run config.toml
```

The [quickstart](../guides/quickstart.md) walks through what each of those does, and how to point the config at your own data instead of the synthetic set.

### Configuration

```toml
[pmodel]
_import_path = "satterc.models.pmodel"
method_optchi = "prentice14"
method_jmaxlim = "wang17"
method_kphio = "temperature"
method_arrhenius = "simple"
```

::: satterc.models.pmodel.pmodel_config
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_signature: false
      show_docstring_description: false
      show_docstring_returns: false
      docstring_section_style: spacy

### Inputs

::: satterc.models.pmodel.pmodel
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_signature: false
      show_docstring_description: false
      show_docstring_returns: false
      docstring_section_style: spacy

`mean_growth_temperature` is produced by this module rather than loaded, from daily temperature:

::: satterc.models.pmodel.mean_growth_temperature
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_signature: false
      separate_signature: false
      show_docstring_returns: false
      docstring_section_style: spacy

`aridity_index` is not: it is PET over precipitation, both accumulated over the record, and PET comes from [SPLASH](splash.md).
`examples/config.toml` derives it in a `[[node]]` rather than hiding the choice of PET inside a wrapper.

### Outputs

::: satterc.models.pmodel.PModelOut
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_docstring_description: false
      docstring_section_style: spacy
      show_bases: false
      members: false

None of these are the units SGAM consumes.
`examples/config.toml` defines `gpp_weekly` and `lue_weekly` as bridge nodes for that purpose, where the conversion factor is visible rather than buried in a wrapper.
iWUE needs no bridge, because `pyrealm` and SGAM agree on µmol mol⁻¹.

### Python API

See the [API documentation](../reference/modules/satterc.models/pmodel.md) for full function signatures and parameter details.

## References

[^stocker2020]: Stocker, B. D., Wang, H., Smith, N. G., Harrison, S. P., Keenan, T. F., Sandoval, D., Davis, T., and Prentice, I. C.: P-model v1.0: an optimality-based light use efficiency model for simulating ecosystem gross primary production, Geosci. Model Dev., 13, 1545–1581, https://doi.org/10.5194/gmd-13-1545-2020, 2020.

[^prentice2014]: Prentice, I. C., Dong, N., Gleason, S. M., Maire, V., and Wright, I. J.: Balancing the costs of carbon gain and water transport: testing a new theoretical framework for plant functional ecology, Ecol. Lett., 17, 82–91, https://doi.org/10.1111/ele.12211, 2014.

[^lemontree2025]: Sanders, N. (ed.): The P model: challenges we face and plan to address, University of Reading LEMONTREE blog, https://research.reading.ac.uk/lemontree/the-p-model-challenges-we-face-and-plan-to-address/, 2025.

[^wang2017]: Wang, H., Prentice, I. C., Keenan, T. F., Davis, T. W., Wright, I. J., Cornwell, W. K., Breon, F. M., Atkin, O. K., and Dong, N.: Towards a universal model for carbon dioxide uptake by plants, Nat. Plants, 3, 734–741, https://doi.org/10.1038/s41477-017-0006-x, 2017.
