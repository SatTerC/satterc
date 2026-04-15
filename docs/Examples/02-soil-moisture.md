---
title: 02 Soil Moisture
marimo-version: 0.23.0
width: medium
header: |-
  # /// script
  # requires-python = ">=3.13"
  # dependencies = [
  #     "marimo",
  #     "matplotlib",
  #     "scipy",
  # ]
  # ///
---

# Optimizing Splash Parameters

This notebook demonstrates how to optimize model parameters using scipy's minimize function,
and how to perform Bayesian inference using the Metropolis-Hastings algorithm.

We use synthetic observations generated from the default parameter values as a sanity check
that both approaches can recover the true `max_soil_moisture` parameter.

```python {.marimo}
import tempfile
import tomllib
from pathlib import Path

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.optimize import minimize, OptimizeResult

from satterc import build_driver
from satterc.config import Config
from satterc.setup_utils.data_gen import generate_synthetic_data
```

## Pipeline configuration

We only need the SPLASH water-balance model, so the config is minimal: daily climate inputs,
static inputs (elevation, plant type, maximum soil moisture), and no output modules
since we will keep results in memory.

```python {.marimo}
_config_toml = """
modules = [
  "models.splash",
  "inputs.daily",
  "inputs.static",
]

[inputs.daily]
path = "daily.nc"
vars = [
  "precipitation_mm",
  "sunshine_fraction",
  "temperature_celcius",
]

[inputs.static]
path = "static.nc"
vars = [
  "elevation",
  "plant_type",
  "max_soil_moisture",
]
"""

parsed_config = Config(tomllib.loads(_config_toml)).parse()
parsed_config
```

```python {.marimo}
# Generate synthetic input data into a temporary directory
_tmpdir = Path(tempfile.mkdtemp())

parsed_config["driver_config"]["daily_inputs_path"] = str(_tmpdir / "daily.nc")
parsed_config["driver_config"]["static_inputs_path"] = str(_tmpdir / "static.nc")

generate_synthetic_data(config=parsed_config, grid=(2, 2), n_days=730, seed=42)
```

```python {.marimo}
dr = build_driver(
    modules=parsed_config["modules"],
    config=parsed_config["driver_config"],
)
```

## Generating synthetic observations

As a sanity check that our optimisation and Bayesian inference are working correctly,
we generate synthetic 'observations' by running the model with its default parameters
and adding a small Gaussian noise perturbation.

```python {.marimo}
_outputs = dr.execute(["max_soil_moisture", "soil_moisture_daily"])
_modelled_sm = _outputs["soil_moisture_daily"].values[:, 0]

np.random.seed(42)
_noise = np.random.normal(0, 5, _modelled_sm.shape)

synthetic_obs_max_soil_moisture = float(_outputs["max_soil_moisture"].values[0])
synthetic_obs = _modelled_sm + _noise
n_pixels = _outputs["soil_moisture_daily"].sizes["pixel"]
```

## Parameter optimisation with synthetic observations

```python {.marimo}
def objective_function(params, dr, observations, n_pixels):
    max_sm = params[0]

    max_sm_grid = xr.DataArray(np.ones(n_pixels) * max_sm, dims=["pixel"])

    outputs = dr.execute(
        final_vars=["soil_moisture_daily"],
        overrides={"max_soil_moisture": max_sm_grid},
    )

    modelled_sm = outputs["soil_moisture_daily"].values[:, 0]

    return np.mean((modelled_sm - observations) ** 2)
```

```python {.marimo}
optimisation_history = []

def logging_callback(intermediate_result: OptimizeResult) -> None:
    """A Callback to log the parameters and objective function at each iteration."""
    optimisation_history.append(
        [float(intermediate_result.x[0]), intermediate_result.fun]
    )

optimisation_result = minimize(
    fun=objective_function,
    x0=[150],  # initial guess: prior lower bound
    args=(dr, synthetic_obs, n_pixels),
    method="Nelder-Mead",
    callback=logging_callback,
    options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 2000},
)
```

```python {.marimo}
# Extract history of parameter (max_soil_moisture) and objective function value
_x, _f = np.array(optimisation_history).T

_fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

_axes[0].plot(_f, marker="o")
_axes[0].axhline(
    y=25, color="k", linestyle="--", alpha=0.6, label="Expected minimum (σ²=25)"
)
_axes[0].set_xlabel("Iteration")
_axes[0].set_ylabel("MSE")
_axes[0].set_title("Objective Function Convergence")
_axes[0].legend()
_axes[0].grid(True)

_axes[1].plot(_x, marker="o")
_axes[1].axhline(
    y=synthetic_obs_max_soil_moisture,
    color="r",
    linestyle="--",
    label=f"True value ({synthetic_obs_max_soil_moisture:.1f})",
)
_axes[1].set_xlabel("Iteration")
_axes[1].set_ylabel("max_soil_moisture")
_axes[1].set_title("Parameter Evolution")
_axes[1].legend()
_axes[1].grid(True)

_fig.tight_layout()
_fig
```

```python {.marimo}
mo.md(f"""
### Optimization Results

**True max_soil_moisture:** {synthetic_obs_max_soil_moisture:.2f}

**Optimized Parameters:**
- max_soil_moisture: {optimisation_result.x[0]:.2f}

**Final MSE:** {optimisation_result.fun:.6f}
""")
```

---

## Bayesian Inference with synthetic observations

Two features of the optimisation results are worth noting before we proceed:

- **The MSE does not converge to zero.** The irreducible minimum is σ² = 25,
  the variance of the noise added to the synthetic observations. Even with the
  exact true parameter, the model cannot fit the noise away.
- **The MLE does not recover the exact true value.** With a finite, noisy
  sample the optimizer finds the parameter that best explains the *noisy*
  signal, not the true one. This bias shrinks with more data but never vanishes.

This motivates Bayesian inference: rather than a single point estimate, we want
a *distribution* over plausible parameter values that honestly reflects the
residual uncertainty. Here we use the Metropolis-Hastings algorithm.

Note that with a uniform prior the posterior is proportional to the likelihood
alone — p(θ|y) ∝ p(y|θ) — so the posterior mean coincides with the MLE.

```python {.marimo}
def make_log_posterior(
    dr,
    synthetic_obs,
    n_pixels,
    prior_low: float,
    prior_high: float,
    likelihood_sigma: float,
):
    π = np.pi
    σ = likelihood_sigma
    N = len(synthetic_obs)

    def log_likelihood(params):
        """Logarithm of a Gaussian likelihood function."""
        mse = objective_function(params, dr, synthetic_obs, n_pixels)
        return -(N / (2 * σ**2)) * mse - (N / 2) * np.log(2 * π * σ**2)

    def log_prior(params):
        if prior_low <= params[0] <= prior_high:
            return 0.0
        return -np.inf

    def log_posterior(params):
        return log_prior(params) + log_likelihood(params)

    return log_posterior
```

```python {.marimo}
_prior_low = 150.0
_prior_high = 250.0
step_size = 0.5
n_iterations = 200
burn_in = 100

# Warm-start from the MLE estimate
current = float(optimisation_result.x[0])
mcmc_history = [current]
accepted = 0

log_posterior = make_log_posterior(
    dr,
    synthetic_obs,
    n_pixels,
    prior_low=_prior_low,
    prior_high=_prior_high,
    likelihood_sigma=5.0,
)

for i in range(burn_in + n_iterations):
    proposed = current + np.random.uniform(-step_size, step_size)

    log_acceptance_prob = log_posterior([proposed]) - log_posterior([current])

    if np.log(np.random.uniform()) < log_acceptance_prob:
        current = proposed

        if i >= burn_in:
            accepted += 1

    mcmc_history.append(current)

acceptance_rate = accepted / n_iterations

acceptance_rate
```

```python {.marimo}
posterior_samples = mcmc_history[burn_in:]

_fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

_axes[0].plot(mcmc_history)
_axes[0].axvline(x=burn_in, color="r", linestyle="--", label=f"Burn-in ({burn_in})")
_axes[0].axhline(
    y=synthetic_obs_max_soil_moisture,
    color="g",
    linestyle=":",
    label=f"True value ({synthetic_obs_max_soil_moisture:.1f})",
)
_axes[0].set_xlabel("Iteration")
_axes[0].set_ylabel("max_soil_moisture")
_axes[0].set_title("MCMC History")
_axes[0].legend()
_axes[0].grid(True)

_axes[1].hist(
    posterior_samples, bins=20, density=True, alpha=0.7, label="Posterior"
)
_x = np.linspace(min(posterior_samples), max(posterior_samples), 100)
prior_pdf = np.ones_like(_x) / (250 - 150)  # uniform prior on [150, 250]
_axes[1].plot(_x, prior_pdf, "r--", linewidth=2, label="Uniform Prior")
_axes[1].axvline(
    x=synthetic_obs_max_soil_moisture,
    color="g",
    linestyle=":",
    label=f"True value ({synthetic_obs_max_soil_moisture:.1f})",
)
_axes[1].set_xlabel("max_soil_moisture")
_axes[1].set_ylabel("Density")
_axes[1].set_title("Posterior Distribution")
_axes[1].legend()
_axes[1].grid(True)

_fig.tight_layout()
_fig
```