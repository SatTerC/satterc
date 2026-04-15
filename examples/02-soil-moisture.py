# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "scipy",
# ]
# ///
import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optimizing Splash Parameters

    This notebook demonstrates how to optimize model parameters using scipy's minimize function,
    and how to perform Bayesian inference using the Metropolis-Hastings algorithm.

    We use synthetic observations generated from the default parameter values as a sanity check
    that both approaches can recover the true `max_soil_moisture` parameter.
    """)
    return


@app.cell
def _():
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

    return (
        Config,
        OptimizeResult,
        Path,
        build_driver,
        generate_synthetic_data,
        minimize,
        mo,
        np,
        plt,
        tempfile,
        tomllib,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pipeline configuration

    We only need the SPLASH water-balance model, so the config is minimal: daily climate inputs,
    static inputs (elevation, plant type, maximum soil moisture), and no output modules
    since we will keep results in memory.
    """)
    return


@app.cell
def _(Config, tomllib):
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
    return (parsed_config,)


@app.cell
def _(Path, generate_synthetic_data, parsed_config, tempfile):
    # Generate synthetic input data into a temporary directory
    _tmpdir = Path(tempfile.mkdtemp())

    parsed_config["driver_config"]["daily_inputs_path"] = str(_tmpdir / "daily.nc")
    parsed_config["driver_config"]["static_inputs_path"] = str(_tmpdir / "static.nc")

    generate_synthetic_data(config=parsed_config, grid=(2, 2), n_days=730, seed=42)
    return


@app.cell
def _(build_driver, parsed_config):
    dr = build_driver(
        modules=parsed_config["modules"],
        config=parsed_config["driver_config"],
    )
    return (dr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generating synthetic observations

    As a sanity check that our optimisation and Bayesian inference are working correctly,
    we generate synthetic 'observations' by running the model with its default parameters
    and adding a small Gaussian noise perturbation.
    """)
    return


@app.cell
def _(dr, np):
    _outputs = dr.execute(["max_soil_moisture", "soil_moisture_daily"])
    _modelled_sm = _outputs["soil_moisture_daily"].values[:, 0]

    np.random.seed(42)
    _noise = np.random.normal(0, 5, _modelled_sm.shape)

    synthetic_obs_max_soil_moisture = float(_outputs["max_soil_moisture"].values[0])
    synthetic_obs = _modelled_sm + _noise
    n_pixels = _outputs["soil_moisture_daily"].sizes["pixel"]
    return n_pixels, synthetic_obs, synthetic_obs_max_soil_moisture


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parameter optimisation with synthetic observations
    """)
    return


@app.cell
def _(np, xr):
    def objective_function(params, dr, observations, n_pixels):
        max_sm = params[0]

        max_sm_grid = xr.DataArray(np.ones(n_pixels) * max_sm, dims=["pixel"])

        outputs = dr.execute(
            final_vars=["soil_moisture_daily"],
            overrides={"max_soil_moisture": max_sm_grid},
        )

        modelled_sm = outputs["soil_moisture_daily"].values[:, 0]

        return np.mean((modelled_sm - observations) ** 2)

    return (objective_function,)


@app.cell
def _(
    OptimizeResult,
    dr,
    minimize,
    n_pixels,
    np,
    objective_function,
    synthetic_obs,
):
    optimisation_history = []

    def logging_callback(intermediate_result: OptimizeResult) -> None:
        """A Callback to log the parameters and objective function at each iteration."""
        optimisation_history.append(
            [float(intermediate_result.x[0]), intermediate_result.fun]
        )

    optimisation_result = minimize(
        fun=objective_function,
        x0=[150],  # initial guess
        args=(dr, synthetic_obs, n_pixels),
        method="Nelder-Mead",
        callback=logging_callback,
    )

    return optimisation_result, optimisation_history


@app.cell
def _(np, plt, optimisation_history, synthetic_obs_max_soil_moisture):
    # Extract history of parameter (max_soil_moisture) and objective function value
    _x, _f = np.array(optimisation_history).T

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

    _axes[0].plot(_f, marker="o")
    _axes[0].set_xlabel("Iteration")
    _axes[0].set_ylabel("MSE")
    _axes[0].set_title("Objective Function Convergence")
    _axes[0].grid(True)
    _axes[0].axhline(y=0, color="k", linestyle="--", alpha=0.3)

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
    return


@app.cell
def _(mo, optimisation_result, synthetic_obs_max_soil_moisture):
    mo.md(f"""
    ### Optimization Results

    **True max_soil_moisture:** {synthetic_obs_max_soil_moisture:.2f}

    **Optimized Parameters:**
    - max_soil_moisture: {optimisation_result.x[0]:.2f}

    **Final MSE:** {optimisation_result.fun:.6f}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Bayesian Inference with synthetic observations

    We ultimately want to perform Bayesian inference to estimate posterior distributions
    for parameters of interest.

    Here, we reuse the synthetic observations to sanity-check a simple approach using
    the Metropolis-Hastings algorithm for a single parameter.
    """)
    return


@app.cell
def _(np, objective_function):
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

    return (make_log_posterior,)


@app.cell
def _(dr, make_log_posterior, n_pixels, np, optimisation_result, synthetic_obs):
    _prior_low = 150.0
    _prior_high = 250.0
    step_size = 3.0
    n_iterations = 200
    burn_in = 100

    # Adaptive Metropolis constants (Haario et al. 2001)
    _d = 1
    _s_d = 2.38**2 / _d  # optimal scaling for 1D Gaussian target
    _ε = 1e-6
    _t0 = burn_in // 2  # switch to adaptive proposals after this many steps

    # Warm-start from the MLE estimate rather than the prior boundary
    current = float(optimisation_result.x[0])
    mcmc_history = [current]
    accepted = 0

    # Welford online mean/variance state
    _n_w = 0
    _mean_w = current
    _var_M2 = 0.0

    log_posterior = make_log_posterior(
        dr,
        synthetic_obs,
        n_pixels,
        prior_low=_prior_low,
        prior_high=_prior_high,
        likelihood_sigma=5.0,
    )

    for i in range(burn_in + n_iterations):
        # Phase 1: fixed uniform step; Phase 2: adaptive Gaussian proposal
        if i < _t0:
            proposed = current + np.random.uniform(-step_size, step_size)
        else:
            _sigma = np.sqrt(_s_d * (_var_M2 / max(1, _n_w - 1) + _ε))
            proposed = np.random.normal(current, _sigma)

        log_acceptance_prob = log_posterior([proposed]) - log_posterior([current])

        if np.log(np.random.uniform()) < log_acceptance_prob:
            current = proposed

            if i > burn_in:  # only track acceptance after burn-in
                accepted += 1

        # Welford update using accepted position
        _n_w += 1
        _delta = current - _mean_w
        _mean_w += _delta / _n_w
        _var_M2 += _delta * (current - _mean_w)

        mcmc_history.append(current)

    acceptance_rate = accepted / n_iterations

    acceptance_rate
    return burn_in, mcmc_history


@app.cell
def _(burn_in, mcmc_history, np, plt, synthetic_obs_max_soil_moisture):
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
    return


if __name__ == "__main__":
    app.run()
