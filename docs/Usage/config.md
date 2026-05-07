---
title: Configuration
icon: lucide/settings
---

# Configuration

A SatTerC pipeline is described by a [TOML](https://toml.io/en/) configuration file.
Each section in the file activates a pipeline component — selecting which data to load,
which models to run, and which results to save.
Sections that are absent are simply not included in the pipeline, so you can build
a lightweight pipeline from only the components you need.

---

## Built-in modules

### Inputs

Input sections declare where to load data from and which variables to extract.
Include one section per temporal frequency for which you have data.

```toml
[inputs.daily]
path = "data/daily.nc"
vars = [
  "temperature_celcius",
  "precipitation_mm",
  "sunshine_fraction",
]

[inputs.weekly]
path = "data/weekly.nc"
vars = [
  "co2_ppm",
  "fapar",
  "ppfd_umol_m2_s1",
  "pressure_pa",
  "vpd_pa",
]

[inputs.monthly]
path = "data/monthly.nc"
vars = ["soil_organic_carbon"]

[inputs.static]
path = "data/static.nc"
vars = [
  "elevation",
  "clay_content",
  "soil_depth",
]
```

`path` may point to a NetCDF file (`.nc`, `.netcdf`), a Zarr store (`.zarr` or a bare
directory), or a flat file (`.csv`, `.parquet`). The format is inferred from the
extension — you do not need to specify it explicitly.

Only include the frequencies for which you have data.
A pipeline with only `[inputs.daily]` and `[inputs.static]` is perfectly valid.

---

### Grid

Include `[grid]` when your input data is a 2D spatial grid (i.e. has `x`/`y` dimensions
and a CRS). It derives latitude and longitude arrays from the common spatial reference
shared by all loaded datasets, and makes them available as DAG nodes (`latitude`,
`longitude`) for downstream models.

```toml
[grid]
```

No parameters are required. Omit this section entirely for point-based or
pre-stacked data.

---

### Models

Include one section per model you want to run.
Any model-specific parameters go in the section body; if a model has no required
parameters (or you are happy with all defaults) the section body can be empty.

```toml
[models.splash]

[models.pmodel]
method_kphio = "sandoval"
method_optchi = "lavergne20_c3"

[models.sgam]

[models.rothc]
n_years_spinup = 3
```

Available built-in models: `splash`, `pmodel`, `sgam`, `rothc`.
See the [Models](../Models/splash.md) section for the parameters each model accepts.

/// admonition | Parameter namespacing
    type: note

All model parameters are merged into a single flat configuration dictionary, so
parameter names must be unique across all active model sections.
If two models share a parameter name, prefix it to disambiguate
(e.g. `pmodel_method_kphio`).
///

---

### Resampling

Use `[[resample]]` (an [array of tables](https://toml.io/en/v1.0.0#array-of-tables))
to resample one or more variables from a finer temporal frequency to a coarser one.
Each entry specifies a direction, a list of variables, and an optional aggregation function.

```toml
[[resample]]
from_freq = "daily"
to_freq = "weekly"
vars = ["temperature_celcius", "precipitation_mm"]
aggfunc = "mean"  # default; can also be "sum"

[[resample]]
from_freq = "daily"
to_freq = "monthly"
vars = ["precipitation_mm"]
aggfunc = "sum"
```

Supported directions:

| `from_freq` | `to_freq`   |
|-------------|-------------|
| `"daily"`   | `"weekly"`  |
| `"daily"`   | `"monthly"` |
| `"weekly"`  | `"monthly"` |

Omit `[[resample]]` entirely if no resampling is needed.

---

### Outputs

Output sections declare which computed variables to save and where.
Include one section per temporal frequency you want to write to disk.

```toml
[outputs.daily]
path = "results/daily.nc"
vars = ["actual_evapotranspiration", "soil_moisture"]

[outputs.weekly]
path = "results/weekly.nc"
vars = ["gpp", "leaf_pool"]

[outputs.monthly]
path = "results/monthly.nc"
vars = ["soil_organic_carbon"]
```

As with inputs, the output format is inferred from the file extension.
Both `path` and at least one entry in `vars` are required; omit the section entirely
to produce no output at that frequency.

---

## Custom modules

You can extend the pipeline with any importable Python module by adding a section
with a `_import_path` key pointing to its dotted module path.
All other keys in the section are passed through to the pipeline as configuration
parameters, exactly as model parameters are.

```toml
[my_custom_model]
_import_path = "mypackage.mymodule"
learning_rate = 0.01
n_iterations = 500
```

The section header (`my_custom_model`) is a free-form human-readable label;
only `_import_path` carries semantic meaning.
The referenced module must follow the same Hamilton DAG conventions as the built-in modules.

/// admonition | Parameter conflicts
    type: warning

Custom module parameters are merged into the same flat configuration dictionary as
built-in model parameters. Ensure your parameter names do not clash with those
of any active built-in section.
///

---

## Auto-generating a config file

The `setup` CLI command generates a valid configuration file through a series of
interactive prompts, inferring which input variables and resampling steps are
required by the models you select.

```bash
satterc setup
```

To skip the interactive prompts and use default file paths:

```bash
satterc setup --models splash pmodel --defaults
```

Key options:

| Flag | Short | Description |
|------|-------|-------------|
| `--models` | `-m` | Space-separated list of built-in models to include |
| `--output` | `-o` | Output path for the generated config file (default: `config.toml`) |
| `--defaults` | `-d` | Use default input/output paths without prompting (requires `--models`) |

/// admonition | What the generator produces
    type: note

The generated config is a starting point. It will include all input variables
required by the selected models, placeholder output sections, and any resampling
steps needed to bridge temporal frequencies. You should review and adjust the
generated file — for example, to remove variables you do not have, or to add
custom module sections.
///
