---
title: Python API
icon: lucide/square-function
---

# Python API

Building and running a pipeline is conduit's job, and satterc adds no wrapper around it.
Everything `satterc run` does is available from Python through [conduit's API][conduit-api]:

```python
import conduit

report = conduit.run("config.toml")
```

What satterc supplies is the modules that config points at, and the scaffolding for getting to a config in the first place.

## Models

A model module is a set of plain functions whose names are node names.
Conduit imports the module by the `_import_path` in the config, so nothing here is normally called by hand, but the functions are ordinary and can be.

| Module | Model |
| --- | --- |
| [`satterc.models.splash`](modules/satterc.models/splash.md) | [SPLASH](../models/splash.md): daily water balance, evapotranspiration and runoff. |
| [`satterc.models.pmodel`](modules/satterc.models/pmodel.md) | [The P-model](../models/pmodel.md): weekly photosynthesis, GPP and light-use efficiency. |
| [`satterc.models.sgam`](modules/satterc.models/sgam.md) | [SGAM](../models/sgam.md): weekly vegetation carbon pools and allocation. |
| [`satterc.models.rothc`](modules/satterc.models/rothc.md) | [RothC](../models/rothc.md): monthly soil carbon turnover. |

The [Models](../models/index.md) section covers the theory and the inputs and outputs of each.
These pages are the signatures.

## Scaffolding

`satterc setup` and `satterc data-gen` are thin wrappers over the functions below, so a config and a set of synthetic inputs can be produced from a script or a notebook as easily as from the terminal.

| Name | Does |
| --- | --- |
| [`get_builtin_models`](modules/satterc.scaffold/config_gen.md#satterc.scaffold.config_gen.get_builtin_models) | The model names `satterc setup` offers. |
| [`get_model_config`](modules/satterc.scaffold/config_gen.md#satterc.scaffold.config_gen.get_model_config) | The settings one model accepts, with their defaults and descriptions. |
| [`generate_config`](modules/satterc.scaffold/config_gen.md#satterc.scaffold.config_gen.generate_config) | A runnable config for a chosen set of models, inputs and outputs wired up. |
| [`generate_synthetic_data`](modules/satterc.scaffold/data_gen/generate.md#satterc.scaffold.data_gen.generate.generate_synthetic_data) | Input files for every `[inputs.*]` section a config names. |

### Modules

| Module | Contents |
| --- | --- |
| [`satterc.scaffold.config_gen`](modules/satterc.scaffold/config_gen.md) | Reading a model's signatures to work out what it needs, and writing the TOML that supplies it. |
| [`satterc.scaffold.bridges`](modules/satterc.scaffold/bridges.md) | Producer/consumer pairs whose units disagree, and the factor between them. |
| [`satterc.scaffold.data_gen.generate`](modules/satterc.scaffold/data_gen/generate.md) | Config in, input files out. |
| [`satterc.scaffold.data_gen.spec`](modules/satterc.scaffold/data_gen/spec.md) | What a variable's generator is handed: the grid, the calendar, and the values already computed. |
| [`satterc.scaffold.data_gen.daily`](modules/satterc.scaffold/data_gen/daily.md) | The daily variables with a hand-written generator. |
| [`satterc.scaffold.data_gen.static`](modules/satterc.scaffold/data_gen/static.md) | The same for time-invariant variables. |
| [`satterc.scaffold.data_gen.fallback`](modules/satterc.scaffold/data_gen/fallback.md) | What a variable gets when no table entry names it: a kind inferred from the name, and noise to match. |

## Shared

| Module | Contents |
| --- | --- |
| [`satterc.temporal`](modules/satterc.temporal.md) | The temporal resolutions a pipeline speaks in, and the pandas offsets behind them. |

[conduit-api]: https://nerc-ceh.github.io/conduit/reference/python-api.html
