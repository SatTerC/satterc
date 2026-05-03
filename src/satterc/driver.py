from importlib import import_module
from typing import Any

from hamilton import driver
from hamilton.settings import ENABLE_POWER_USER_MODE

from .pipeline import inputs, outputs, models, resample

MODULES = {
    "inputs.daily": inputs.daily,
    "inputs.weekly": inputs.weekly,
    "inputs.monthly": inputs.monthly,
    "inputs.static": inputs.static,
    "inputs.grid": inputs.grid,
    "outputs.daily": outputs.daily,
    "outputs.weekly": outputs.weekly,
    "outputs.monthly": outputs.monthly,
    "outputs.static": outputs.static,
    "resample": resample,
    "models.splash": models.splash,
    "models.pmodel": models.pmodel,
    "models.sgam": models.sgam,
    "models.rothc": models.rothc,
}


def build_driver(
    modules: list[str],
    config: dict[str, Any],
    allow_module_overrides: bool = False,
) -> driver.Driver:
    config[ENABLE_POWER_USER_MODE] = True

    modules_ = []
    for mod in modules:
        if mod in MODULES:
            modules_.append(MODULES[mod])
        else:
            if mod.startswith("models."):
                known = sorted(m for m in MODULES if m.startswith("models."))
                raise ValueError(f"Unknown model '{mod}'. Known models: {known}")
            try:
                modules_.append(import_module(mod))
            except ModuleNotFoundError as exc:
                raise ValueError(
                    f"Cannot load module '{mod}': not a known satterc module "
                    f"and not importable as a Python module."
                ) from exc

    dr = driver.Builder().with_modules(*modules_).with_config(config)

    if allow_module_overrides:
        dr = dr.allow_module_overrides()

    return dr.build()
