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

    modules_ = [MODULES[mod] for mod in modules]

    # TODO: fix this
    modules_ += [MODULES["inputs.grid"]]

    dr = driver.Builder().with_modules(*modules_).with_config(config)

    if allow_module_overrides:
        dr = dr.allow_module_overrides()

    return dr.build()
