from typing import Any

from hamilton import driver
from hamilton.settings import ENABLE_POWER_USER_MODE

from .pipeline import inputs, outputs, models, resample


def get_model_modules(module_names: list[str]) -> list:
    # TODO: support custom modules / models
    return [getattr(models, name) for name in module_names]


def build_driver(
    modules: list[str],
    config: dict[str, Any] | None = None,
    allow_module_overrides: bool = False,
) -> driver.Driver:
    config = dict(config) if config else {}
    config[ENABLE_POWER_USER_MODE] = True

    modules_ = [
        inputs.daily,
        inputs.weekly,
        # inputs.monthly,
        inputs.static,
        inputs.grid,
        outputs.daily,
        outputs.weekly,
        outputs.monthly,
        resample,
    ] + get_model_modules(modules)

    dr = driver.Builder().with_modules(*modules_).with_config(config)
    if allow_module_overrides:
        dr = dr.allow_module_overrides()
    return dr.build()
