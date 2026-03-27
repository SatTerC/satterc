from typing import Any

from hamilton import driver
from hamilton.settings import ENABLE_POWER_USER_MODE

from .pipeline import inputs, outputs, models, resample


def get_model_modules(module_names: list[str]) -> list:
    return [getattr(models, name) for name in module_names]


def build_driver(
    modules: list[str],
    config: dict[str, Any] | None = None,
    allow_module_overrides: bool = False,
) -> driver.Driver:
    config = dict(config) if config else {}
    config[ENABLE_POWER_USER_MODE] = True

    modules_ = []

    if "inputs.daily" in modules:
        modules_.append(inputs.daily)
    if "inputs.weekly" in modules:
        modules_.append(inputs.weekly)
    if "inputs.monthly" in modules:
        modules_.append(inputs.monthly)
    if "inputs.static" in modules:
        modules_.append(inputs.static)

    input_freqs = [
        f for f in ["daily", "weekly", "monthly", "static"] if f"inputs.{f}" in modules
    ]
    if input_freqs:
        modules_.append(inputs.grid)

    if "outputs.daily" in modules:
        modules_.append(outputs.daily)
    if "outputs.weekly" in modules:
        modules_.append(outputs.weekly)
    if "outputs.monthly" in modules:
        modules_.append(outputs.monthly)
    if "outputs.static" in modules:
        modules_.append(outputs.static)

    if "resample" in modules:
        modules_.append(resample)

    model_names = [
        m.removeprefix("models.")
        for m in modules
        if not m.startswith("inputs.")
        and not m.startswith("outputs.")
        and m != "resample"
    ]
    modules_ += get_model_modules(model_names)

    dr = driver.Builder().with_modules(*modules_).with_config(config)
    if allow_module_overrides:
        dr = dr.allow_module_overrides()
    return dr.build()
