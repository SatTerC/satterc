from typing import Any

from hamilton import driver
from hamilton.settings import ENABLE_POWER_USER_MODE

from .inputs import grid, daily, weekly, monthly, static
from .dynamic import unpack, aggregate
from .outputs import daily as daily_out, weekly as weekly_out, monthly as monthly_out
from .models import splash, pmodel, sgam, rothc


_MODULES = dict(
    splash=splash,
    pmodel=pmodel,
    sgam=sgam,
    rothc=rothc,
)


def get_model_modules(module_names: list[str]) -> list:
    return [_MODULES[name] for name in module_names]


def build_driver(
    modules: list[str],
    config: dict[str, Any] | None = None,
    allow_module_overrides: bool = False,
) -> driver.Driver:
    config = dict(config) if config else {}
    config[ENABLE_POWER_USER_MODE] = True

    modules_ = [
        grid,
        daily,
        weekly,
        monthly,
        static,
        unpack,
        aggregate,
        daily_out,
        weekly_out,
        monthly_out,
    ] + get_model_modules(modules)

    dr = driver.Builder().with_modules(*modules_).with_config(config)
    if allow_module_overrides:
        dr = dr.allow_module_overrides()
    return dr.build()
