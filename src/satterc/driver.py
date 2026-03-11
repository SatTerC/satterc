from types import ModuleType
from typing import Any

from hamilton import driver

from .models import splash, pmodel, sgam, rothc
from .inputs import grid, daily, weekly, monthly, static
from . import outputs
# from .extras import synthetic_inputs

# TODO:
# * Create a module registry.
# * Create a function that extracts a module or list of modules from the registry by key
# * Create a function that extracts a list of all modules in the registry
# * Allow users to provide a list of modules from the configuration file, which can include (i)
#   keys corresponding to modules from the registry, (ii) special value '*' or 'all' which loads
#   all registry modules, (iii) a path to their custom modules, which will be loaded dynamically
#   using importlib. These will be loaded last, after any built-in/registry modules.
# * Possibly future: function to let users register their custom modules to the registry.

_MODULES = dict(
    splash=splash,
    pmodel=pmodel,
    sgam=sgam,
    rothc=rothc,
    # synthetic_inputs=synthetic_inputs,
)


def get_modules(modules: list[str] | None = None) -> list[ModuleType]:
    # TODO: this can call private functions such as _get_all_modules_from_registry, _get_module_from_registry(),
    # _load_module_from_path etc.
    # For now, just attempt to extract from dict
    if not modules:
        modules = list(_MODULES.keys())
    return [_MODULES[key] for key in modules]


def build_driver(
    modules: list[str] | None = None,
    config: dict[str, Any] | None = None,
    allow_module_overrides: bool = False,
) -> driver.Driver:
    modules_ = [grid, daily, weekly, monthly, static, outputs] + get_modules(modules)
    dr = driver.Builder().with_modules(*modules_).with_config(config or {})
    if allow_module_overrides:
        dr = dr.allow_module_overrides()
    return dr.build()
