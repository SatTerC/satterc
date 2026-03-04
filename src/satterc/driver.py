from types import ModuleType
from typing import Any

from hamilton import driver

from .models import splash
from .models import pmodel

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
)


def get_modules(modules: list[str] | None = None) -> list[ModuleType]:
    # TODO: this can call private functions such as _get_all_modules_from_registry, _get_module_from_registry(),
    # _load_module_from_path etc.
    # For now, just attempt to extract from dict
    if not modules:
        modules = ["splash", "pmodel"]
    return [_MODULES[key] for key in modules]


def build_driver(
    modules: list[str] | None = None, config: dict[str, Any] | None = None
) -> driver.Driver:
    modules_ = get_modules(modules)
    return driver.Builder().with_modules(*modules_).with_config(config or {}).build()
