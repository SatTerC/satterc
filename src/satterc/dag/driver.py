"""Build Hamilton drivers from configured module lists."""

from importlib import import_module
from typing import Any

from hamilton import driver
from hamilton.settings import ENABLE_POWER_USER_MODE

MODULES: dict[str, str] = {
    "derive": "satterc.dag.derive",
    "resample": "satterc.dag.resample",
    "models.splash": "satterc.dag.splash",
    "models.pmodel": "satterc.dag.pmodel",
    "models.sgam": "satterc.dag.sgam",
    "models.rothc": "satterc.dag.rothc",
}


def build_driver(
    modules: list[str],
    config: dict[str, Any],
    allow_module_overrides: bool = False,
) -> driver.Driver:
    """Build a Hamilton driver from a list of module names and config.

    Parameters
    ----------
    modules
        List of module identifiers (e.g., "models.pmodel", "resample").
    config
        Configuration dict passed to the Hamilton driver.
    allow_module_overrides
        If True, allow later modules to override earlier ones.

    Returns
    -------
    driver.Driver
        A configured Hamilton driver ready for execution.
    """
    config[ENABLE_POWER_USER_MODE] = True

    from satterc.dag.derive import make_derive_module

    modules_ = []
    for mod in modules:
        if mod == "derive":
            modules_.append(make_derive_module(config.get("derive_specs", [])))
        elif mod in MODULES:
            modules_.append(import_module(MODULES[mod]))
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
