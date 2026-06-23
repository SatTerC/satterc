"""Build Hamilton drivers from configured module lists."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from hamilton import driver
from hamilton.settings import ENABLE_POWER_USER_MODE

if TYPE_CHECKING:
    from satterc.config import BlockingSpec, CacheSpec

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
    cache: "CacheSpec | None" = None,
    blocking: "BlockingSpec | None" = None,
    pixel_inputs: list[str] | None = None,
    output_nodes: list[str] | None = None,
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
    cache
        If provided, enable Hamilton result caching according to this spec.
    blocking
        If provided, enable outer grid-level pixel blocking: a generated
        ``Parallelizable``/``Collect`` module is prepended and dynamic execution
        is enabled. ``pixel_inputs`` and ``output_nodes`` must also be supplied.
    pixel_inputs
        Names of the pixel-bearing external inputs to slice per block (only used
        when ``blocking`` is set).
    output_nodes
        Canonical names of the requested output nodes to collect back into grids
        (only used when ``blocking`` is set).

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

    if blocking is not None:
        from satterc.dag.blocking import make_blocking_module

        if pixel_inputs is None or output_nodes is None:
            raise ValueError(
                "build_driver(blocking=...) requires 'pixel_inputs' and "
                "'output_nodes' to generate the blocking module."
            )
        modules_.append(make_blocking_module(pixel_inputs, output_nodes))

    dr = driver.Builder().with_modules(*modules_).with_config(config)

    if allow_module_overrides:
        dr = dr.allow_module_overrides()

    if cache is not None:
        from satterc.dag.caching import apply_cache

        dr = apply_cache(dr, cache)

    if blocking is not None:
        from satterc.dag.blocking import apply_blocking

        dr = apply_blocking(dr, blocking)

    built = dr.build()

    # Build-time unit-consistency check; a no-op in "off" mode (the conftest
    # default), so this does not affect builds that opt out of unit handling.
    from satterc.dag.unit_check import check_dag_units

    check_dag_units(built)

    return built
