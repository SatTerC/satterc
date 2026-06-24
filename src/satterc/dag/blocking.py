"""Blocked driver execution for pixel-dimension parallelism (Mechanism B).

Partitions the stacked ``pixel`` dimension into fixed-size blocks and calls
``dr.execute`` per block, then concatenates results along ``pixel``.  Peak
memory is bounded to a small multiple of one block's footprint.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import xarray as xr

if TYPE_CHECKING:
    from hamilton import driver

    from satterc.config import BlockingSpec, CacheSpec


def _pixel_input_names(inputs: dict[str, Any]) -> list[str]:
    """Return the names of inputs that have a ``pixel`` dimension."""
    return [
        name
        for name, val in inputs.items()
        if isinstance(val, xr.DataArray) and "pixel" in val.dims
    ]


def _make_blocks(
    inputs: dict[str, Any],
    pixel_names: list[str],
    block_size: int,
) -> Generator[dict[str, Any]]:
    """Yield sliced input dicts, one per pixel block.

    Partition is deterministic and fixed-size (independent of worker count)
    so cache keys are stable across machines and restarts.
    """
    if not pixel_names:
        yield inputs
        return

    n_pixels = inputs[pixel_names[0]].sizes["pixel"]
    for start in range(0, n_pixels, block_size):
        sl = slice(start, start + block_size)
        yield {
            name: val.isel(pixel=sl) if name in pixel_names else val
            for name, val in inputs.items()
        }


def _concat_results(
    block_results: list[dict[str, Any]],
    final_vars: list[str],
) -> dict[str, Any]:
    """Concatenate per-block results along the ``pixel`` dimension."""
    return {
        var: xr.concat([r[var] for r in block_results], dim="pixel")
        for var in final_vars
    }


def _mp_worker(
    args: tuple[list[str], dict[str, Any], Any, dict[str, Any], list[str]],
) -> dict[str, Any]:
    """Per-process worker: rebuilds the driver and executes one block.

    Must be top-level (not a closure) to be picklable across processes.
    """
    from satterc.dag.driver import build_driver

    modules, config, cache_spec, block_inputs, final_vars = args
    dr = build_driver(modules, config, cache=cache_spec)
    return dr.execute(final_vars, inputs=block_inputs)  # type: ignore[reportArgumentType]


def execute_blocked(
    dr: driver.Driver,
    inputs: dict[str, Any],
    final_vars: list[str],
    spec: BlockingSpec,
    *,
    build_params: tuple[list[str], dict[str, Any], CacheSpec | None] | None = None,
) -> dict[str, Any]:
    """Execute the driver in pixel blocks and concatenate results.

    Parameters
    ----------
    dr
        A built Hamilton driver.  Shared directly for ``synchronous``/
        ``threading``; the ``build_params`` tuple is used to rebuild one
        driver per worker for ``multiprocessing``.
    inputs
        Full-grid input dict as returned by ``load_inputs``.
    final_vars
        Node names to compute, as returned by ``get_final_vars``.
    spec
        Blocking configuration (``block_size``, ``executor``, ``max_workers``).
    build_params
        Required when ``spec.executor == "multiprocessing"``.  A
        ``(modules, config, cache_spec)`` tuple forwarded to ``build_driver``
        inside each worker process.
    """
    pixel_names = _pixel_input_names(inputs)
    blocks = list(_make_blocks(inputs, pixel_names, spec.block_size))

    if spec.executor == "synchronous":
        results = [
            dr.execute(final_vars, inputs=block_inputs)  # type: ignore[reportArgumentType]
            for block_inputs in blocks
        ]

    elif spec.executor == "threading":

        def _run(block_inputs: dict[str, Any]) -> dict[str, Any]:
            return dr.execute(final_vars, inputs=block_inputs)  # type: ignore[reportArgumentType]

        with ThreadPoolExecutor(max_workers=spec.max_workers) as pool:
            results = list(pool.map(_run, blocks))

    else:  # multiprocessing
        if build_params is None:
            raise ValueError(
                "execute_blocked requires 'build_params' "
                "when executor='multiprocessing'."
            )
        modules, config, cache_spec = build_params
        mp_args = [
            (modules, config, cache_spec, block_inputs, final_vars)
            for block_inputs in blocks
        ]
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=spec.max_workers, mp_context=ctx) as pool:
            results = list(pool.map(_mp_worker, mp_args))

    return _concat_results(results, final_vars)
