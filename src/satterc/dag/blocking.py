"""Outer grid-level parallelism: in-DAG ``Parallelizable``/``Collect`` pixel blocking.

This module turns the existing (whole-grid) model DAG into a *blocked* one without
touching any model node. It generates a thin Hamilton module — mirroring
``dag/derive.py``'s ``types.ModuleType`` + attach pattern — that is prepended to the
configured modules when blocking is enabled:

- ``pixel_block`` (a ``Parallelizable[slice]``) yields a deterministic, fixed-size
  partition of the stacked ``pixel`` axis;
- one *slicing node* per pixel-bearing input ``X``: it depends on ``pixel_block`` and a
  renamed external input ``X__full`` and returns ``X__full.isel(pixel=pixel_block)``.
  Because the slicing node is *named* ``X``, it shadows what was previously the external
  input ``X``, so every downstream model node transparently receives the per-block slice
  (and thus fans out over blocks);
- a single *bundle/collect* pair: ``block_bundle`` packs all requested per-block outputs
  into one dict (so the whole region funnels through a single ``Collect`` — Hamilton
  allows only one ``Collect`` per ``Parallelizable``), ``grid_bundle`` gathers
  ``Collect[block_bundle]`` and concatenates each output along ``pixel``, and one tiny
  ``Y__grid`` extractor per requested output reads its grid back out of the bundle.

Each block carries its ``pixel`` coords, which ``caching.py``'s DataArray fingerprint
folds in, so per-block model outputs are content-keyed per block
(``notes/parallelism.md`` 7.4). The partition is a fixed ``block_size`` and therefore
worker-count-independent.

See ``notes/parallelism.md`` sections 7-8 for the full rationale.
"""

import sys
import types
import uuid
from typing import TYPE_CHECKING

import xarray as xr
from hamilton.execution.executors import (
    MultiThreadingExecutor,
    SynchronousLocalTaskExecutor,
)
from hamilton.htypes import Collect, Parallelizable

if TYPE_CHECKING:
    from hamilton import driver

    from satterc.config import BlockingSpec

GRID_SUFFIX = "__grid"
FULL_SUFFIX = "__full"
BLOCK_BUNDLE = "blocking_block_bundle"
GRID_BUNDLE = "blocking_grid_bundle"


def pixel_input_names(inputs: dict) -> list[str]:
    """Names in ``inputs`` whose value is a DataArray carrying a ``pixel`` dim."""
    return [
        name
        for name, val in inputs.items()
        if isinstance(val, xr.DataArray) and "pixel" in val.dims
    ]


def collected_name(node: str) -> str:
    """Hamilton node name of the ``Collect`` wrapper for output ``node``."""
    return f"{node}{GRID_SUFFIX}"


def rewrite_inputs_for_blocking(
    inputs: dict, spec: "BlockingSpec", pixel_inputs: list[str]
) -> dict:
    """Adapt a flat inputs dict for a blocked run.

    Renames every pixel-bearing input ``X`` -> ``X__full`` (the slicing node now owns
    the name ``X``) and injects the ``n_pixels``/``block_size`` the ``pixel_block``
    partition node needs. Non-pixel inputs (dates, scalars) are passed through unchanged
    and are global to every branch.
    """
    n_pixels = int(inputs[pixel_inputs[0]].sizes["pixel"])
    rewritten = dict(inputs)
    for name in pixel_inputs:
        rewritten[f"{name}{FULL_SUFFIX}"] = rewritten.pop(name)
    rewritten["n_pixels"] = n_pixels
    rewritten["block_size"] = spec.block_size
    return rewritten


def decollect_results(results: dict, output_nodes: list[str]) -> dict:
    """Map ``{Y__grid: ...}`` results back to their canonical ``{Y: ...}`` names."""
    out = dict(results)
    for node in output_nodes:
        grid = collected_name(node)
        if grid in out:
            out[node] = out.pop(grid)
    return out


def _make_pixel_block(module_name: str):
    # exec (not a closure) so __qualname__ == "pixel_block" and the function resolves
    # by reference via the generated module (keeps task executors that serialize by
    # reference, e.g. threading, working uniformly).
    src = (
        "def pixel_block(n_pixels, block_size):\n"
        "    for start in range(0, n_pixels, block_size):\n"
        "        yield slice(start, min(start + block_size, n_pixels))\n"
    )
    ns: dict = {}
    exec(src, ns)
    fn = ns["pixel_block"]
    fn.__annotations__ = {
        "n_pixels": int,
        "block_size": int,
        "return": Parallelizable[slice],  # type: ignore[index]
    }
    fn.__module__ = module_name
    return fn


def _make_slice_node(name: str, module_name: str):
    full = f"{name}{FULL_SUFFIX}"
    src = (
        f"def {name}(pixel_block, {full}):\n    return {full}.isel(pixel=pixel_block)\n"
    )
    ns: dict = {}
    exec(src, ns)
    fn = ns[name]
    fn.__annotations__ = {
        "pixel_block": slice,
        full: xr.DataArray,
        "return": xr.DataArray,
    }
    fn.__module__ = module_name
    return fn


def _make_block_bundle(output_nodes: list[str], module_name: str):
    # Pack all per-block outputs into one dict, so the whole parallel region funnels
    # through a single Collect (Hamilton allows only one Collect per Parallelizable).
    params = ", ".join(output_nodes)
    items = ", ".join(f'"{out}": {out}' for out in output_nodes)
    src = f"def {BLOCK_BUNDLE}({params}):\n    return {{{items}}}\n"
    ns: dict = {}
    exec(src, ns)
    fn = ns[BLOCK_BUNDLE]
    fn.__annotations__ = {out: xr.DataArray for out in output_nodes} | {"return": dict}
    fn.__module__ = module_name
    return fn


def _make_grid_bundle(module_name: str):
    src = (
        f"def {GRID_BUNDLE}({BLOCK_BUNDLE}):\n"
        f"    blocks = list({BLOCK_BUNDLE})\n"
        f"    return {{\n"
        f"        n: xr.concat([b[n] for b in blocks], dim='pixel').sortby('pixel')\n"
        f"        for n in blocks[0]\n"
        f"    }}\n"
    )
    ns: dict = {"xr": xr}
    exec(src, ns)
    fn = ns[GRID_BUNDLE]
    fn.__annotations__ = {
        BLOCK_BUNDLE: Collect[dict],  # type: ignore[index]
        "return": dict,
    }
    fn.__module__ = module_name
    return fn


def _make_grid_extractor(name: str, module_name: str):
    grid = collected_name(name)
    src = f'def {grid}({GRID_BUNDLE}):\n    return {GRID_BUNDLE}["{name}"]\n'
    ns: dict = {}
    exec(src, ns)
    fn = ns[grid]
    fn.__annotations__ = {GRID_BUNDLE: dict, "return": xr.DataArray}
    fn.__module__ = module_name
    return fn


def make_blocking_module(
    pixel_inputs: list[str], output_nodes: list[str]
) -> types.ModuleType:
    """Generate the Hamilton blocking module for a given input/output signature.

    Parameters
    ----------
    pixel_inputs
        Names of the pixel-bearing external inputs to slice per block.
    output_nodes
        Canonical names of the requested output nodes to collect back into grids.
    """
    mod = types.ModuleType(f"satterc_blocking_generated_{uuid.uuid4().hex[:8]}")
    name = mod.__name__

    setattr(mod, "pixel_block", _make_pixel_block(name))  # noqa: B010
    for inp in pixel_inputs:
        setattr(mod, inp, _make_slice_node(inp, name))

    setattr(mod, BLOCK_BUNDLE, _make_block_bundle(output_nodes, name))
    setattr(mod, GRID_BUNDLE, _make_grid_bundle(name))
    for out in output_nodes:
        setattr(mod, collected_name(out), _make_grid_extractor(out, name))

    sys.modules[name] = mod
    return mod


def apply_blocking(builder: "driver.Builder", spec: "BlockingSpec") -> "driver.Builder":
    """Enable dynamic (parallel) execution on a Builder according to a BlockingSpec."""
    builder = builder.enable_dynamic_execution(allow_experimental_mode=True)
    if spec.executor == "threading":
        max_tasks = spec.max_tasks if spec.max_tasks is not None else 4
        return builder.with_remote_executor(MultiThreadingExecutor(max_tasks=max_tasks))
    return builder.with_remote_executor(SynchronousLocalTaskExecutor())
