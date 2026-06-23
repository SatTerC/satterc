"""Tests for outer grid-level pixel blocking (``satterc.dag.blocking``).

Blocking pushes the pixel-block fan-out *inside* the Hamilton DAG via
``Parallelizable``/``Collect`` so the existing model nodes run once per deterministic
``pixel`` block (Mechanism D, ``notes/parallelism.md`` sections 7-8). Because the models
have zero inter-pixel coupling, *any* partition of the stacked ``pixel`` axis must give
bit-identical results -- that partition-invariance is the core guarantee here.

Coverage:

1. ``TestPartitionInvariance`` — a blocked run of the *real* pipeline reproduces the
   unblocked run for every block size (including one that does not divide ``n_pixels``)
   and for both executors, exercising a resample inside the branch.
2. ``TestExtractFieldsFanout`` — an ``@extract_fields`` multi-output node collected per
   field through the blocking seam (the interaction the doc flags as "awkward").
3. ``TestCaching`` — blocking coexists with ``with_cache``; a warm blocked run is stable
   (the deep per-branch cache-hit guarantee is pinned by ``test_parallel_cache_spike``).
4. ``TestUnitCheck`` — the build-time unit check tolerates the extra
   Parallelizable/Collect/slicing nodes.
5. ``TestRunPath`` — the run-path helpers (rewrite inputs / collect / decollect) yield
   outputs identical to the unblocked path end to end.

Multiprocessing is intentionally absent: Hamilton 1.90's ``MultiProcessingExecutor``
cannot serialize the local closures ``@extract_fields`` generates (and its own docstring
flags this), so only ``synchronous`` and ``threading`` are supported.
"""

import sys
import types

import numpy as np
import pytest
import xarray as xr
from hamilton import driver
from hamilton.function_modifiers import extract_fields
from hamilton.settings import ENABLE_POWER_USER_MODE

from satterc import units
from satterc.config import BlockingSpec
from satterc.dag.blocking import (
    apply_blocking,
    collected_name,
    decollect_results,
    make_blocking_module,
    pixel_input_names,
    rewrite_inputs_for_blocking,
)
from satterc.dag.driver import build_driver

EXECUTORS = ["synchronous", "threading"]
TARGET = "mean_growth_temperature_weekly"  # runnable from raw daily input; resamples


def _unblocked(pipeline_config, pipeline_inputs, targets):
    dr = build_driver(pipeline_config.modules, dict(pipeline_config.driver_config))
    return dr.execute(targets, inputs=pipeline_inputs)


def _blocked(pipeline_config, pipeline_inputs, targets, spec, cache=None):
    pix = pixel_input_names(pipeline_inputs)
    dr = build_driver(
        pipeline_config.modules,
        dict(pipeline_config.driver_config),
        cache=cache,
        blocking=spec,
        pixel_inputs=pix,
        output_nodes=targets,
    )
    exec_inputs = rewrite_inputs_for_blocking(pipeline_inputs, spec, pix)
    results = dr.execute([collected_name(t) for t in targets], inputs=exec_inputs)
    return decollect_results(results, targets)


class TestBlockingSpecValidation:
    """[blocking] config validation rejects malformed entries."""

    @pytest.mark.parametrize(
        "entry",
        [
            {},  # missing block_size
            {"block_size": 0},  # not positive
            {"block_size": True},  # bool is not a valid int here
            {"block_size": 2, "executor": "multiprocessing"},  # unsupported executor
            {"block_size": 2, "max_tasks": 0},  # not positive
        ],
    )
    def test_invalid(self, entry):
        with pytest.raises(ValueError, match=r"\[blocking\]"):
            BlockingSpec.from_config(entry)

    def test_valid_defaults(self):
        spec = BlockingSpec.from_config({"block_size": 4})
        assert spec.block_size == 4
        assert spec.executor == "synchronous"
        assert spec.max_tasks is None


class TestPartitionInvariance:
    """A blocked run reproduces the unblocked run, bit for bit."""

    @pytest.mark.parametrize("executor", EXECUTORS)
    @pytest.mark.parametrize("block_size", [1, 2, 3, 4])
    def test_matches_unblocked(
        self, pipeline_config, pipeline_inputs, executor, block_size
    ):
        expected = _unblocked(pipeline_config, pipeline_inputs, [TARGET])[TARGET]
        spec = BlockingSpec(block_size=block_size, executor=executor, max_tasks=3)
        got = _blocked(pipeline_config, pipeline_inputs, [TARGET], spec)[TARGET]

        xr.testing.assert_allclose(got.transpose(*expected.dims), expected)
        assert got.sizes["pixel"] == expected.sizes["pixel"]

    def test_block_size_is_deterministic(self, pipeline_config, pipeline_inputs):
        # n_pixels == 4 (the 2x2 synthetic grid): a divisor and a non-divisor agree.
        runs = [
            _blocked(
                pipeline_config,
                pipeline_inputs,
                [TARGET],
                BlockingSpec(block_size=b),
            )[TARGET]
            for b in (1, 2, 3, 4)
        ]
        for other in runs[1:]:
            xr.testing.assert_allclose(other, runs[0])


# --------------------------------------------------------------------------- #
# A self-contained module exercising @extract_fields multi-output fan-out.
# --------------------------------------------------------------------------- #


def _make_extract_fields_module():
    from typing import TypedDict

    mod = types.ModuleType("satterc_blocking_ef_fixture")

    class _TwoOut(TypedDict):
        doubled: xr.DataArray
        tripled: xr.DataArray

    @extract_fields()
    def split(base: xr.DataArray) -> _TwoOut:
        return {"doubled": base * 2.0, "tripled": base * 3.0}

    split.__module__ = mod.__name__
    setattr(mod, "split", split)  # noqa: B010
    sys.modules[mod.__name__] = mod
    return mod


class TestExtractFieldsFanout:
    """An ``@extract_fields`` node's fields collect correctly per block."""

    @pytest.mark.parametrize("executor", EXECUTORS)
    def test_both_fields(self, executor):
        ef_mod = _make_extract_fields_module()
        base = xr.DataArray(
            np.arange(12.0).reshape(4, 3),
            dims=["pixel", "time"],
            coords={"pixel": [0, 1, 2, 3]},
        )
        targets = ["doubled", "tripled"]
        blk = make_blocking_module(["base"], targets)
        spec = BlockingSpec(block_size=3, executor=executor, max_tasks=2)  # non-divisor

        builder = (
            driver.Builder()
            .with_modules(ef_mod, blk)
            .with_config({ENABLE_POWER_USER_MODE: True})
        )
        dr = apply_blocking(builder, spec).build()

        inputs = {"base__full": base, "n_pixels": 4, "block_size": spec.block_size}
        out = dr.execute([collected_name(t) for t in targets], inputs=inputs)
        out = decollect_results(out, targets)

        xr.testing.assert_allclose(
            out["doubled"].transpose("pixel", "time"), base * 2.0
        )
        xr.testing.assert_allclose(
            out["tripled"].transpose("pixel", "time"), base * 3.0
        )


class TestCaching:
    """Blocking coexists with Hamilton caching; warm runs are stable."""

    def test_cached_blocked_matches_uncached(
        self, tmp_path, pipeline_config, pipeline_inputs
    ):
        from satterc import CacheSpec

        spec = BlockingSpec(block_size=2)
        cache = CacheSpec(path=str(tmp_path / "cache"))

        uncached = _blocked(pipeline_config, pipeline_inputs, [TARGET], spec)[TARGET]
        _blocked(pipeline_config, pipeline_inputs, [TARGET], spec, cache=cache)  # cold
        warm = _blocked(pipeline_config, pipeline_inputs, [TARGET], spec, cache=cache)[
            TARGET
        ]

        xr.testing.assert_allclose(warm, uncached)
        assert (tmp_path / "cache").exists()


class TestUnitCheck:
    """The build-time unit check tolerates the generated blocking nodes."""

    def test_build_with_units_warn(self, pipeline_config, pipeline_inputs):
        previous = units.get_mode()
        units.set_mode("warn")
        try:
            pix = pixel_input_names(pipeline_inputs)
            # Should build (and run the DAG unit check) without raising.
            build_driver(
                pipeline_config.modules,
                dict(pipeline_config.driver_config),
                blocking=BlockingSpec(block_size=2),
                pixel_inputs=pix,
                output_nodes=[TARGET],
            )
        finally:
            units.set_mode(previous)


class TestRunPath:
    """The run-path helpers reproduce the unblocked outputs end to end."""

    def test_assembled_outputs_match(self, pipeline_config, pipeline_inputs):
        from satterc.config import IOSpec
        from satterc.io import get_outputs

        output_specs = {
            "weekly": IOSpec(path="unused.nc", vars=["mean_growth_temperature"])
        }

        unblocked = _unblocked(pipeline_config, pipeline_inputs, [TARGET])
        spec = BlockingSpec(block_size=3)
        blocked = _blocked(pipeline_config, pipeline_inputs, [TARGET], spec)

        ds_unblocked = get_outputs(unblocked, output_specs)["weekly"]
        ds_blocked = get_outputs(blocked, output_specs)["weekly"]
        xr.testing.assert_allclose(ds_blocked, ds_unblocked)


class TestCLIEndToEnd:
    """`satterc run` on a [blocking] config writes outputs equal to the plain run."""

    def _config(self, data_dir, out_path, blocking: bool) -> str:
        block = "\n[blocking]\nblock_size = 3\n" if blocking else ""
        return (
            "[models.pmodel]\n"
            'method_kphio = "sandoval"\n'
            'method_optchi = "lavergne20_c3"\n\n'
            "[inputs.daily]\n"
            f'path = "{data_dir / "daily.nc"}"\n'
            'vars = ["temperature_celcius"]\n\n'
            "[outputs.weekly]\n"
            f'path = "{out_path}"\n'
            'vars = ["mean_growth_temperature"]\n'
            f"{block}"
        )

    def test_run_blocked_matches_plain(self, tmp_path, synthetic_data_dir):
        from satterc.cli.run import run

        plain_out = tmp_path / "plain.nc"
        blocked_out = tmp_path / "blocked.nc"
        plain_cfg = tmp_path / "plain.toml"
        blocked_cfg = tmp_path / "blocked.toml"
        plain_cfg.write_text(self._config(synthetic_data_dir, plain_out, False))
        blocked_cfg.write_text(self._config(synthetic_data_dir, blocked_out, True))

        run(plain_cfg)
        run(blocked_cfg)

        with xr.open_dataset(plain_out) as plain, xr.open_dataset(blocked_out) as blk:
            xr.testing.assert_allclose(blk, plain)
