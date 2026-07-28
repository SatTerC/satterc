"""Tests for satterc.setup_utils.data_gen.spec — the table-driven machinery."""

import ast
import inspect
from itertools import pairwise

import numpy as np
import pytest

from satterc.setup_utils.data_gen import daily, static
from satterc.setup_utils.data_gen.daily import DAILY_VARS
from satterc.setup_utils.data_gen.fallback import fallback_var
from satterc.setup_utils.data_gen.spec import (
    Grid,
    Resolver,
    StaticCtx,
    Var,
    collect_vars,
)
from satterc.setup_utils.data_gen.static import STATIC_VARS

N_DAYS = 40
N_LAT, N_LON = 2, 3
N_PIXELS = N_LAT * N_LON


def _documented_names(module) -> set[str]:
    """Module-level names followed by a string literal, i.e. an attribute docstring."""
    body = ast.parse(inspect.getsource(module)).body
    return {
        node.targets[0].id
        for node, following in pairwise(body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(following, ast.Expr)
        and isinstance(following.value, ast.Constant)
        and isinstance(following.value.value, str)
    }


def make_resolver(daily_vars=None, static_vars=None, seed=0) -> Resolver:
    return Resolver(
        grid=Grid(n_lat=N_LAT, n_lon=N_LON, n_days=N_DAYS),
        daily_vars=DAILY_VARS if daily_vars is None else daily_vars,
        static_vars=STATIC_VARS if static_vars is None else static_vars,
        fallback=fallback_var,
        seed=seed,
    )


@pytest.fixture
def resolver():
    return make_resolver()


class TestGrid:
    def test_pixels_are_the_lat_lon_product(self):
        grid = Grid(n_lat=N_LAT, n_lon=N_LON, n_days=N_DAYS)
        assert grid.n_pixels == N_PIXELS
        assert len(grid.pixel_coords) == N_PIXELS

    def test_time_coord_length(self):
        grid = Grid(n_lat=1, n_lon=1, n_days=N_DAYS)
        assert len(grid.time_coord) == N_DAYS


def _static_ctx(grid: Grid) -> StaticCtx:
    """A bare context, for exercising the geometry helpers without a table."""
    return StaticCtx(Resolver(grid, {}, {}, fallback_var), np.random.default_rng(0))


class TestGridGeometry:
    """The box and start date are arguments, not constants."""

    def test_lat_lon_ranges_are_honoured(self):
        grid = Grid(2, 3, 10, lat_range=(-45.0, -40.0), lon_range=(170.0, 175.0))
        assert grid.lat.min() == -45.0
        assert grid.lat.max() == -40.0
        assert grid.lon.min() == 170.0
        assert grid.lon.max() == 175.0

    def test_start_date_is_honoured(self):
        grid = Grid(1, 1, 10, start_date="2016-02-25")
        assert str(grid.time_coord[0]) == "2016-02-25"
        assert str(grid.time_coord[5]) == "2016-03-01"  # 2016 is a leap year

    def test_norm_spans_the_grid_wherever_it_sits(self):
        ctx = _static_ctx(
            Grid(3, 3, 10, lat_range=(-5.0, 5.0), lon_range=(100.0, 110.0))
        )
        assert ctx.lat_norm.min() == 0.0
        assert ctx.lat_norm.max() == 1.0
        assert ctx.lon_norm.min() == 0.0
        assert ctx.lon_norm.max() == 1.0

    def test_norm_of_a_single_pixel_does_not_divide_by_zero(self):
        ctx = _static_ctx(Grid(1, 1, 10))
        assert ctx.lat_norm.tolist() == [0.0]
        assert ctx.lon_norm.tolist() == [0.0]


class TestClimateFollowsTheBox:
    """Moving the grid must move the climate, or a bbox option is a trap."""

    def _monthly_mean(self, lat_range) -> np.ndarray:
        resolver = Resolver(
            Grid(2, 2, 730, lat_range=lat_range, lon_range=(0.0, 5.0)),
            DAILY_VARS,
            STATIC_VARS,
            fallback_var,
            seed=1,
        )
        temperature = resolver.daily("temperature")
        return temperature.mean("pixel").groupby("time.month").mean().values

    def test_tropics_are_warm_and_aseasonal(self):
        monthly = self._monthly_mean((-5.0, 5.0))
        assert monthly.mean() > 20.0
        assert monthly.max() - monthly.min() < 3.0

    def test_northern_midlatitudes_peak_in_summer(self):
        monthly = self._monthly_mean((50.0, 54.0))
        assert 5.0 < monthly.mean() < 15.0
        assert monthly.argmax() + 1 in (6, 7, 8)

    def test_southern_midlatitudes_peak_in_december(self):
        monthly = self._monthly_mean((-45.0, -40.0))
        assert monthly.argmax() + 1 in (12, 1, 2)


class TestBuild:
    """Shape, bounds and dtype are the resolver's job, not the lambda's."""

    def test_scalar_is_broadcast_to_full_shape(self):
        table = {"const": Var("1", "constant", lambda _: 3.0)}
        result = make_resolver(daily_vars=table).daily("const")
        assert result.shape == (N_DAYS, N_PIXELS)
        assert np.all(result.values == 3.0)

    def test_column_is_broadcast_across_pixels(self):
        table = {"uniform_in_space": Var("1", "", lambda g: g.day)}
        result = make_resolver(daily_vars=table).daily("uniform_in_space")
        assert result.shape == (N_DAYS, N_PIXELS)
        assert np.all(result.values[5, :] == 5.0)

    def test_bounds_are_applied(self):
        table = {"clipped": Var("1", "", lambda g: g.normal(0.0, 10.0), (0.0, 1.0))}
        values = make_resolver(daily_vars=table).daily("clipped").values
        assert values.min() >= 0.0
        assert values.max() <= 1.0

    def test_dtype_is_applied(self):
        table = {"label": Var("1", "", lambda g: g.uniform(1.0, 3.0), dtype=np.int32)}
        result = make_resolver(static_vars=table).static("label")
        assert result.values.dtype == np.int32

    def test_values_are_writable(self, resolver):
        """Broadcasting yields a read-only view; the resolver must copy."""
        result = resolver.daily("temperature")
        result.values[0, 0] = 0.0  # must not raise

    def test_units_and_long_name_become_attrs(self, resolver):
        result = resolver.daily("temperature")
        assert result.attrs["units"] == "degC"
        assert result.attrs["long_name"] == "air temperature"


class TestDependencies:
    def test_dependency_is_resolved(self, resolver):
        """`pressure` reads `elevation`, a static, from within a daily lambda."""
        pressure = resolver.daily("pressure")
        elevation = resolver.static("elevation")
        assert pressure.shape == (N_DAYS, N_PIXELS)
        # Pressure falls with elevation, so the two are strongly anticorrelated
        # across pixels. (Per-pixel ranking is not safe: synoptic noise can swap
        # neighbouring pixels whose elevations are close.)
        correlation = np.corrcoef(pressure.mean("time").values, elevation.values)[0, 1]
        assert correlation < -0.9

    def test_result_is_memoised(self, resolver):
        assert resolver.daily("temperature") is resolver.daily("temperature")

    def test_dependents_see_the_same_field(self):
        """Two consumers of one random variable must not each redraw it."""
        table = {
            "source": Var("1", "", lambda g: g.normal(0.0, 1.0)),
            "first": Var("1", "", lambda g: g.static("source")),
            "second": Var("1", "", lambda g: g.static("source")),
        }
        resolver = make_resolver(static_vars=table)
        np.testing.assert_array_equal(
            resolver.static("first").values, resolver.static("second").values
        )

    def test_circular_dependency_is_reported(self):
        table = {
            "a": Var("1", "", lambda g: g.static("b")),
            "b": Var("1", "", lambda g: g.static("a")),
        }
        with pytest.raises(ValueError, match="Circular dependency"):
            make_resolver(static_vars=table).static("a")


class TestSeed:
    def test_same_seed_gives_same_values(self):
        first = make_resolver(seed=7).daily("temperature").values
        second = make_resolver(seed=7).daily("temperature").values
        np.testing.assert_array_equal(first, second)

    def test_different_seed_gives_different_values(self):
        first = make_resolver(seed=7).daily("temperature").values
        second = make_resolver(seed=8).daily("temperature").values
        assert not np.array_equal(first, second)

    def test_unaffected_by_global_random_state(self):
        """Generation must not depend on whatever else touched np.random."""
        np.random.seed(1)
        first = make_resolver(seed=7).daily("temperature").values
        np.random.seed(2)
        np.random.random(1000)
        second = make_resolver(seed=7).daily("temperature").values
        np.testing.assert_array_equal(first, second)

    def test_variable_is_independent_of_what_else_was_requested(self):
        """The point of per-variable streams: no coupling through draw order."""
        alone = make_resolver(seed=7).daily("temperature").values

        resolver = make_resolver(seed=7)
        resolver.daily("precipitation")
        resolver.static("clay_content")
        after_others = resolver.daily("temperature").values

        np.testing.assert_array_equal(alone, after_others)

    def test_different_variables_do_not_share_a_stream(self):
        """Names must seed distinctly, or two variables would be identical."""
        table = {
            "first": Var("1", "", lambda g: g.normal(0.0, 1.0)),
            "second": Var("1", "", lambda g: g.normal(0.0, 1.0)),
        }
        resolver = make_resolver(static_vars=table)
        assert not np.array_equal(
            resolver.static("first").values, resolver.static("second").values
        )


class TestTables:
    @pytest.mark.parametrize("name", sorted(DAILY_VARS))
    def test_every_daily_var_generates(self, resolver, name):
        result = resolver.daily(name)
        assert result.shape == (N_DAYS, N_PIXELS)
        low, high = DAILY_VARS[name].bounds
        if low is not None:
            assert np.nanmin(result.values) >= low
        if high is not None:
            assert np.nanmax(result.values) <= high

    @pytest.mark.parametrize("name", sorted(STATIC_VARS))
    def test_every_static_var_generates(self, resolver, name):
        result = resolver.static(name)
        assert result.shape == (N_PIXELS,)
        low, high = STATIC_VARS[name].bounds
        if low is not None:
            assert np.nanmin(result.values) >= low
        if high is not None:
            assert np.nanmax(result.values) <= high

    @pytest.mark.parametrize("name", sorted(DAILY_VARS) + sorted(STATIC_VARS))
    def test_metadata_is_populated(self, name):
        var = DAILY_VARS.get(name) or STATIC_VARS[name]
        assert var.units
        assert var.long_name

    @pytest.mark.parametrize("module", [daily, static])
    def test_every_entry_has_a_docstring(self, module):
        """The docstring *is* the documentation — mkdocstrings renders it as-is."""
        assert _documented_names(module) == set(collect_vars(module))

    def test_single_pixel_grid(self):
        """A 1x1 grid is the quickstart's default and must not degenerate."""
        resolver = Resolver(
            grid=Grid(n_lat=1, n_lon=1, n_days=N_DAYS),
            daily_vars=DAILY_VARS,
            static_vars=STATIC_VARS,
            fallback=fallback_var,
        )
        assert resolver.daily("temperature").shape == (N_DAYS, 1)
        assert resolver.static("elevation").shape == (1,)
