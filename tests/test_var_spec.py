"""Tests for satterc.setup_utils.data_gen.spec — the table-driven machinery."""

import ast
import inspect
from itertools import pairwise

import numpy as np
import pytest

from satterc.setup_utils.data_gen import daily, static
from satterc.setup_utils.data_gen.daily import DAILY_VARS
from satterc.setup_utils.data_gen.fallback import fallback_var
from satterc.setup_utils.data_gen.spec import Grid, Resolver, Var, collect_vars
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


def make_resolver(daily_vars=None, static_vars=None) -> Resolver:
    return Resolver(
        grid=Grid(n_lat=N_LAT, n_lon=N_LON, n_days=N_DAYS),
        daily_vars=DAILY_VARS if daily_vars is None else daily_vars,
        static_vars=STATIC_VARS if static_vars is None else static_vars,
        fallback=fallback_var,
    )


@pytest.fixture(autouse=True)
def _seeded():
    """Several assertions here are statistical; pin the global draw sequence."""
    np.random.seed(0)


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
        np.random.seed(7)
        first = make_resolver().daily("temperature").values
        np.random.seed(7)
        second = make_resolver().daily("temperature").values
        np.testing.assert_array_equal(first, second)

    def test_different_seed_gives_different_values(self):
        np.random.seed(7)
        first = make_resolver().daily("temperature").values
        np.random.seed(8)
        second = make_resolver().daily("temperature").values
        assert not np.array_equal(first, second)


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
