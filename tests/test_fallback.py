"""Tests for satterc.scaffold.data_gen.fallback."""

import numpy as np
import pytest

from satterc.scaffold.data_gen.fallback import fallback_var, infer_kind
from satterc.scaffold.data_gen.spec import Grid, Resolver

N_DAYS = 10
N_LAT, N_LON = 2, 3
N_PIXELS = N_LAT * N_LON


@pytest.fixture
def resolver():
    """A resolver with empty tables, so every name falls back."""
    return Resolver(
        grid=Grid(n_lat=N_LAT, n_lon=N_LON, n_days=N_DAYS),
        daily_vars={},
        static_vars={},
        fallback=fallback_var,
    )


class TestInferKind:
    @pytest.mark.parametrize(
        ("name", "kind"),
        [
            ("sunshine_fraction", "bounded"),
            ("dpm_rpm_ratio", "bounded"),
            ("fapar_weekly", "bounded"),
            ("precipitation", "positive"),
            ("lai_daily", "positive"),
            ("gpp_monthly", "positive"),
            ("vpd_weekly", "positive"),
            ("pressure_daily", "positive"),
            ("ppfd", "positive"),
            ("plant_type", "integer"),
            ("land_class", "integer"),
            ("cover_flag", "integer"),
            ("temperature", "gaussian"),
            ("elevation", "gaussian"),
            ("co2", "gaussian"),
        ],
    )
    def test_kind_inferred_from_name(self, name, kind):
        assert infer_kind(name) == kind


class TestFallbackVar:
    def test_long_name_is_the_variable_name(self):
        assert fallback_var("some_var").long_name == "some_var"

    def test_bounded_names_are_clipped_to_unit_interval(self):
        assert fallback_var("sunshine_fraction").bounds == (0.0, 1.0)


class TestFallbackAsDaily:
    def test_dims_and_shape(self, resolver):
        result = resolver.daily("some_var")
        assert result.dims == ("time", "pixel")
        assert result.sizes == {"time": N_DAYS, "pixel": N_PIXELS}

    def test_bounded_values_in_range(self, resolver):
        values = resolver.daily("sunshine_fraction").values
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)

    def test_positive_values_non_negative(self, resolver):
        assert np.all(resolver.daily("precipitation").values >= 0.0)

    def test_integer_values_are_whole_numbers(self, resolver):
        values = resolver.daily("plant_type").values
        np.testing.assert_array_equal(values, np.floor(values))


class TestFallbackAsStatic:
    """The same fallback serves both kinds; only the context's shape differs."""

    def test_dims_and_shape(self, resolver):
        result = resolver.static("some_var")
        assert result.dims == ("pixel",)
        assert result.sizes == {"pixel": N_PIXELS}

    def test_bounded_values_in_range(self, resolver):
        values = resolver.static("sunshine_fraction").values
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)

    def test_positive_values_non_negative(self, resolver):
        assert np.all(resolver.static("precipitation").values >= 0.0)

    def test_integer_values_are_whole_numbers(self, resolver):
        values = resolver.static("plant_type").values
        np.testing.assert_array_equal(values, np.floor(values))
