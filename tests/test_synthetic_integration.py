"""
Integration tests for the satterc pipeline using synthetic data.

This script can be run directly with:
    python -m tests.test_synthetic_integration

Or via pytest:
    pytest tests/test_synthetic_integration.py -v
"""

import numpy as np

from satterc import driver


def test_default_config(synthetic_inputs_defaults):
    """Test synthetic inputs with default configuration."""
    config = synthetic_inputs_defaults.copy()
    dr = driver.build_driver(
        modules=["synthetic_inputs", "resample"],
        config=config,
    )

    result = dr.execute(["temperature_celcius_daily"])
    assert result["temperature_celcius_daily"].shape == (731, 4)


def test_custom_grid(synthetic_inputs_defaults):
    """Test synthetic inputs with custom grid dimensions."""
    config = synthetic_inputs_defaults.copy()
    config["n_lat"] = 4
    config["n_lon"] = 4

    dr = driver.build_driver(
        modules=["synthetic_inputs", "resample"],
        config=config,
    )

    result = dr.execute(["temperature_celcius_daily"])
    assert result["temperature_celcius_daily"].shape == (731, 16)


def test_reproducibility(synthetic_inputs_defaults):
    """Test that same random seed produces identical results."""
    config1 = synthetic_inputs_defaults.copy()
    config1["random_seed"] = 42

    config2 = synthetic_inputs_defaults.copy()
    config2["random_seed"] = 42

    dr1 = driver.build_driver(modules=["synthetic_inputs"], config=config1)
    dr2 = driver.build_driver(modules=["synthetic_inputs"], config=config2)

    result1 = dr1.execute(["temperature_celcius_daily"])
    result2 = dr2.execute(["temperature_celcius_daily"])

    np.testing.assert_array_almost_equal(
        result1["temperature_celcius_daily"].values,
        result2["temperature_celcius_daily"].values,
    )


def test_different_seeds_differ(synthetic_inputs_defaults):
    """Test that different random seeds produce different results."""
    config1 = synthetic_inputs_defaults.copy()
    config1["random_seed"] = 42
    config2 = synthetic_inputs_defaults.copy()
    config2["random_seed"] = 123

    dr1 = driver.build_driver(modules=["synthetic_inputs"], config=config1)
    dr2 = driver.build_driver(modules=["synthetic_inputs"], config=config2)

    result1 = dr1.execute(["temperature_celcius_daily"])
    result2 = dr2.execute(["temperature_celcius_daily"])

    assert not np.allclose(
        result1["temperature_celcius_daily"].values,
        result2["temperature_celcius_daily"].values,
    )


def test_all_daily_functions(synthetic_driver_3x3):
    """Test that all daily synthetic input functions work."""
    daily_funcs = [
        "temperature_celcius_daily",
        "precipitation_mm_daily",
        "sunshine_fraction_daily",
        "pressure_pa_daily",
        "vpd_pa_daily",
        "co2_ppm_daily",
        "fapar_daily",
        "ppfd_umol_m2_s1_daily",
        "evaporation_daily",
        "aridity_index_daily",
        "mean_growth_temperature_daily",
        "plant_cover_daily",
        "dpm_rpm_ratio_daily",
        "carbon_input_daily",
        "farmyard_manure_input_daily",
    ]

    result = synthetic_driver_3x3.execute(daily_funcs)

    for name in daily_funcs:
        da = result[name]
        assert da.shape == (731, 9), f"{name} has wrong shape"
        assert not np.any(np.isnan(da.values)), f"{name} contains NaN"
        assert not np.any(np.isinf(da.values)), f"{name} contains Inf"


def test_aggregation(synthetic_driver):
    """Test that weekly and monthly aggregation works."""
    result = synthetic_driver.execute(
        [
            "temperature_celcius_weekly",
            "temperature_celcius_monthly",
            "precipitation_mm_weekly",
            "precipitation_mm_monthly",
        ]
    )

    assert result["temperature_celcius_weekly"].shape[0] == 105
    assert result["temperature_celcius_monthly"].shape[0] == 25
    assert result["precipitation_mm_weekly"].shape[0] == 105
    assert result["precipitation_mm_monthly"].shape[0] == 25


def test_grid_consistency(synthetic_driver_4x5):
    """Test that all DataArrays share consistent grid."""
    result = synthetic_driver_4x5.execute(
        [
            "temperature_celcius_daily",
            "precipitation_mm_daily",
            "evaporation_daily",
            "latitude",
            "longitude",
            "elevation",
        ]
    )

    ref = result["temperature_celcius_daily"]

    for name, da in result.items():
        if "time" in da.dims:
            assert da.coords["time"].equals(ref.coords["time"])
        if "pixel" in da.dims:
            assert da.coords["pixel"].equals(ref.coords["pixel"])


def test_custom_bounds(synthetic_inputs_defaults):
    """Test custom lat/lon bounds."""
    config = synthetic_inputs_defaults.copy()
    config["n_lat"] = 3
    config["n_lon"] = 3
    config["lat_min"] = 48.0
    config["lat_max"] = 60.0
    config["lon_min"] = -10.0
    config["lon_max"] = 5.0

    dr = driver.build_driver(
        modules=["synthetic_inputs"],
        config=config,
    )

    result = dr.execute(["latitude", "longitude"])

    lat = result["latitude"].values
    lon = result["longitude"].values

    assert lat[0] == 48.0
    assert lat[-1] == 60.0
    assert lon[0] == -10.0
    assert lon[-1] == 5.0


def test_time_coord(synthetic_driver_leap_year):
    """Test custom time configuration."""
    result = synthetic_driver_leap_year.execute(["dates"])

    dates = result["dates"].values
    assert len(dates) == 366


if __name__ == "__main__":
    print("Running integration tests...")

    from tests.conftest import (
        synthetic_driver,
        synthetic_driver_3x3,
        synthetic_driver_4x5,
        synthetic_driver_leap_year,
        synthetic_inputs_defaults,
    )

    test_default_config(synthetic_inputs_defaults)
    print("  test_default_config: PASSED")

    test_custom_grid(synthetic_inputs_defaults)
    print("  test_custom_grid: PASSED")

    test_reproducibility(synthetic_inputs_defaults)
    print("  test_reproducibility: PASSED")

    test_different_seeds_differ(synthetic_inputs_defaults)
    print("  test_different_seeds_differ: PASSED")

    test_all_daily_functions(synthetic_driver_3x3)
    print("  test_all_daily_functions: PASSED")

    test_aggregation(synthetic_driver)
    print("  test_aggregation: PASSED")

    test_grid_consistency(synthetic_driver_4x5)
    print("  test_grid_consistency: PASSED")

    test_custom_bounds(synthetic_inputs_defaults)
    print("  test_custom_bounds: PASSED")

    test_time_coord(synthetic_driver_leap_year)
    print("  test_time_coord: PASSED")

    print("\nAll integration tests PASSED!")
