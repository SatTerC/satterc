from importlib.resources import files
import pytest
import xarray as xr


def _get_data_path(filename: str) -> str:
    """Get path to test data file using importlib.resources."""
    return str(files("satterc.synthetic_data.data").joinpath(filename))


@pytest.fixture
def daily_ds():
    return xr.open_dataset(
        _get_data_path("daily.nc"), engine="scipy", decode_coords="all"
    )


@pytest.fixture
def weekly_ds():
    return xr.open_dataset(
        _get_data_path("weekly.nc"), engine="scipy", decode_coords="all"
    )


@pytest.fixture
def monthly_ds():
    return xr.open_dataset(
        _get_data_path("monthly.nc"), engine="scipy", decode_coords="all"
    )


@pytest.fixture
def static_ds():
    return xr.open_dataset(
        _get_data_path("static.nc"), engine="scipy", decode_coords="all"
    )
