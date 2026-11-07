"""The `satterc.latitude` workaround for catena's unlabelled latitude."""

import numpy as np
import pytest
import rioxarray  # noqa: F401  (registers the `.rio` accessor)
import xarray as xr
import xarray_annotated as xra
from catena.gridded.io import compute_lat_lon
from xarray_annotated.units import UnitError

from satterc.latitude import check_latitude


def _latitude(**attrs) -> xr.DataArray:
    return xr.DataArray([-30.0, 52.0], dims="pixel", attrs=attrs)


def test_workaround_is_still_needed():
    """Fails once catena labels the latitude it derives (NERC-CEH/catena#21).

    When it does, delete `satterc.latitude` and this file, declare
    ``Unit("degrees")`` on `splash`'s ``latitude``, and add ``("latitude", "lat")``
    back to ``SPLASH_INPUT_UNITS`` in ``tests/test_pyrealm_units.py``.
    """
    ds = xr.Dataset(
        {"elevation": (("y", "x"), np.zeros((2, 3)))},
        coords={"y": [51.0, 52.0], "x": [-1.0, 0.0, 1.0]},
    ).rio.write_crs("EPSG:4326")
    latitude, _ = compute_lat_lon({"static": ds})
    assert "units" not in latitude.attrs, (
        "catena now labels the latitude it derives, so `satterc.latitude` can go."
    )


class TestCheckLatitude:
    @pytest.mark.parametrize("units", ["degrees", "degree", "degrees_north"])
    def test_accepts_degrees(self, units):
        latitude = _latitude(units=units)
        with xra.enabled(True):
            assert check_latitude(latitude, "splash") is latitude

    def test_accepts_unlabelled(self):
        latitude = _latitude()
        with xra.enabled(True):
            assert check_latitude(latitude, "splash") is latitude

    @pytest.mark.parametrize("units", ["radians", "m"])
    def test_rejects_other_units(self, units):
        with xra.enabled(True), pytest.raises(UnitError, match=r"\[splash\]"):
            check_latitude(_latitude(units=units), "splash")

    def test_checks_nothing_when_switched_off(self):
        with xra.enabled(False):
            check_latitude(_latitude(units="radians"), "splash")
