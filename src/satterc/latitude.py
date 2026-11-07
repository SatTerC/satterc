"""Workaround for the unlabelled `latitude` catena derives from a gridded input.

For a CRS-bearing input, `catena.load_inputs` computes ``latitude`` itself
(`catena.gridded.io.compute_lat_lon`) and gives it no ``units`` attribute, so a
model cannot declare ``Unit("degrees")`` on it: every gridded pipeline would
fail the check. Until catena labels it (NERC-CEH/catena#21), `splash` leaves
``latitude`` undeclared and calls `check_latitude` in its body.

SGAM and RothC need no check: they read only latitude's sign, to pick a
hemisphere, and the sign is the same in any unit.

Remove this module, and declare ``Annotated[DataArray, Unit("degrees")]`` on
`splash`'s ``latitude`` instead, once
``tests/test_latitude.py::test_workaround_is_still_needed`` fails.
"""

import xarray as xr
from xarray_annotated.units import check_units

#: The unit the wrapped models take latitude in.
LATITUDE_UNITS = "degrees"


def check_latitude(latitude: xr.DataArray, qualname: str) -> xr.DataArray:
    """Check ``latitude`` is in degrees, accepting catena's unlabelled one.

    A labelled array is checked exactly as a ``Unit("degrees")`` declaration
    would check it, so radians, or a unit that is not an angle, raise
    `xarray_annotated.units.UnitError`. ``degrees_north`` passes, since catena
    enables the CF unit registry. An array with no ``units`` attribute is assumed
    to be catena's, which is always in EPSG:4326 degrees, and passes unchecked.

    Parameters
    ----------
    latitude
        Per-pixel latitude.
    qualname
        The calling node, named in any error.

    Returns
    -------
    DataArray
        ``latitude`` itself, unchanged.
    """
    if "units" not in latitude.attrs:
        return latitude
    return check_units(latitude, LATITUDE_UNITS, "latitude", qualname)
