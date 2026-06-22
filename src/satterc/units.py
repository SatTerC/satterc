"""Runtime unit validation helpers for the SatTerC DAG.

Each model node owns its own unit declarations via the
:func:`satterc.dag._utils.xarray_io` decorator (``input_units`` / ``output_units``);
there is deliberately no central registry of variable units, so the framework
never has to anticipate what inputs a user might bring. This module provides the
validation/conversion helpers and the validation-mode state used by that decorator.

Only ``DataArray`` inputs/outputs carry units and are validated/stamped. Other
node arguments (``DatetimeIndex``, scalar ``float``/``int``/``str`` config
parameters) have no attached unit metadata at runtime, so there is nothing to
validate or convert and they are left untouched.

Importing this module configures ``pint-xarray`` to use the UDUNITS-aware pint
registry shipped by ``cf-xarray``. That registry parses CF-convention unit
strings such as ``"umol m-2 s-1"`` and ``"g m-2 d-1"`` (which the plain pint
registry cannot), so declared units and ``units`` attributes read from CF
NetCDF/Zarr files are interpreted consistently.

Units are declared in canonical UDUNITS style (e.g. ``"umol m-2 s-1"``, not
``"µmol/m²/s"``) to match CF attributes on disk.
"""

import os
import warnings
from contextlib import contextmanager
from typing import Literal

import cf_xarray.units  # noqa: F401 -- registers the UDUNITS-aware pint registry
import pint
import pint_xarray
import xarray as xr
from cf_xarray.units import units as _UREG
from pint_xarray.errors import PintExceptionGroup

# Point pint-xarray's accessor at the cf-xarray (UDUNITS) registry, once, on
# import. Every ``.pint.quantify()`` call below then understands CF unit strings.
pint_xarray.setup_registry(_UREG)

Mode = Literal["strict", "warn", "off"]

VALID_MODES: frozenset[str] = frozenset({"strict", "warn", "off"})

#: Validation mode used when none is set via config/env. ``warn`` flags missing
#: ``units`` attributes without failing, which suits development and keeps the
#: feature non-breaking for inputs that lack CF metadata.
DEFAULT_MODE: Mode = "warn"

#: Environment variable that overrides the configured/default mode.
MODE_ENV_VAR = "SATTERC_UNITS_MODE"

_process_mode: Mode | None = None


# ---------------------------------------------------------------------------
# Mode handling
# ---------------------------------------------------------------------------


def _validate_mode(mode: str) -> Mode:
    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid units mode {mode!r}. Choose one of {sorted(VALID_MODES)}."
        )
    return mode  # type: ignore[return-value]


def set_mode(mode: str | None) -> None:
    """Set the process-wide unit validation mode.

    Passing ``None`` clears the process override so the default (or the
    ``SATTERC_UNITS_MODE`` environment variable) applies.
    """
    global _process_mode
    _process_mode = None if mode is None else _validate_mode(mode)


def get_mode() -> Mode:
    """Resolve the active unit validation mode.

    Resolution order: ``SATTERC_UNITS_MODE`` environment variable, then the
    value set via :func:`set_mode`, then :data:`DEFAULT_MODE`.
    """
    env = os.environ.get(MODE_ENV_VAR)
    if env:
        return _validate_mode(env.lower())
    if _process_mode is not None:
        return _process_mode
    return DEFAULT_MODE


@contextmanager
def mode(mode: str | None):
    """Temporarily override the unit validation mode.

    Restores the previous mode on exit, even if an exception is raised.

    >>> with units.mode("strict"):
    ...     check_units(da, "Pa", "vpd", get_mode())
    """
    global _process_mode
    old = _process_mode
    set_mode(mode)
    try:
        yield
    finally:
        _process_mode = old


# ---------------------------------------------------------------------------
# Unit resolution & checking
# ---------------------------------------------------------------------------


def resolve_input_unit(name: str, input_units: dict[str, str] | None) -> str | None:
    """Return the unit declared for an input parameter, or ``None``.

    Units are owned entirely by each model node's ``@xarray_io`` declaration;
    there is no central registry. Returns ``None`` when the node does not
    declare a unit for ``name`` (in which case the input is left unchecked).
    """
    if input_units is None:
        return None
    return input_units.get(name)


def resolve_output_unit(
    name: str | None, output_units: dict[str, str] | str | None
) -> str | None:
    """Return the unit to stamp on an output array, or ``None``.

    ``output_units`` may be a bare string (single-array return) or a dict keyed
    by output name. Returns ``None`` when no unit is declared for ``name``.
    """
    if isinstance(output_units, str):
        return output_units
    if isinstance(output_units, dict) and name is not None:
        return output_units.get(name)
    return None


def check_units(da: xr.DataArray, declared: str, name: str, mode: Mode) -> xr.DataArray:
    """Validate and convert an input ``DataArray`` to its declared unit.

    Returns a ``DataArray`` whose data is expressed in ``declared`` and whose
    ``units`` attribute equals ``declared``. If the input carries no ``units``
    attribute, behaviour follows ``mode`` (``strict`` raises, ``warn`` warns and
    returns the array unchanged, ``off`` returns unchanged). A dimensional
    incompatibility raises ``pint.DimensionalityError`` regardless of mode.
    """
    have = da.attrs.get("units")
    if have is None:
        if mode == "strict":
            raise ValueError(
                f"input {name!r} has no 'units' attribute (declared {declared!r})"
            )
        if mode == "warn":
            warnings.warn(
                f"input {name!r} unvalidated: no 'units' attribute "
                f"(declared {declared!r})",
                stacklevel=2,
            )
        return da
    try:
        converted = da.pint.quantify().pint.to(declared).pint.dequantify()
    except PintExceptionGroup as group:
        # pint-xarray wraps conversion failures in an ExceptionGroup; surface the
        # underlying DimensionalityError directly for a clean, catchable error.
        dim_errors = [
            exc for exc in group.exceptions if isinstance(exc, pint.DimensionalityError)
        ]
        if dim_errors:
            err = dim_errors[0]
            err.add_note(f"while validating input {name!r}")
            raise err from None
        raise
    # dequantify writes pint's canonical unit name (e.g. 'pascal'); restore the
    # declared UDUNITS string so downstream re-parsing uses our spelling.
    converted.attrs["units"] = declared
    return converted
