"""Unit declarations and runtime unit validation for the SatTerC DAG.

This module is the single source of truth for the physical units that model
nodes expect (``VARIABLE_UNITS``) and provides the runtime validation/conversion
helpers used by the :func:`satterc.dag._utils.xarray_io` decorator.

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
# Central name -> unit registry
# ---------------------------------------------------------------------------
# Keyed by Hamilton variable (node) name. Because Hamilton wires edges by name,
# the edge identity and the unit key are the same string, so a single table
# pins both the producer's stamped output unit and each consumer's expected
# input unit. Only variables that genuinely carry units at runtime (CF leaf
# inputs and model outputs stamped by ``xarray_io``) are listed here; structural
# / dimensionless bridge nodes are intentionally omitted so they are not
# spuriously flagged as "missing units".
VARIABLE_UNITS: dict[str, str] = {
    # --- climate / environmental leaf inputs ---
    "temperature_celcius_daily": "degC",
    "temperature_celcius_weekly": "degC",
    "temperature_celcius_monthly": "degC",
    "mean_growth_temperature_weekly": "degC",
    "vpd_pa_weekly": "Pa",
    "co2_ppm_weekly": "ppm",
    "pressure_pa_weekly": "Pa",
    "fapar_weekly": "1",
    "ppfd_umol_m2_s1_weekly": "umol m-2 s-1",
    "aridity_index_weekly": "1",
    "sunshine_fraction_daily": "1",
    "precipitation_mm_daily": "mm",
    "precipitation_mm_monthly": "mm",
    "evaporation_monthly": "mm",
    "lai_daily": "1",
    # --- pmodel outputs ---
    "gpp_daily": "g m-2 d-1",
    "gpp_weekly": "g m-2 d-1",
    "lue_weekly": "g MJ-1",
    "iwue_weekly": "Pa",
    # --- splash outputs ---
    "actual_evapotranspiration_daily": "mm d-1",
    "runoff_daily": "mm d-1",
    "soil_moisture_daily": "mm",
    "soil_moisture_weekly": "mm",
    # --- rothc outputs (tonnes C per hectare) ---
    "decomposable_plant_material_monthly": "t ha-1",
    "resistant_plant_material_monthly": "t ha-1",
    "microbial_biomass_monthly": "t ha-1",
    "humified_organic_matter_monthly": "t ha-1",
    "soil_organic_carbon_monthly": "t ha-1",
    "heterotrophic_respiration_monthly": "t ha-1",
}


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


# ---------------------------------------------------------------------------
# Unit resolution & checking
# ---------------------------------------------------------------------------


def resolve_input_unit(name: str, input_units: dict[str, str] | None) -> str | None:
    """Return the declared unit for an input parameter.

    Prefers an explicit ``input_units`` mapping, falling back to
    :data:`VARIABLE_UNITS`. Returns ``None`` if neither declares the name.
    """
    if input_units is not None and name in input_units:
        return input_units[name]
    return VARIABLE_UNITS.get(name)


def resolve_output_unit(
    name: str | None, output_units: dict[str, str] | str | None
) -> str | None:
    """Return the declared unit to stamp on an output array.

    ``output_units`` may be a bare string (single-array return), a dict keyed by
    output name, or ``None`` (fall back to :data:`VARIABLE_UNITS`).
    """
    if isinstance(output_units, str):
        return output_units
    if isinstance(output_units, dict) and name is not None and name in output_units:
        return output_units[name]
    if name is not None:
        return VARIABLE_UNITS.get(name)
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
