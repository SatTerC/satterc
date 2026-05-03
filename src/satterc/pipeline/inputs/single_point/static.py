from os import PathLike

import xarray as xr
from hamilton.function_modifiers import extract_fields, ResolveAt

from ..._hamilton_fixes import FixedResolve, NoOpDecorator
from ._utils import load_static


def loaded_static_inputs(static_inputs_path: str | PathLike) -> xr.Dataset:
    """Load single-point static inputs from a JSON, YAML, or TOML file.

    Parameters
    ----------
    static_inputs_path : str | PathLike
        Path to a .json, .yaml/.yml, or .toml file containing a flat
        key/value mapping of variable names to scalar numeric values.
        A 'pixel' dimension of size 1 is added automatically.

    Returns
    -------
    xr.Dataset
        Dataset with dim (pixel,).
    """
    return load_static(static_inputs_path)


def stacked_static_inputs(loaded_static_inputs: xr.Dataset) -> xr.Dataset:
    """Pass through: single-point data already has a 'pixel' dimension."""
    # stack_if_spatial passes through when 'pixel' dim already present
    return loaded_static_inputs


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda static_inputs_vars: (
        extract_fields({var: xr.DataArray for var in static_inputs_vars})
        if static_inputs_vars
        else NoOpDecorator()
    ),
)
def split_static_inputs(
    stacked_static_inputs: xr.Dataset,
    static_inputs_vars: list[str],
) -> dict[str, xr.DataArray]:
    """Unpack the static inputs dataset into individual DataArrays."""
    return {
        str(var): stacked_static_inputs[var] for var in stacked_static_inputs.data_vars
    }
