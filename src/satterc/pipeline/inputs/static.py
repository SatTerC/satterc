from os import PathLike

import xarray as xr
from hamilton.function_modifiers import config, extract_fields, ResolveAt

from ._utils import load_dataset, load_static, stack_if_spatial
from .._hamilton_fixes import FixedResolve, NoOpDecorator


@config.when(static_inputs_format="netcdf")
def loaded_static_inputs__netcdf(static_inputs_path: str | PathLike) -> xr.Dataset:
    """Load static inputs from a NetCDF or Zarr file."""
    return load_dataset(static_inputs_path)


@config.when(static_inputs_format="flat")
def loaded_static_inputs__flat(static_inputs_path: str | PathLike) -> xr.Dataset:
    """Load static inputs from a JSON, YAML, or TOML file."""
    return load_static(static_inputs_path)


def stacked_static_inputs(loaded_static_inputs: xr.Dataset) -> xr.Dataset:
    """Stack spatial dimensions of static inputs dataset.

    Parameters
    ----------
    loaded_static_inputs : xr.Dataset
        The loaded static inputs dataset.

    Returns
    -------
    xr.Dataset
        Dataset with spatial dimensions stacked into 'pixel' dimension.
    """
    return stack_if_spatial(loaded_static_inputs)


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
    """Unpacks the static dataset into individual arrays of input variables.

    Spatial coordinates are stacked into a single "pixel" dimension.

    Parameters
    ----------
    stacked_static_inputs : xr.Dataset
        The loaded static inputs dataset.
    static_inputs_vars : list[str]
        List of variable names to extract (resolved from config).

    Returns
    -------
    dict[str, xr.DataArray]
        The data arrays.
    """
    return {
        str(var): stacked_static_inputs[var] for var in stacked_static_inputs.data_vars
    }
