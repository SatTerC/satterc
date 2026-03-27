from os import PathLike

import xarray as xr
from hamilton.function_modifiers import extract_fields, ResolveAt

from ._utils import load_dataset, stack_spatial_dims
from .._hamilton_fixes import FixedResolve


def static_inputs(static_inputs_path: str | PathLike) -> xr.Dataset:
    """Load static input dataset from file.

    Parameters
    ----------
    static_inputs_path : Path
        Path to the NetCDF or Zarr dataset.

    Returns
    -------
    xr.Dataset
        The loaded dataset.
    """
    return load_dataset(static_inputs_path)


def static_inputs_stacked(static_inputs: xr.Dataset) -> xr.Dataset:
    """Stack spatial dimensions of static inputs dataset.

    Parameters
    ----------
    static_inputs : xr.Dataset
        The loaded static inputs dataset.

    Returns
    -------
    xr.Dataset
        Dataset with spatial dimensions stacked into 'pixel' dimension.
    """
    return stack_spatial_dims(static_inputs)


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda static_inputs_vars: extract_fields(
        {var: xr.DataArray for var in static_inputs_vars}
    ),
)
def unpack_static_inputs(
    static_inputs_stacked: xr.Dataset,
    static_inputs_vars: list[str],
) -> dict[str, xr.DataArray]:
    """Unpacks the static dataset into individual arrays of input variables.

    Spatial coordinates are stacked into a single "pixel" dimension.

    Parameters
    ----------
    static_inputs_stacked : xr.Dataset
        The loaded static inputs dataset.
    static_inputs_vars : List[str]
        List of variable names to extract (resolved from config).

    Returns
    -------
    dict[str, xr.DataArray]
        The data arrays.
    """
    return {
        str(var): static_inputs_stacked[var] for var in static_inputs_stacked.data_vars
    }
