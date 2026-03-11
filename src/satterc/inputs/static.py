from os import PathLike

from hamilton.function_modifiers import extract_fields
import xarray as xr

from ._utils import load_dataset, stack_spatial_dims

STATIC_INPUTS = [
    "elevation",
    "plant_type",
    "max_soil_moisture",
    "clay_content",
    "soil_depth",
    "organic_carbon_stocks",
    "root_pool_init",
    "leaf_pool_init",
    "stem_pool_init",
]


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


@extract_fields(STATIC_INPUTS)
def unpack_static_inputs(static_inputs_stacked: xr.Dataset) -> dict[str, xr.DataArray]:
    """Unpacks the static dataset into individual arrays of input variables.

    Spatial coordinates are stacked into a single "pixel" dimension.

    Parameters
    ----------
    static_inputs_stacked : xr.Dataset
        The loaded dataset with coordinate reference system information.

    Returns
    -------
    dict[str, xr.DataArray]
        The data arrays.
    """
    return {
        str(var): static_inputs_stacked[var] for var in static_inputs_stacked.data_vars
    }
