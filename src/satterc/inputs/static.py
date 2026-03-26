from os import PathLike

import xarray as xr

from ._utils import load_dataset, stack_spatial_dims


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
