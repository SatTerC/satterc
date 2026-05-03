import xarray as xr
from hamilton.function_modifiers import config, group, inject, source
from hamilton.function_modifiers.delayed import ResolveAt

from .._hamilton_fixes import FixedResolve, NoOpDecorator
from ._utils import (
    _save_dataset,
    dataset_to_dataframe,
    save_timeseries,
    unstack_if_grid,
)


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda daily_outputs_vars: (
        inject(
            daily_outputs_list=group(
                *[source(f"{var}_daily") for var in daily_outputs_vars]
            )
        )
        if daily_outputs_vars
        else NoOpDecorator()
    ),
)
def merged_daily_outputs(
    daily_outputs_list: list[xr.DataArray],
    daily_outputs_vars: list[str],
) -> xr.Dataset:
    """Merge daily output data arrays into a single dataset.

    Parameters
    ----------
    daily_outputs_list : list[xr.DataArray]
        List of daily output data arrays.
    daily_outputs_vars : list[str]
        List of variable names to merge (resolved from config).

    Returns
    -------
    xr.Dataset
        Merged dataset with stacked spatial dimensions.
    """
    return xr.merge(daily_outputs_list)


def unstacked_daily_outputs(merged_daily_outputs: xr.Dataset) -> xr.Dataset:
    """Unstack spatial dimensions from daily outputs.

    Parameters
    ----------
    merged_daily_outputs : xr.Dataset
        Daily outputs with stacked spatial dimensions.

    Returns
    -------
    xr.Dataset
        Daily outputs with original spatial dimensions restored.
    """
    return unstack_if_grid(merged_daily_outputs)


@config.when(daily_outputs_format="netcdf")
def save_daily_outputs__netcdf(
    unstacked_daily_outputs: xr.Dataset, daily_outputs_path: str
) -> None:
    """Save daily outputs to a NetCDF or Zarr file."""
    _save_dataset(unstacked_daily_outputs, daily_outputs_path)


@config.when(daily_outputs_format="flat")
def save_daily_outputs__flat(
    unstacked_daily_outputs: xr.Dataset, daily_outputs_path: str
) -> None:
    """Save daily outputs to a CSV or Parquet file."""
    save_timeseries(dataset_to_dataframe(unstacked_daily_outputs), daily_outputs_path)
