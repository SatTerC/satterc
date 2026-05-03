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
    decorate_with=lambda monthly_outputs_vars: (
        inject(
            monthly_outputs_list=group(
                *[source(f"{var}_monthly") for var in monthly_outputs_vars]
            )
        )
        if monthly_outputs_vars
        else NoOpDecorator()
    ),
)
def merged_monthly_outputs(
    monthly_outputs_list: list[xr.DataArray],
    monthly_outputs_vars: list[str],
) -> xr.Dataset:
    """Merge monthly output data arrays into a single dataset.

    Parameters
    ----------
    monthly_outputs_list : list[xr.DataArray]
        List of monthly output data arrays.
    monthly_outputs_vars : list[str]
        List of variable names to merge (resolved from config).

    Returns
    -------
    xr.Dataset
        Merged dataset with stacked spatial dimensions.
    """
    return xr.merge(monthly_outputs_list)


def unstacked_monthly_outputs(merged_monthly_outputs: xr.Dataset) -> xr.Dataset:
    """Unstack spatial dimensions from monthly outputs.

    Parameters
    ----------
    merged_monthly_outputs : xr.Dataset
        Monthly outputs with stacked spatial dimensions.

    Returns
    -------
    xr.Dataset
        Monthly outputs with original spatial dimensions restored.
    """
    return unstack_if_grid(merged_monthly_outputs)


@config.when(monthly_outputs_format="netcdf")
def save_monthly_outputs__netcdf(
    unstacked_monthly_outputs: xr.Dataset, monthly_outputs_path: str
) -> None:
    """Save monthly outputs to a NetCDF or Zarr file."""
    _save_dataset(unstacked_monthly_outputs, monthly_outputs_path)


@config.when(monthly_outputs_format="flat")
def save_monthly_outputs__flat(
    unstacked_monthly_outputs: xr.Dataset, monthly_outputs_path: str
) -> None:
    """Save monthly outputs to a CSV or Parquet file."""
    save_timeseries(
        dataset_to_dataframe(unstacked_monthly_outputs), monthly_outputs_path
    )
