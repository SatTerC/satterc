import xarray as xr
from hamilton.function_modifiers import group, inject, source
from hamilton.function_modifiers.delayed import ResolveAt

from ..._hamilton_fixes import FixedResolve, NoOpDecorator
from .._utils import unstack_if_grid
from ._utils import dataset_to_dataframe, save_timeseries


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
    """Merge monthly output DataArrays into a single Dataset."""
    return xr.merge(monthly_outputs_list)


def unstacked_monthly_outputs(merged_monthly_outputs: xr.Dataset) -> xr.Dataset:
    """Pass through for single-point data (pixel=[0] is not a MultiIndex)."""
    return unstack_if_grid(merged_monthly_outputs)


def save_monthly_outputs(
    unstacked_monthly_outputs: xr.Dataset, monthly_outputs_path: str
) -> None:
    """Convert monthly outputs to a time-series DataFrame and save to CSV or Parquet.

    Parameters
    ----------
    unstacked_monthly_outputs : xr.Dataset
        Monthly outputs with dims (time, pixel) where pixel has size 1.
    monthly_outputs_path : str
        Destination path (.csv or .parquet).
    """
    df = dataset_to_dataframe(unstacked_monthly_outputs)
    save_timeseries(df, monthly_outputs_path)
