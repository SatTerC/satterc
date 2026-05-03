import xarray as xr
from hamilton.function_modifiers import group, inject, source
from hamilton.function_modifiers.delayed import ResolveAt

from ..._hamilton_fixes import FixedResolve, NoOpDecorator
from .._utils import unstack_if_grid
from ._utils import dataset_to_dataframe, save_timeseries


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda weekly_outputs_vars: (
        inject(
            weekly_outputs_list=group(
                *[source(f"{var}_weekly") for var in weekly_outputs_vars]
            )
        )
        if weekly_outputs_vars
        else NoOpDecorator()
    ),
)
def merged_weekly_outputs(
    weekly_outputs_list: list[xr.DataArray],
    weekly_outputs_vars: list[str],
) -> xr.Dataset:
    """Merge weekly output DataArrays into a single Dataset."""
    return xr.merge(weekly_outputs_list)


def unstacked_weekly_outputs(merged_weekly_outputs: xr.Dataset) -> xr.Dataset:
    """Pass through for single-point data (pixel=[0] is not a MultiIndex)."""
    return unstack_if_grid(merged_weekly_outputs)


def save_weekly_outputs(
    unstacked_weekly_outputs: xr.Dataset, weekly_outputs_path: str
) -> None:
    """Convert weekly outputs to a time-series DataFrame and save to CSV or Parquet.

    Parameters
    ----------
    unstacked_weekly_outputs : xr.Dataset
        Weekly outputs with dims (time, pixel) where pixel has size 1.
    weekly_outputs_path : str
        Destination path (.csv or .parquet).
    """
    df = dataset_to_dataframe(unstacked_weekly_outputs)
    save_timeseries(df, weekly_outputs_path)
