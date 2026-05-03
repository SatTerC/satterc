import xarray as xr
from hamilton.function_modifiers import group, inject, source
from hamilton.function_modifiers.delayed import ResolveAt

from ..._hamilton_fixes import FixedResolve, NoOpDecorator
from .._utils import unstack_if_grid
from ._utils import dataset_to_dataframe, save_timeseries


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
    """Merge daily output DataArrays into a single Dataset."""
    return xr.merge(daily_outputs_list)


def unstacked_daily_outputs(merged_daily_outputs: xr.Dataset) -> xr.Dataset:
    """Pass through for single-point data (pixel=[0] is not a MultiIndex)."""
    return unstack_if_grid(merged_daily_outputs)


def save_daily_outputs(
    unstacked_daily_outputs: xr.Dataset, daily_outputs_path: str
) -> None:
    """Convert daily outputs to a time-series DataFrame and save to CSV or Parquet.

    Any JAX-backed arrays are materialised to numpy during the conversion,
    identically to how to_netcdf() handles them.

    Parameters
    ----------
    unstacked_daily_outputs : xr.Dataset
        Daily outputs with dims (time, pixel) where pixel has size 1.
    daily_outputs_path : str
        Destination path (.csv or .parquet).
    """
    df = dataset_to_dataframe(unstacked_daily_outputs)
    save_timeseries(df, daily_outputs_path)
