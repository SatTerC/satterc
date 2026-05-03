import xarray as xr
from hamilton.function_modifiers import group, inject, source
from hamilton.function_modifiers.delayed import ResolveAt

from ..._hamilton_fixes import FixedResolve, NoOpDecorator
from .._utils import unstack_if_grid
from ._utils import dataset_to_dataframe, save_timeseries


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda static_outputs_vars: (
        inject(static_outputs_list=group(*[source(var) for var in static_outputs_vars]))
        if static_outputs_vars
        else NoOpDecorator()
    ),
)
def merged_static_outputs(
    static_outputs_list: list[xr.DataArray],
    static_outputs_vars: list[str],
) -> xr.Dataset:
    """Merge static output DataArrays into a single Dataset."""
    return xr.merge(static_outputs_list)


def unstacked_static_outputs(merged_static_outputs: xr.Dataset) -> xr.Dataset:
    """Pass through for single-point data (pixel=[0] is not a MultiIndex)."""
    return unstack_if_grid(merged_static_outputs)


def save_static_outputs(
    unstacked_static_outputs: xr.Dataset, static_outputs_path: str
) -> None:
    """Convert static outputs to a DataFrame and save to CSV or Parquet.

    Parameters
    ----------
    unstacked_static_outputs : xr.Dataset
        Static outputs with dim (pixel,) where pixel has size 1.
    static_outputs_path : str
        Destination path (.csv or .parquet).
    """
    df = dataset_to_dataframe(unstacked_static_outputs)
    save_timeseries(df, static_outputs_path)
