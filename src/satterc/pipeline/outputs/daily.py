import xarray as xr
from hamilton.function_modifiers import group, inject, source
from hamilton.function_modifiers.delayed import ResolveAt

from .._hamilton_fixes import FixedResolve
from ._utils import _save_dataset


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda daily_outputs_vars: inject(
        daily_outputs_list=group(
            *[source(f"{var}_daily") for var in daily_outputs_vars]
        )
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
    return merged_daily_outputs.unstack("pixel")


def save_daily_outputs(
    unstacked_daily_outputs: xr.Dataset, daily_outputs_path: str
) -> None:
    """Save daily outputs to file.

    Parameters
    ----------
    unstacked_daily_outputs : xr.Dataset
        Daily outputs dataset.
    daily_outputs_path : str
        Path to save the dataset.
    """
    _save_dataset(unstacked_daily_outputs, daily_outputs_path)
