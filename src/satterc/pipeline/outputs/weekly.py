import xarray as xr
from hamilton.function_modifiers import group, inject, source
from hamilton.function_modifiers.delayed import ResolveAt

from .._hamilton_fixes import FixedResolve, NoOpDecorator
from ._utils import _save_dataset


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
    """Merge weekly output data arrays into a single dataset.

    Parameters
    ----------
    weekly_outputs_list : list[xr.DataArray]
        List of weekly output data arrays.
    weekly_outputs_vars : list[str]
        List of variable names to merge (resolved from config).

    Returns
    -------
    xr.Dataset
        Merged dataset with stacked spatial dimensions.
    """
    return xr.merge(weekly_outputs_list)


def unstacked_weekly_outputs(merged_weekly_outputs: xr.Dataset) -> xr.Dataset:
    """Unstack spatial dimensions from weekly outputs.

    Parameters
    ----------
    merged_weekly_outputs : xr.Dataset
        Weekly outputs with stacked spatial dimensions.

    Returns
    -------
    xr.Dataset
        Weekly outputs with original spatial dimensions restored.
    """
    return merged_weekly_outputs.unstack("pixel")


def save_weekly_outputs(
    unstacked_weekly_outputs: xr.Dataset, weekly_outputs_path: str
) -> None:
    """Save weekly outputs to file.

    Parameters
    ----------
    unstacked_weekly_outputs : xr.Dataset
        Weekly outputs dataset.
    weekly_outputs_path : str
        Path to save the dataset.
    """
    _save_dataset(unstacked_weekly_outputs, weekly_outputs_path)
