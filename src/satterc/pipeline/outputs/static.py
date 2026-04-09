import xarray as xr
from hamilton.function_modifiers import group, inject, source
from hamilton.function_modifiers.delayed import ResolveAt

from .._hamilton_fixes import FixedResolve, NoOpDecorator
from ._utils import _save_dataset


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
    """Merge static output data arrays into a single dataset.

    Parameters
    ----------
    static_outputs_list : list[xr.DataArray]
        List of static output data arrays.
    static_outputs_vars : list[str]
        List of variable names to merge (resolved from config).

    Returns
    -------
    xr.Dataset
        Merged dataset with stacked spatial dimensions.
    """
    return xr.merge(static_outputs_list)


def unstacked_static_outputs(merged_static_outputs: xr.Dataset) -> xr.Dataset:
    """Unstack spatial dimensions from static outputs.

    Parameters
    ----------
    merged_static_outputs : xr.Dataset
        Static outputs with stacked spatial dimensions.

    Returns
    -------
    xr.Dataset
        Static outputs with original spatial dimensions restored.
    """
    return merged_static_outputs.unstack("pixel")


def save_static_outputs(
    unstacked_static_outputs: xr.Dataset, static_outputs_path: str
) -> None:
    """Save static outputs to file.

    Parameters
    ----------
    unstacked_static_outputs : xr.Dataset
        Static outputs dataset.
    static_outputs_path : str
        Path to save the dataset.
    """
    _save_dataset(unstacked_static_outputs, static_outputs_path)
