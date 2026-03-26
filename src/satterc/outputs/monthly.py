import xarray as xr
from hamilton.function_modifiers import group, inject, source
from hamilton.function_modifiers.delayed import ResolveAt

from ..dynamic._hamilton_fixes import FixedResolve
from ._utils import _save_dataset


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda monthly_outputs_vars: inject(
        monthly_outputs_list=group(
            *[source(f"{var}_monthly") for var in monthly_outputs_vars]
        )
    ),
)
def monthly_outputs_stacked(
    monthly_outputs_list: list[xr.DataArray],
    monthly_outputs_vars: list[str],
) -> xr.Dataset:
    """Merge monthly output data arrays into a single dataset.

    Parameters
    ----------
    monthly_outputs_list : list[xr.DataArray]
        List of monthly output data arrays.
    monthly_outputs : list[str]
        List of variable names to merge (resolved from config).

    Returns
    -------
    xr.Dataset
        Merged dataset with stacked spatial dimensions.
    """
    return xr.merge(monthly_outputs_list)


def monthly_outputs(monthly_outputs_stacked: xr.Dataset) -> xr.Dataset:
    """Unstack spatial dimensions from monthly outputs.

    Parameters
    ----------
    monthly_outputs_stacked : xr.Dataset
        Monthly outputs with stacked spatial dimensions.

    Returns
    -------
    xr.Dataset
        Monthly outputs with original spatial dimensions restored.
    """
    return monthly_outputs_stacked.unstack("pixel")


def saved_monthly_outputs(
    monthly_outputs: xr.Dataset, monthly_outputs_path: str
) -> None:
    """Save monthly outputs to file.

    Parameters
    ----------
    monthly_outputs : xr.Dataset
        Monthly outputs dataset.
    monthly_outputs_path : str
        Path to save the dataset.
    """
    _save_dataset(monthly_outputs, monthly_outputs_path)
