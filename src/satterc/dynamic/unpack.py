import xarray as xr

from hamilton.function_modifiers import (
    resolve,
    ResolveAt,
)

from .._hamilton_utils import make_extract_fields_resolver
from .._hamilton_utils import LazyExtractFields, NoOpDecorator


@make_extract_fields_resolver("daily", "_daily")
def unpack_daily_inputs(
    daily_inputs_stacked: xr.Dataset,
    daily: list[str],
) -> dict[str, xr.DataArray]:
    """Unpacks the stacked daily inputs dataset into individual arrays of input variables.

    Parameters
    ----------
    daily_inputs_stacked : xr.Dataset
        The loaded dataset with coordinate reference system information.
    daily : list[str]
        List of variable names to extract (resolved from config).

    Returns
    -------
    dict[str, xr.DataArray]
            The data arrays.
    """
    return {
        f"{var}_daily": daily_inputs_stacked[var]
        for var in daily_inputs_stacked.data_vars
    }


@resolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda weekly: LazyExtractFields(
        {f"{var}_weekly": xr.DataArray for var in weekly}
    ),
)
def unpack_weekly_inputs(
    weekly_inputs_stacked: xr.Dataset,
    weekly: list[str],
) -> dict[str, xr.DataArray]:
    """Unpacks the raw dataset into individual arrays of input variables.

    Spatial coordinates are stacked into a single "pixel" dimension.

    Parameters
    ----------
    weekly_inputs_stacked : xr.Dataset
        The loaded dataset with coordinate reference system information.
    weekly : List[str]
        List of variable names to extract (resolved from config).

    Returns
    -------
    dict[str, xr.DataArray]
            The data arrays.
    """
    return {
        f"{var}_weekly": weekly_inputs_stacked[var]
        for var in weekly_inputs_stacked.data_vars
    }
