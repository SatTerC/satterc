import xarray as xr

from hamilton.function_modifiers import extract_fields, ResolveAt

from .._hamilton_utils import FixedResolve


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda daily: extract_fields(
        {f"{var}_daily": xr.DataArray for var in daily}
    ),
)
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


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda weekly: extract_fields(
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


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda monthly: extract_fields(
        {f"{var}_monthly": xr.DataArray for var in monthly}
    ),
)
def unpack_monthly_inputs(
    monthly_inputs_stacked: xr.Dataset,
    monthly: list[str],
) -> dict[str, xr.DataArray]:
    """Unpacks the raw dataset into individual arrays of input variables.

    Spatial coordinates are stacked into a single "pixel" dimension.

    Parameters
    ----------
    monthly_inputs_stacked : xr.Dataset
        The loaded dataset with coordinate reference system information.
    monthly : list[str]
        List of variable names to extract (resolved from config).

    Returns
    -------
    dict[str, xr.DataArray]
            The data arrays.
    """
    return {
        f"{var}_monthly": monthly_inputs_stacked[var]
        for var in monthly_inputs_stacked.data_vars
    }


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda static: extract_fields({var: xr.DataArray for var in static}),
)
def unpack_static_inputs(
    static_inputs_stacked: xr.Dataset,
    static: list[str],
) -> dict[str, xr.DataArray]:
    """Unpacks the static dataset into individual arrays of input variables.

    Spatial coordinates are stacked into a single "pixel" dimension.

    Parameters
    ----------
    static_inputs_stacked : xr.Dataset
        The loaded static inputs dataset.
    static : List[str]
        List of variable names to extract (resolved from config).

    Returns
    -------
    dict[str, xr.DataArray]
        The data arrays.
    """
    return {
        str(var): static_inputs_stacked[var] for var in static_inputs_stacked.data_vars
    }
