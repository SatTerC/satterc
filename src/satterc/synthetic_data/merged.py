"""Custom merged and save functions for synthetic data generation."""

from pathlib import Path

import xarray as xr
from hamilton.function_modifiers import group, inject, source
from hamilton.function_modifiers.delayed import ResolveAt

from ..pipeline._hamilton_fixes import FixedResolve, NoOpDecorator


def _add_crs_metadata(ds: xr.Dataset) -> xr.Dataset:
    """Add CRS metadata to dataset for rioxarray compatibility."""
    ds = ds.assign_attrs({"crs": "EPSG:4326"})
    for coord in ["x", "y"]:
        if coord in ds.coords:
            ds[coord].attrs["crs"] = "EPSG:4326"
    return ds


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
    """Merge daily output data arrays into a single dataset.

    Parameters
    ----------
    daily_outputs_list : list[xr.DataArray]
        List of daily output data arrays.
    daily_outputs_vars : list[str]
        List of variable names to merge.

    Returns
    -------
    xr.Dataset
        Merged dataset.
    """
    return xr.merge(daily_outputs_list)


def save_daily_outputs(
    merged_daily_outputs: xr.Dataset, daily_outputs_path: str
) -> None:
    """Save daily outputs to file.

    Parameters
    ----------
    merged_daily_outputs : xr.Dataset
        Daily outputs dataset.
    daily_outputs_path : str
        Path to save the dataset.
    """
    p = Path(daily_outputs_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _add_crs_metadata(merged_daily_outputs).to_netcdf(
        daily_outputs_path, format="NETCDF3_CLASSIC"
    )


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda daily_to_weekly: (
        inject(
            weekly_outputs_list=group(
                *[source(f"{var}_weekly") for var in daily_to_weekly]
            )
        )
        if daily_to_weekly
        else NoOpDecorator()
    ),
)
def merged_weekly_outputs(
    weekly_outputs_list: list[xr.DataArray],
    daily_to_weekly: list[str],
) -> xr.Dataset:
    """Merge weekly output data arrays into a single dataset.

    Parameters
    ----------
    weekly_outputs_list : list[xr.DataArray]
        List of weekly output data arrays.
    daily_to_weekly : list[str]
        List of variables resampled from daily.

    Returns
    -------
    xr.Dataset
        Merged dataset.
    """
    return xr.merge(weekly_outputs_list)


def save_weekly_outputs(
    merged_weekly_outputs: xr.Dataset, weekly_outputs_path: str
) -> None:
    """Save weekly outputs to file.

    Parameters
    ----------
    merged_weekly_outputs : xr.Dataset
        Weekly outputs dataset.
    weekly_outputs_path : str
        Path to save the dataset.
    """
    p = Path(weekly_outputs_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _add_crs_metadata(merged_weekly_outputs).to_netcdf(
        weekly_outputs_path, format="NETCDF3_CLASSIC"
    )


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda daily_to_monthly: (
        inject(
            monthly_outputs_list=group(
                *[source(f"{var}_monthly") for var in daily_to_monthly]
            )
        )
        if daily_to_monthly
        else NoOpDecorator()
    ),
)
def merged_monthly_outputs(
    monthly_outputs_list: list[xr.DataArray],
    daily_to_monthly: list[str],
) -> xr.Dataset:
    """Merge monthly output data arrays into a single dataset.

    Parameters
    ----------
    monthly_outputs_list : list[xr.DataArray]
        List of monthly output data arrays.
    daily_to_monthly : list[str]
        List of variables resampled from daily.

    Returns
    -------
    xr.Dataset
        Merged dataset.
    """
    return xr.merge(monthly_outputs_list)


def save_monthly_outputs(
    merged_monthly_outputs: xr.Dataset, monthly_outputs_path: str
) -> None:
    """Save monthly outputs to file.

    Parameters
    ----------
    merged_monthly_outputs : xr.Dataset
        Monthly outputs dataset.
    monthly_outputs_path : str
        Path to save the dataset.
    """
    p = Path(monthly_outputs_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _add_crs_metadata(merged_monthly_outputs).to_netcdf(
        monthly_outputs_path, format="NETCDF3_CLASSIC"
    )
