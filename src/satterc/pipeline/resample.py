import xarray as xr
from hamilton.function_modifiers import parameterize_sources, ResolveAt

from ._hamilton_fixes import FixedResolve, NoOpDecorator


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda daily_to_weekly: (
        parameterize_sources(
            **{
                f"{var}_weekly": {"var_daily": f"{var}_daily"}
                for var in daily_to_weekly
            }
        )
        if daily_to_weekly
        else NoOpDecorator()
    ),
)
def resample_daily_to_weekly(var_daily: xr.DataArray) -> xr.DataArray:
    """Resamples daily xarray data to weekly mean."""
    return var_daily.resample(time="7D").mean()


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda daily_to_monthly: (
        parameterize_sources(
            **{
                f"{var}_monthly": {"var_daily": f"{var}_daily"}
                for var in daily_to_monthly
            }
        )
        if daily_to_monthly
        else NoOpDecorator()
    ),
)
def resample_daily_to_monthly(var_daily: xr.DataArray) -> xr.DataArray:
    """Resample daily data to monthly mean.

    Parameters
    ----------
    var_daily : xr.DataArray
        Daily input data.

    Returns
    -------
    xr.DataArray
        Monthly averaged data.
    """
    return var_daily.resample(time="1ME").mean()


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda weekly_to_monthly: (
        parameterize_sources(
            **{
                f"{var}_monthly": {"var_weekly": f"{var}_weekly"}
                for var in weekly_to_monthly
            }
        )
        if weekly_to_monthly
        else NoOpDecorator()
    ),
)
def resample_weekly_to_monthly(var_weekly: xr.DataArray) -> xr.DataArray:
    """Resample weekly data to monthly mean.

    Parameters
    ----------
    var_weekly : xr.DataArray
        Weekly input data.

    Returns
    -------
    xr.DataArray
        Monthly averaged data.
    """
    return var_weekly.resample(time="1ME").mean()
