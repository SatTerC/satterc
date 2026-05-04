import xarray as xr
from hamilton.function_modifiers import parameterize, source, value, ResolveAt

from ._hamilton_fixes import FixedResolve, NoOpDecorator
from ..config import ResampleSpec


@FixedResolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda resample_specs: (
        parameterize(
            **{
                f"{var}_{spec.to}": {
                    "var_in": source(f"{var}_{spec.from_}"),
                    "aggfunc": value(spec.aggfunc),
                    "freq": value(spec.freq),
                }
                for spec in resample_specs
                for var in spec.vars
            }
        )
        if resample_specs
        else NoOpDecorator()
    ),
)
def resample(var_in: xr.DataArray, aggfunc: str, freq: str) -> xr.DataArray:
    """Resample a DataArray to a coarser frequency using the given aggregation function.

    aggfunc must be a valid xarray DataArrayResample method (e.g. 'mean', 'sum').
    freq must be a valid pandas offset alias (e.g. '7D', '1ME').
    # TODO: consider closed/label options for finer control over bin edges
    """
    return getattr(var_in.resample(time=freq), aggfunc)()
