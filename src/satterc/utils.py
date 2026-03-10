import functools
from typing import Any, Callable, Type

import numpy as np
import pandas as pd
import xarray as xr

# Define the supported backends explicitly
SupportedArrayTypes: tuple[Type, ...] = (np.ndarray,)

try:
    import jax

    # Add jax.Array to our allowed types for the decorator's type-checking
    SupportedArrayTypes += (jax.Array,)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def xarray_io() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """A decorator to bridge Xarray to NumPy/JAX functions.

    This is designed to decorate functions that take one of the `SupportedArrayTypes` (`numpy.ndarray`
    or `jax.Array`) as arguments, to allow them to take `xarray.DataArray` arguments instead. Any
    `xarray.DataArray` inputs passed to the decorated function will be internally converted to a
    SupportedArrayType by accessing its `data` attribute and passing this to the wrapped function.
    Similarly, any returned values that are `SupportedArrayTypes` will be converted back into
    `xarray.DataArray`.

    Parameters:
    -----------
    inject_time
        If non-False, the datetime index of the first `xarray.DataArray` will be passed to the
        decorated function kwargs. If `True`, the kwarg will be called `time`. If instead a `str`
        is provided, this will be used instead.

    Note:
    -----
    Currently, the time dimension MUST be called "time" and the spatial coordinate dimension
    MUST be called "pixel".
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if one or more xarray.DataArrays were provided as input.
            # Pull out the first DataArray to use as a metadata template
            all_inputs = list(args) + list(kwargs.values())
            da_inputs = [v for v in all_inputs if isinstance(v, xr.DataArray)]

            # If no xarray.DataArray, pass the unmodified args through and return the result
            if not da_inputs:
                return func(*args, **kwargs)

            def _is_valid_reference(da: xr.DataArray) -> bool:
                """Checks if a DataArray can serve as a reference for reconstructing dimensions."""
                return (
                    (da.ndim == 2)
                    and da.dims[0] == "time"
                    and isinstance(da.indexes.get("time", None), pd.DatetimeIndex)
                )

            valid_references = [v for v in da_inputs if _is_valid_reference(v)]
            if not valid_references:
                raise Exception(
                    "None of the xarray.DataArray inputs satisfy the criteria for this decorator."
                )
            reference_da = valid_references[0]

            def _unpack_args(v: Any) -> Any:
                """Return 'data' attribute for any xarray.DataArray arguments."""
                return v.data if isinstance(v, xr.DataArray) else v

            # Convert xarray.DataArrays to np.ndrrays or jax.Arrays
            new_args = [_unpack_args(arg) for arg in args]
            new_kwargs = {name: _unpack_args(kwarg) for name, kwarg in kwargs.items()}

            # Execute the inner function (returns T, e.g., np.ndarray)
            inner_returns = func(*new_args, **new_kwargs)

            def _repack_returns(v: Any, name: str | None = None) -> Any:
                """Recursively repack any SupportedArrayTypes into xarray.DataArrays."""
                if isinstance(v, SupportedArrayTypes):
                    if v.ndim == 0:
                        return v
                    elif v.ndim == 1:
                        new_dims = ("pixel",)
                    elif v.ndim == 2:
                        new_dims = ("time", "pixel")
                    else:
                        # TODO: bad
                        raise Exception("no")

                    return xr.DataArray(
                        v,
                        coords={
                            d: reference_da.coords[d]
                            for d in new_dims
                            if d in reference_da.coords
                        },
                        dims=new_dims,
                        attrs=reference_da.attrs,
                        name=name,
                    )
                elif isinstance(v, dict):
                    return {kk: _repack_returns(vv, name=kk) for kk, vv in v.items()}
                elif isinstance(inner_returns, (list, tuple)):
                    return type(inner_returns)([_repack_returns(vv) for vv in v])
                else:
                    return v

            return _repack_returns(inner_returns)

        return wrapper

    return decorator
