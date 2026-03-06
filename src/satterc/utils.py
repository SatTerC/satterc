import functools
from typing import Any, Callable, Type, TypeVar, ParamSpec

import numpy as np
import xarray as xr

# T represents the wrapped function's return type
T = TypeVar("T")

# Define the supported backends explicitly
SupportedArrayTypes: tuple[Type, ...] = (np.ndarray,)

try:
    import jax

    # Add jax.Array to our allowed types for the decorator's type-checking
    SupportedArrayTypes += (jax.Array,)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# A Type Alias for your own documentation/type hinting
# InnerArray = Union[np.ndarray, "jax.Array"] if HAS_JAX else np.ndarray


def xarray_io(
    flatten_spatial: bool = False,
    inject_time: str | bool = False,
) -> Callable[[Callable[..., T]], Callable[..., Any]]:
    """A decorator to bridge Xarray to NumPy/JAX functions.

    This is designed to decorate functions that take one of the `SupportedArrayTypes` (`numpy.ndarray`
    or `jax.Array`) as arguments, to allow them to take `xarray.DataArray` arguments instead. Any
    `xarray.DataArray` inputs passed to the decorated function will be internally converted to a
    SupportedArrayType by accessing its `data` attribute and passing this to the wrapped function.
    Similarly, any returned values that are `SupportedArrayTypes` will be converted back into
    `xarray.DataArray`.

    Parameters:
    -----------
    flatten_spatial
        If `True`, spatial dimensions (all but the first) are flattened to produce a 2D array with
        dimensions (time, pixel) which is passed to the decorated function.
    inject_time
        If non-False, the datetime index of the first `xarray.DataArray` will be passed to the
        decorated function kwargs. If `True`, the kwarg will be called `time`. If instead a `str`
        is provided, this will be used instead.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if one or more xarray.DataArrays were provided as input.
            # Pull out the first DataArray to use as a metadata template
            all_inputs = list(args) + list(kwargs.values())
            reference_da = next(
                (val for val in all_inputs if isinstance(val, xr.DataArray)), None
            )

            # If no xarray.DataArray, pass the unmodified args through and return the result
            if reference_da is None:
                return func(*args, **kwargs)

            def _unpack_args(v: Any) -> Any:
                """Return 'data' attribute for any xarray.DataArray arguments.

                Also, transform (time, *spatial_dims) -> (time, pixel) if requested.
                """
                # WARN: this will flatten all dims except time, which is not what we want
                # if the DataArray is carrying around extra dims e.g. soil depth.
                return (
                    (v.stack(pixel=v.dims[1:]).data if flatten_spatial else v.data)
                    if isinstance(v, xr.DataArray)
                    else v
                )

            # Process arguments while maintaining JAX/NumPy dispatching
            new_args = [_unpack_args(arg) for arg in args]
            new_kwargs = {name: _unpack_args(kwarg) for name, kwarg in kwargs.items()}

            # Inject time index (Pandas/Xarray index) if requested by the pure function
            if inject_time:
                time_kwarg = "time" if inject_time is True else inject_time
                new_kwargs[time_kwarg] = reference_da.indexes.get("time")

            # Execute the inner function (returns T, e.g., np.ndarray)
            inner_returns: T = func(*new_args, **new_kwargs)

            unpacked_reference_da = _unpack_args(reference_da)

            def _repack_returns(v: Any, name: str | None = None) -> Any:
                """Recursively repack any SupportedArrayTypes into xarray.DataArrays."""
                if isinstance(v, SupportedArrayTypes):
                    # Compare rank to reference to handle case where func reduces over time dim
                    new_dims = (
                        unpacked_reference_da.dims
                        if v.ndim == unpacked_reference_da.ndim
                        else unpacked_reference_da.dims[1:]
                    )

                    v_da = xr.DataArray(
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
                    return (
                        v_da.unstack()
                        if flatten_spatial and "pixel" in v_da.dims
                        else v_da
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
