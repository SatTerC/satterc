import functools
import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ..units import (
    assert_valid_unit,
    check_units,
    get_mode,
    units_from_signature,
)

# Define the supported backends explicitly
SupportedArrayTypes: tuple[type, ...] = (np.ndarray,)

try:
    import jax  # type: ignore[reportMissingImports]

    # Add jax.Array to our allowed types for the decorator's type-checking
    SupportedArrayTypes += (jax.Array,)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def declare_units(func: Callable[..., Any]) -> Callable[..., Any]:
    """Apply a node's signature-declared units at runtime.

    Reads the decorated node's own type annotations once, via
    :func:`satterc.units.units_from_signature`: parameters annotated
    ``Annotated[DataArray, "<unit>"]` declare input units, and a ``TypedDict``
    return (or a bare ``Annotated[DataArray, "<unit>"]`` return) declares output
    units. Those annotations are the single source of truth — the same one the
    (Phase 2) static DAG check reads off the Hamilton node — so units are never
    duplicated.

    At call time the wrapper:

    1. validates/converts each declared ``DataArray`` input to its unit via
       :func:`satterc.units.check_units`, honouring the active mode
       (`satterc.units.get_mode`); validation is skipped entirely in ``off`` mode;
    2. runs the node body;
    3. stamps each declared output ``DataArray`` with its unit (a ``dict`` return
       is stamped per output name; a single ``DataArray`` return is stamped with
       the bare declared unit).

    Only ``DataArray`` values are validated/stamped. Non-``DataArray`` arguments
    (`DatetimeIndex`, `Dataset`, scalar config parameters) carry no unit metadata
    and pass through untouched. This decorator does **not** convert between
    ``DataArray`` and ``ndarray`` — that boundary belongs to :func:`xarray_io` on
    the inner numpy implementation.

    Every declared unit string is checked against the registry **here, at
    decoration time**, so a malformed or undefined unit (a typo) fails fast at
    import — regardless of the active mode — rather than only when the node runs.
    """
    input_units, output_units = units_from_signature(func)
    sig = inspect.signature(func)

    # Fail fast on unparseable declarations (independent of validation mode).
    qualname = getattr(func, "__qualname__", repr(func))
    for name, unit in input_units.items():
        assert_valid_unit(unit, f"{qualname} input {name!r}")
    if isinstance(output_units, str):
        assert_valid_unit(output_units, f"{qualname} output")
    elif isinstance(output_units, dict):
        for name, unit in output_units.items():
            assert_valid_unit(unit, f"{qualname} output {name!r}")

    def _stamp(result: Any) -> Any:
        if isinstance(output_units, str):
            if isinstance(result, xr.DataArray):
                result.attrs["units"] = output_units
        elif isinstance(output_units, dict) and isinstance(result, dict):
            for name, value in result.items():
                declared = output_units.get(name)
                if declared is not None and isinstance(value, xr.DataArray):
                    value.attrs["units"] = declared
        return result

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        mode = get_mode()
        if mode != "off" and input_units:
            bound = sig.bind_partial(*args, **kwargs)
            for name, val in list(bound.arguments.items()):
                declared = input_units.get(name)
                if declared is not None and isinstance(val, xr.DataArray):
                    bound.arguments[name] = check_units(val, declared, name, mode)
            args, kwargs = bound.args, bound.kwargs

        return _stamp(func(*args, **kwargs))

    return wrapper


def xarray_io() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Bridge xarray.DataArray inputs to NumPy/JAX functions.

    This is designed to decorate functions that take one of the `SupportedArrayTypes`
    (`numpy.ndarray` or `jax.Array`) as arguments, to allow them to take
    `xarray.DataArray` arguments instead. Any `xarray.DataArray` inputs passed to
    the decorated function will be internally converted to a SupportedArrayType by
    accessing its `data` attribute and passing this to the wrapped function.
    Similarly, any returned values that are `SupportedArrayTypes` will be converted
    back into `xarray.DataArray`.

    This decorator handles the ``DataArray`` ↔ ``ndarray`` boundary only; unit
    declaration, validation and stamping live in :func:`declare_units` on the
    public node. Output DataArrays built here never inherit a ``units`` attribute
    from the reference input (a latent mislabelling bug); the declared unit is
    stamped later by :func:`declare_units`.

    Note
    ----
    Currently, the time dimension MUST be called "time" and the spatial coordinate
    dimension MUST be called "pixel".
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if one or more xarray.DataArrays were provided as input.
            # Pull out the first DataArray to use as a metadata template
            all_inputs = list(args) + list(kwargs.values())
            da_inputs = [v for v in all_inputs if isinstance(v, xr.DataArray)]

            # If no xarray.DataArray, pass unmodified args through and return
            if not da_inputs:
                return func(*args, **kwargs)

            def _is_valid_reference(da: xr.DataArray) -> bool:
                """Check if a DataArray can serve as a dimension reference."""
                return (
                    (da.ndim == 2)
                    and da.dims[0] == "time"
                    and isinstance(da.indexes.get("time", None), pd.DatetimeIndex)
                )

            valid_references = [v for v in da_inputs if _is_valid_reference(v)]
            if not valid_references:
                raise ValueError(
                    "None of the xarray.DataArray inputs satisfy "
                    "the criteria for this decorator."
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
                        ref_time_len = len(reference_da.coords["time"])
                        ref_pixel_len = len(reference_da.coords["pixel"])
                        if ref_time_len == ref_pixel_len:
                            raise ValueError(
                                f"Cannot infer dimension for 1D array of length "
                                f"{len(v)}: time length ({ref_time_len}) equals "
                                f"pixel length ({ref_pixel_len}). "
                                "Return the array in a dict with an explicit key, "
                                "e.g. {'time': arr} or {'pixel': arr}."
                            )
                        if len(v) == ref_time_len:
                            new_dims = ("time",)
                        elif len(v) == ref_pixel_len:
                            new_dims = ("pixel",)
                        else:
                            raise ValueError(
                                f"Cannot infer dimension for 1D array of length "
                                f"{len(v)}: does not match time length "
                                f"({ref_time_len}) or pixel length ({ref_pixel_len})."
                            )
                    elif v.ndim == 2:
                        # Check if shape is (pixel, time) or (time, pixel)
                        # and swap if needed to match reference
                        ref_time_len = len(reference_da.coords["time"])
                        ref_pixel_len = len(reference_da.coords["pixel"])
                        if v.shape[0] == ref_pixel_len and v.shape[1] == ref_time_len:
                            # Shape is (pixel, time) - transpose to (time, pixel)
                            v = v.T
                            new_dims = ("time", "pixel")
                        else:
                            new_dims = ("time", "pixel")
                    else:
                        raise NotImplementedError(
                            f"Repacking {v.ndim}D arrays is not supported. "
                            "xarray_io-decorated functions may only return "
                            "0D, 1D, or 2D arrays."
                        )

                    # Do not inherit the reference input's `units` attribute: it
                    # is meaningless for a transformed output and was a latent
                    # mislabelling bug. The declared output unit is stamped later
                    # by `declare_units` on the public node.
                    out_attrs = {
                        k: val for k, val in reference_da.attrs.items() if k != "units"
                    }

                    return xr.DataArray(
                        v,
                        coords={
                            d: reference_da.coords[d]
                            for d in new_dims
                            if d in reference_da.coords
                        },
                        dims=new_dims,
                        attrs=out_attrs,
                        name=name,
                    )
                elif isinstance(v, dict):
                    return {kk: _repack_returns(vv, name=kk) for kk, vv in v.items()}
                elif isinstance(inner_returns, list | tuple):
                    return type(inner_returns)([_repack_returns(vv) for vv in v])
                else:
                    return v

            return _repack_returns(inner_returns)

        return wrapper

    return decorator
