"""Fallback synthetic data generators for variables without explicit logic."""

import hashlib
import logging
import sys
import types

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_SUFFIX_RULES: list[tuple[list[str], str]] = [
    (["_fraction", "_ratio", "fapar", "sunshine"], "bounded"),
    (["precipitation", "lai", "gpp", "ppfd", "vpd", "pressure"], "positive"),
    (["_type", "_class", "_flag"], "integer"),
]


def _infer_fallback_type(var_name: str) -> str:
    name = var_name.lower()
    for keywords, ftype in _SUFFIX_RULES:
        if any(k in name for k in keywords):
            return ftype
    return "gaussian"


def _make_daily_fallback(var_name: str):
    ftype = _infer_fallback_type(var_name)

    def fn(
        time_coord: NDArray[np.datetime64],
        pixel_coords: pd.MultiIndex,
    ) -> xr.DataArray:
        logger.warning(
            "No generator for '%s'; emitting %s fallback noise.", var_name, ftype
        )
        shape = (len(time_coord), len(pixel_coords))
        if ftype == "bounded":
            data = np.clip(np.random.normal(0.5, 0.2, shape), 0.0, 1.0)
        elif ftype == "positive":
            data = np.abs(np.random.normal(1.0, 0.5, shape))
        elif ftype == "integer":
            data = np.random.randint(1, 4, shape).astype(float)
        else:
            data = np.random.normal(0.0, 1.0, shape)
        return xr.DataArray(
            data=data,
            dims=["time", "pixel"],
            coords={"time": time_coord, "pixel": pixel_coords},
            attrs={
                "units": "unknown",
                "long_name": var_name,
                "note": "synthetic fallback",
            },
            name=var_name,
        )

    fn.__name__ = fn.__qualname__ = f"{var_name}_daily"
    return fn


def _make_static_fallback(var_name: str):
    ftype = _infer_fallback_type(var_name)

    def fn(
        n_lat: int,
        n_lon: int,
        pixel_coords: pd.MultiIndex,
    ) -> xr.DataArray:
        logger.warning(
            "No generator for '%s'; emitting %s fallback noise.", var_name, ftype
        )
        n_pixels = n_lat * n_lon
        if ftype == "bounded":
            data = np.clip(np.random.normal(0.5, 0.2, n_pixels), 0.0, 1.0)
        elif ftype == "positive":
            data = np.abs(np.random.normal(1.0, 0.5, n_pixels))
        elif ftype == "integer":
            data = np.random.randint(1, 4, n_pixels).astype(float)
        else:
            data = np.random.normal(0.0, 1.0, n_pixels)
        return xr.DataArray(
            data=data,
            dims=["pixel"],
            coords={"pixel": pixel_coords},
            attrs={
                "units": "unknown",
                "long_name": var_name,
                "note": "synthetic fallback",
            },
            name=var_name,
        )

    fn.__name__ = fn.__qualname__ = var_name
    return fn


def build_fallback_module(
    unknown_daily: list[str],
    unknown_static: list[str],
) -> types.ModuleType:
    """Build a dynamic Hamilton-compatible module containing fallback generators.

    Hamilton resolves nodes by function ``__name__``, so each fallback function is
    renamed to match the expected node name before being attached to the module.

    Both the ``__module__`` reassignment and the `sys.modules` registration are
    load-bearing. Hamilton collects a module's nodes with `inspect.getmodule`,
    which resolves a function's ``__module__`` *through* `sys.modules`; an
    unregistered synthetic module resolves back to this file instead, and every
    fallback is silently skipped — surfacing much later as "Unknown nodes
    requested". The module name is keyed on its contents so that repeated calls
    in one process reuse an entry rather than accumulating them.
    """
    key = hashlib.sha256(
        repr((sorted(unknown_daily), sorted(unknown_static))).encode()
    ).hexdigest()[:12]
    name = f"satterc.setup_utils.data_gen._fallbacks_{key}"

    mod = types.ModuleType(name)
    factories = [
        (_make_daily_fallback, unknown_daily),
        (_make_static_fallback, unknown_static),
    ]
    for make, names in factories:
        for var in names:
            fn = make(var)
            fn.__module__ = name
            setattr(mod, fn.__name__, fn)

    sys.modules[name] = mod
    return mod
