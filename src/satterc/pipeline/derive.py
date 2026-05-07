import sys
import types
import uuid
from importlib import import_module
from typing import Any, TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from satterc.config import DeriveSpec


def make_derive_module(derive_specs: list["DeriveSpec"]) -> types.ModuleType:
    """Generate a Hamilton-compatible module with one function per derive spec."""
    mod = types.ModuleType(f"satterc_derive_generated_{uuid.uuid4().hex[:8]}")
    ns: dict = {"xr": xr, "Any": Any, "import_module": import_module}
    for spec in derive_specs:
        exec(_build_fn_code(spec), ns)
        fn = ns[spec.output]
        fn.__module__ = mod.__name__
        setattr(mod, spec.output, fn)
    sys.modules[mod.__name__] = mod
    return mod


def _build_fn_code(spec: "DeriveSpec") -> str:
    params = ", ".join(f"{inp}: Any" for inp in spec.inputs)
    if spec.expression is not None:
        body = f"    return {spec.expression}"
    else:
        kwargs = ", ".join(f"{inp}={inp}" for inp in spec.inputs)
        body = (
            f"    _fn = getattr(import_module({spec.import_path!r}), {spec.function!r})\n"
            f"    return _fn({kwargs})"
        )
    return f"def {spec.output}({params}) -> xr.DataArray:\n{body}\n"
