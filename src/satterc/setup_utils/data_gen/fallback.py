"""Synthesise a `Var` for a variable that has no table entry.

The tables in `satterc.setup_utils.data_gen.daily` / `.static` only need to cover
variables whose *structure* matters — those a model responds to seasonally, or
that other variables derive from. Everything else can be noise of roughly the
right character, which is what this module infers from the variable's name.

Falling back is the expected path, not an error: a config may name any input, and
generation should still produce a runnable file. The tables exist to make
particular variables realistic, not to enumerate the world.
"""

import logging

import numpy as np

from .spec import Var

logger = logging.getLogger(__name__)

#: Name fragments mapped to a shape of distribution. First match wins, so more
#: specific fragments must come first.
_NAME_RULES: list[tuple[list[str], str]] = [
    (["_fraction", "_ratio", "fapar", "sunshine"], "bounded"),
    (["precipitation", "lai", "gpp", "ppfd", "vpd", "pressure"], "positive"),
    (["_type", "_class", "_flag"], "integer"),
]

_FALLBACK_VARS: dict[str, Var] = {
    "bounded": Var("1", "", lambda g: g.normal(0.5, 0.2), bounds=(0.0, 1.0)),
    "positive": Var("unknown", "", lambda g: np.abs(g.normal(1.0, 0.5))),
    "integer": Var("1", "", lambda g: g.integers(1, 4)),
    "gaussian": Var("unknown", "", lambda g: g.normal(0.0, 1.0)),
}


def infer_kind(var_name: str) -> str:
    """Infer a distribution shape from a variable name.

    Returns one of ``"bounded"``, ``"positive"``, ``"integer"`` or ``"gaussian"``.
    """
    name = var_name.lower()
    for fragments, kind in _NAME_RULES:
        if any(fragment in name for fragment in fragments):
            return kind
    return "gaussian"


def fallback_var(var_name: str) -> Var:
    """Build a plausible `Var` for ``var_name`` from its name alone.

    The result works as either a daily or a static variable: it draws at whatever
    shape the context it is given asks for.
    """
    kind = infer_kind(var_name)
    logger.info("No table entry for '%s'; generating %s noise.", var_name, kind)
    template = _FALLBACK_VARS[kind]
    return Var(
        units=template.units,
        long_name=var_name,
        gen=template.gen,
        bounds=template.bounds,
        dtype=template.dtype,
    )
