"""Synthetic input data generation.

Variables are described declaratively as `Var` table entries in `daily` and
`static`; `spec` holds the machinery that turns a table row into an array.
"""

from .daily import DAILY_VARS
from .generate import generate_synthetic_data
from .spec import Grid, Resolver, Var
from .static import STATIC_VARS

__all__ = [
    "DAILY_VARS",
    "STATIC_VARS",
    "Grid",
    "Resolver",
    "Var",
    "generate_synthetic_data",
]
