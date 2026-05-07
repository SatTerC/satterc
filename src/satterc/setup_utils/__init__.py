"""Setup utilities for generating configurations and synthetic data."""

from enum import StrEnum

from .config_gen import (
    generate_config,
    get_builtin_models,
    get_model_params,
)

__all__ = [
    "BuiltinModels",
    "generate_config",
    "get_builtin_models",
    "get_model_params",
]


class BuiltinModels(StrEnum):
    """Enumeration of built-in model names."""

    SPLASH = "splash"
    PMODEL = "pmodel"
    SGAM = "sgam"
    ROTHC = "rothc"
