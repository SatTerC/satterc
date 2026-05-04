from enum import StrEnum

from .config_gen import (
    generate_config,
    get_builtin_models,
    get_model_params,
)

__all__ = [
    "generate_config",
    "get_builtin_models",
    "get_model_params",
    "BuiltinModels",
]


class BuiltinModels(StrEnum):
    SPLASH = "splash"
    PMODEL = "pmodel"
    SGAM = "sgam"
    ROTHC = "rothc"
