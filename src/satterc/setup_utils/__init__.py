from enum import StrEnum

from .config_gen import (
    generate_config,
    analyze_model_module,
    infer_required_data,
    get_builtin_models,
    get_model_params,
)

__all__ = [
    "generate_config",
    "analyze_model_module",
    "infer_required_data",
    "get_builtin_models",
    "get_model_params",
    "BuiltinModels",
]


class BuiltinModels(StrEnum):
    SPLASH = "splash"
    PMODEL = "pmodel"
    SGAM = "sgam"
    ROTHC = "rothc"
