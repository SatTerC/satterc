from . import inputs, outputs, models, resample

__all__ = ["inputs", "outputs", "models", "resample", "MODULES"]

MODULES = [
    inputs.daily,
    inputs.weekly,
    inputs.monthly,
    inputs.static,
    inputs.grid,
    outputs.daily,
    outputs.weekly,
    outputs.monthly,
    resample,
    models.splash,
    models.pmodel,
    models.sgam,
    models.rothc,
]
