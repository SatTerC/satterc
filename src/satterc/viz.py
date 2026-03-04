from types import ModuleType

from hamilton import driver

from .models import splash
from .models import pmodel

_MODULES = dict(
    splash=splash,
    pmodel=pmodel,
)


def get_modules(*keys) -> list[ModuleType]:
    modules = []
    for key in keys:
        try:
            modules.append(_MODULES[key])
        except KeyError:
            raise
    return modules


def build_pipeline(*modules: str):
    return driver.Builder().with_modules(*get_modules(*modules)).build()


def viz():
    import subprocess

    dr = build_pipeline("splash", "pmodel")
    dr.display_all_functions(
        output_file_path="pipeline.dot",
        graphviz_kwargs=dict(graph_attr={"rankdir": "TB"}),
    )
    subprocess.run(["dot", "-Tpng", "pipeline.dot", "-o", "pipeline.png"])
