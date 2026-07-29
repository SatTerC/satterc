"""Carbon-cycle model modules for conduit pipelines.

Each module here is an ordinary conduit "bring your own module": a set of plain
functions whose names are node names and whose parameter names are upstream node
names, with units and frequency contracts declared on the signatures. Reference
one from a config by its dotted path, e.g.::

    [rothc]
    _import_path = "satterc.models.rothc"
    n_years_spinup = 1
"""

from . import pmodel, rothc, sgam, splash

__all__ = ["pmodel", "rothc", "sgam", "splash"]
