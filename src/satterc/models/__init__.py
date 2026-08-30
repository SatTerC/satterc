"""Carbon-cycle model modules for conduit pipelines.

Each module here is an ordinary conduit "bring your own module": a set of plain
functions whose names are node names and whose parameter names are upstream node
names, with units and frequency contracts declared on the signatures.

satterc registers all four under conduit's ``conduit.modules`` entry-point group,
so a config names one by its section alone and needs no ``_import_path``::

    [rothc]
    n_years_spinup = 1

The section name is the module name: ``splash``, ``pmodel``, ``sgam``, ``rothc``.
"""

from . import pmodel, rothc, sgam, splash

__all__ = ["pmodel", "rothc", "sgam", "splash"]
