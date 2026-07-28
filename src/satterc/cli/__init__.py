"""Command-line interface for SatTerC.

The pipeline commands (`run`, `graph`, `gridded`) are conduit's, mounted here
unchanged so that a satterc user has a single entry point rather than two. Only
`setup`, `data-gen` and `version` are satterc's own.
"""

import typer
from conduit.cli.graph import app as graph_app
from conduit.cli.run import app as run_app
from conduit.gridded.cli import app as gridded_app

from .data_gen import app as data_gen_app
from .setup import app as setup_app
from .version import app as version_app

app = typer.Typer(
    help="Command-line interface for SatTerC, built on the conduit framework.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(graph_app)
app.add_typer(setup_app)
app.add_typer(run_app)
app.add_typer(gridded_app, name="gridded")
app.add_typer(data_gen_app, name="data-gen")
app.add_typer(version_app)


def main() -> None:
    """Entry point for the satterc CLI."""
    app()
