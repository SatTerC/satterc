"""Show the installed version of SatTerC."""

import typer
from conduit import __version__ as conduit_version

from .._version import __version__

app = typer.Typer(help="Show the installed version of SatTerC.")


@app.command()
def version() -> None:
    """Show the installed versions of SatTerC and the conduit framework."""
    typer.echo(f"satterc version {__version__}")
    typer.echo(f"conduit version {conduit_version}")
