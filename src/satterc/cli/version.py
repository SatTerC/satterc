import typer

from .._version import __version__

app = typer.Typer(help="Show the installed version of SatTerC.")


@app.command()
def version() -> None:
    """Show the installed version of SatTerC."""
    typer.echo(f"satterc version {__version__}")
