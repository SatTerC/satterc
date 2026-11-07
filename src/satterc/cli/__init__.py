"""Command-line interface for SatTerC.

The pipeline commands (`run`, `graph`, `gridded`) are catena's, mounted here
unchanged so that a satterc user has a single entry point rather than two. Only
`setup` and `data-gen` are satterc's own.
"""

from typing import Annotated

import typer
from catena import __version__ as catena_version
from catena.cli.graph import app as graph_app
from catena.cli.run import app as run_app
from catena.errors import CatenaError
from catena.gridded.cli import app as gridded_app

from .._version import __version__
from .data_gen import app as data_gen_app
from .setup import app as setup_app

app = typer.Typer(
    help="Command-line interface for SatTerC, built on the catena framework.",
    context_settings={"help_option_names": ["-h", "--help"]},
    # `main` renders CatenaError itself; typer's traceback would bury the message.
    pretty_exceptions_enable=False,
)


def _show_version(value: bool) -> None:
    """Print the installed versions and exit.

    Attached to `--version` as an *eager* option callback, so it runs before the
    subcommand is resolved and `satterc --version` answers without a command.
    """
    if value:
        typer.echo(f"satterc version {__version__}")
        typer.echo(f"catena version {catena_version}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            callback=_show_version,
            is_eager=True,
            help="Show the installed versions of SatTerC and catena, and exit.",
        ),
    ] = False,
) -> None:
    """Run before any subcommand."""


app.add_typer(graph_app)
app.add_typer(setup_app)
app.add_typer(run_app)
app.add_typer(gridded_app, name="gridded")
app.add_typer(data_gen_app)


def main() -> None:
    """Entry point for the satterc CLI.

    A `CatenaError` is a condition catena anticipated and wrote a message for,
    so the message is all a user needs; the frames above it are catena's own
    call stack. Every other exception propagates with its traceback, because that
    is a bug and the frames are the point. This mirrors `catena.cli.main`, which
    a satterc user never reaches because the commands are mounted here.
    """
    try:
        app()
    except CatenaError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise SystemExit(1) from exc
