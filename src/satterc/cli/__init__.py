import typer

from .graph import app as graph_app
from .config_gen import app as config_gen_app
from .run import app as run_app
from .data_gen import app as data_gen_app
from .version import app as version_app

app = typer.Typer(
    help="Command-line interface for the SatTerC framework.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(graph_app)
app.add_typer(config_gen_app, name="config-gen")
app.add_typer(run_app)
app.add_typer(data_gen_app, name="data-gen")
app.add_typer(version_app)


def main() -> None:
    app()
