import typer

from .graph import app as graph_app
from .init_config import app as init_config_app
from .run import app as run_app
from .synthetic import app as synthetic_app
from .version import app as version_app

app = typer.Typer(
    help="Command-line interface for the SatTerC framework.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(graph_app)
app.add_typer(init_config_app)
app.add_typer(run_app)
app.add_typer(synthetic_app)
app.add_typer(version_app)


def main() -> None:
    app()
