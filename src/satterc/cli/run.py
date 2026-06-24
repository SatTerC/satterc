"""Execute a pipeline defined in a configuration file."""

from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from ..config import CacheSpec, load_config
from ..dag.blocking import execute_blocked
from ..dag.driver import build_driver
from ..io import get_final_vars, get_outputs, load_inputs, save_outputs

app = typer.Typer(help="Execute a pipeline defined in a configuration file.")


@app.command()
def run(
    config_file: Annotated[
        Path, typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True)
    ],
    allow_overrides: Annotated[
        bool,
        typer.Option(
            "--allow-overrides",
            help="Allow later modules to override earlier ones.",
        ),
    ] = False,
    cache: Annotated[
        bool | None,
        typer.Option(
            "--cache/--no-cache",
            help="Enable or disable result caching, overriding the [cache] "
            "section of the config.",
        ),
    ] = None,
    cache_dir: Annotated[
        Path | None,
        typer.Option(
            "--cache-dir",
            help="Directory for cached results (implies caching is enabled).",
        ),
    ] = None,
) -> None:
    """Execute a pipeline defined in a configuration file."""
    parsed = load_config(config_file)

    if parsed.units_mode is not None:
        from ..units import set_mode

        set_mode(parsed.units_mode)

    if parsed.units_exact is not None:
        from ..units import set_exact_match

        set_exact_match(parsed.units_exact)

    cache_spec = _resolve_cache(parsed.cache_spec, cache, cache_dir)

    inputs = load_inputs(parsed.input_specs)

    dr = build_driver(
        modules=parsed.modules,
        config=parsed.driver_config,
        allow_module_overrides=allow_overrides,
        cache=cache_spec,
    )

    if parsed.output_specs:
        target_vars = get_final_vars(parsed.output_specs)
        if parsed.blocking_spec is not None:
            results = execute_blocked(
                dr,
                inputs,
                target_vars,
                parsed.blocking_spec,
                build_params=(
                    parsed.modules,
                    parsed.driver_config,
                    cache_spec,
                    allow_overrides,
                ),
            )
        else:
            results = dr.execute(target_vars, inputs=inputs)  # type: ignore[reportArgumentType]
        output_datasets = get_outputs(results, parsed.output_specs)
        save_outputs(output_datasets, parsed.output_specs)


def _resolve_cache(
    config_cache: "CacheSpec | None",
    cache_flag: bool | None,
    cache_dir: Path | None,
) -> "CacheSpec | None":
    """Combine the config's [cache] spec with CLI overrides.

    ``--no-cache`` always wins. ``--cache`` or ``--cache-dir`` enable caching
    with defaults when the config has no [cache] section.
    """
    if cache_flag is False:
        return None
    spec = config_cache
    if cache_flag is True and spec is None:
        spec = CacheSpec()
    if cache_dir is not None:
        spec = replace(spec or CacheSpec(), path=str(cache_dir))
    return spec
