"""Tests for the satterc CLI commands."""

import shutil
import tomllib

import pytest
from conduit import load_config
from typer.testing import CliRunner

from satterc._version import __version__
from satterc.cli import app
from satterc.cli.data_gen import (
    _parse_bbox,
    _parse_duration,
    _parse_start_date,
    _validate_output_paths,
)
from satterc.cli.setup import (
    _display_models,
    _import_error,
    _parse_selections,
    _toggle_selections,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_toml(tmp_path, synthetic_data_dir):
    """Config TOML pointing to session-scoped synthetic NetCDF files."""
    content = f"""\
[pmodel]
_import_path = "satterc.models.pmodel"
method_kphio = "sandoval"
method_optchi = "lavergne20_c3"

[rothc]
_import_path = "satterc.models.rothc"
n_years_spinup = 1

[inputs.daily]
path = "{synthetic_data_dir / "daily.nc"}"
vars = ["precipitation", "sunshine_fraction", "temperature", "lai", "gpp"]

[inputs.weekly]
path = "{synthetic_data_dir / "weekly.nc"}"
vars = ["co2", "fapar", "ppfd", "pressure", "vpd"]

[inputs.monthly]
path = "{synthetic_data_dir / "monthly.nc"}"
vars = ["dummy_variable"]

[inputs.static]
path = "{synthetic_data_dir / "static.nc"}"
suffix = ""
vars = [
  "elevation", "plant_type", "max_soil_moisture", "clay_content",
  "soil_depth", "organic_carbon_stocks", "root_pool_init",
  "leaf_pool_init", "stem_pool_init",
]
"""
    p = tmp_path / "config.toml"
    p.write_text(content)
    return p


@pytest.fixture
def datagen_config_toml(tmp_path):
    """Config TOML and output data dir for data-gen tests.

    The parent directory exists but no NetCDF files have been written yet.
    """
    data_dir = tmp_path / "data"
    content = f"""\
[rothc]
_import_path = "satterc.models.rothc"
n_years_spinup = 1

[inputs.daily]
path = "{data_dir / "daily.nc"}"
vars = ["precipitation", "sunshine_fraction", "temperature"]

[inputs.weekly]
path = "{data_dir / "weekly.nc"}"
vars = ["co2", "fapar", "ppfd", "pressure", "vpd"]

[inputs.monthly]
path = "{data_dir / "monthly.nc"}"
vars = ["dummy_variable"]

[inputs.static]
path = "{data_dir / "static.nc"}"
suffix = ""
vars = ["elevation", "plant_type", "clay_content", "soil_depth", "organic_carbon_stocks"]
"""
    toml_path = tmp_path / "datagen_config.toml"
    toml_path.write_text(content)
    return toml_path, data_dir


# ---------------------------------------------------------------------------
# --version
# ---------------------------------------------------------------------------


class TestVersionFlag:
    @pytest.mark.parametrize("flag", ["--version", "-v"])
    def test_exits_zero(self, flag):
        result = runner.invoke(app, [flag])
        assert result.exit_code == 0

    @pytest.mark.parametrize("flag", ["--version", "-v"])
    def test_shows_version_string(self, flag):
        result = runner.invoke(app, [flag])
        assert __version__ in result.output


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


class TestRunCommand:
    def test_exits_zero(self, config_toml):
        result = runner.invoke(app, ["run", str(config_toml)])
        assert result.exit_code == 0, result.output

    def test_missing_config_fails(self, tmp_path):
        result = runner.invoke(app, ["run", str(tmp_path / "nonexistent.toml")])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------


class TestGraphCommand:
    @pytest.mark.skipif(not shutil.which("dot"), reason="graphviz not installed")
    def test_generates_dot_file(self, config_toml, tmp_path):
        out = tmp_path / "pipeline"
        result = runner.invoke(app, ["graph", str(config_toml), "--output", str(out)])
        assert result.exit_code == 0, result.output
        dot = out.with_suffix(".dot")
        assert dot.exists()
        text = dot.read_text()
        # Declared units appear in node labels in place of the "DataArray" type.
        assert "t ha-1" in text  # e.g. rothc soil carbon pools
        gpp_line = next(
            line
            for line in text.splitlines()
            if line.strip().startswith("gpp_flux_weekly ")
        )
        assert "<i>ug m-2 s-1</i>" in gpp_line
        assert "DataArray" not in gpp_line

    def test_missing_config_fails(self, tmp_path):
        result = runner.invoke(app, ["graph", str(tmp_path / "no.toml")])
        assert result.exit_code != 0


class TestComposedApp:
    """The pipeline commands are conduit's, mounted into satterc's app."""

    @pytest.mark.parametrize(
        "command", ["run", "graph", "gridded", "setup", "data-gen"]
    )
    def test_command_is_registered(self, command):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert command in result.output

    @pytest.mark.parametrize("subcommand", ["create-store", "merge"])
    def test_gridded_subcommands_registered(self, subcommand):
        result = runner.invoke(app, ["gridded", "--help"])
        assert result.exit_code == 0
        assert subcommand in result.output


# ---------------------------------------------------------------------------
# data-gen helpers
# ---------------------------------------------------------------------------


class TestDataGenHelpers:
    def test_parse_duration_years(self):
        assert _parse_duration("2y") == int(2 * 365.25)

    def test_parse_duration_months(self):
        assert _parse_duration("6m") == int(6 * 30.44)

    def test_parse_duration_days(self):
        assert _parse_duration("30d") == 30

    def test_parse_duration_case_insensitive(self):
        assert _parse_duration("1Y") == _parse_duration("1y")

    def test_parse_duration_invalid_format_raises(self):
        import typer

        with pytest.raises(typer.BadParameter):
            _parse_duration("bad")

    def test_parse_bbox_returns_lat_and_lon_ranges(self):
        assert _parse_bbox("50,54,-4,2") == ((50.0, 54.0), (-4.0, 2.0))

    def test_parse_bbox_accepts_negative_longitudes(self):
        """The reason the box is one string: click reads a leading '-' as a flag."""
        assert _parse_bbox("-45,-40,-170,-160") == ((-45.0, -40.0), (-170.0, -160.0))

    @pytest.mark.parametrize(
        "bbox",
        [
            "50,54,-4",  # too few
            "50,54,-4,2,9",  # too many
            "50,54,west,2",  # not numeric
            "54,50,-4,2",  # lat min above max
            "50,54,2,-4",  # lon min above max
            "50,95,-4,2",  # latitude off the planet
            "50,54,-4,200",  # longitude off the planet
        ],
    )
    def test_parse_bbox_invalid_raises(self, bbox):
        import typer

        with pytest.raises(typer.BadParameter):
            _parse_bbox(bbox)

    def test_parse_start_date_accepts_iso(self):
        assert _parse_start_date("2016-02-29") == "2016-02-29"

    @pytest.mark.parametrize("value", ["2020-13-01", "01/01/2020", "not-a-date"])
    def test_parse_start_date_invalid_raises(self, value):
        import typer

        with pytest.raises(typer.BadParameter):
            _parse_start_date(value)

    def test_validate_output_paths_fresh_files(self, datagen_config_toml):
        toml_path, data_dir = datagen_config_toml
        config = load_config(toml_path)
        paths, dirs_to_create, files_to_overwrite = _validate_output_paths(config)
        # data_dir does not exist yet → all four paths land in dirs_to_create
        assert len(paths) == 4
        assert data_dir in dirs_to_create
        assert files_to_overwrite == []

    def test_validate_output_paths_existing_files(self, datagen_config_toml):
        toml_path, data_dir = datagen_config_toml
        data_dir.mkdir()
        (data_dir / "daily.nc").write_bytes(b"")
        config = load_config(toml_path)
        _, _, files_to_overwrite = _validate_output_paths(config)
        assert any("daily.nc" in str(p) for p in files_to_overwrite)


# ---------------------------------------------------------------------------
# data-gen command
# ---------------------------------------------------------------------------


class TestDataGenGenerateCommand:
    def test_generate_creates_files(self, datagen_config_toml):
        toml_path, data_dir = datagen_config_toml
        result = runner.invoke(
            app,
            ["data-gen", str(toml_path), "--duration", "30d"],
        )
        assert result.exit_code == 0, result.output
        assert (data_dir / "daily.nc").exists()
        assert (data_dir / "static.nc").exists()

    def test_shows_generation_params_in_output(self, datagen_config_toml):
        toml_path, _ = datagen_config_toml
        result = runner.invoke(
            app,
            ["data-gen", str(toml_path), "--duration", "30d"],
        )
        assert "Grid dimensions" in result.output
        assert "Duration" in result.output
        assert "Random seed" in result.output

    def test_overwrite_confirmed_reruns_successfully(self, datagen_config_toml):
        toml_path, _data_dir = datagen_config_toml
        # First run creates files.
        runner.invoke(app, ["data-gen", str(toml_path), "--duration", "30d"])
        # Second run: files exist → prompt → confirm overwrite.
        result = runner.invoke(
            app,
            ["data-gen", str(toml_path), "--duration", "30d"],
            input="y\n",
        )
        assert result.exit_code == 0, result.output

    def test_overwrite_declined_aborts(self, datagen_config_toml):
        toml_path, _data_dir = datagen_config_toml
        runner.invoke(app, ["data-gen", str(toml_path), "--duration", "30d"])
        result = runner.invoke(
            app,
            ["data-gen", str(toml_path), "--duration", "30d"],
            input="n\n",
        )
        assert result.exit_code != 0

    def test_invalid_duration_fails(self, datagen_config_toml):
        toml_path, _ = datagen_config_toml
        result = runner.invoke(app, ["data-gen", str(toml_path), "--duration", "bad"])
        assert result.exit_code != 0

    def test_missing_config_fails(self, tmp_path):
        result = runner.invoke(app, ["data-gen", str(tmp_path / "no.toml")])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# setup helpers
# ---------------------------------------------------------------------------


class TestSetupHelpers:
    def test_parse_selections_comma_separated(self):
        assert _parse_selections("a,b") == ["a", "b"]

    def test_parse_selections_space_separated(self):
        assert _parse_selections("a b") == ["a", "b"]

    def test_parse_selections_mixed_delimiters(self):
        assert _parse_selections("a, b c") == ["a", "b", "c"]

    def test_parse_selections_empty_string(self):
        assert _parse_selections("") == []

    def test_toggle_adds_new_item(self):
        result = _toggle_selections([], ["splash"])
        assert "splash" in result

    def test_toggle_removes_existing_item(self):
        result = _toggle_selections(["splash"], ["splash"])
        assert "splash" not in result

    def test_toggle_skips_item_not_in_available_set(self):
        result = _toggle_selections([], ["unknown"], available={"splash", "pmodel"})
        assert result == []

    def test_display_models_marks_selected(self, capsys):
        _display_models(["splash", "pmodel", "rothc"], {"splash"})
        captured = capsys.readouterr()
        assert "[x]" in captured.out
        assert "splash" in captured.out
        assert "pmodel" in captured.out


# ---------------------------------------------------------------------------
# setup command — non-interactive (--defaults)
# ---------------------------------------------------------------------------


class TestModelSelectionFlag:
    """`--models` must accept every form a user would reasonably try.

    It was a single-valued option, so `-m splash -m pmodel` silently kept only
    the last — quietly generating a one-model config from a four-model command
    line, with no warning and exit 0.
    """

    FOUR = ("splash", "pmodel", "sgam", "rothc")

    def _models_in(self, path) -> list[str]:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        # A built-in model section carries no `_import_path` -- satterc
        # registers the module with conduit -- so it is identified by name.
        return [k for k in data if k in self.FOUR]

    @pytest.mark.parametrize(
        "flags",
        [
            pytest.param(
                ["-m", "splash", "-m", "pmodel", "-m", "sgam", "-m", "rothc"],
                id="repeated-flag",
            ),
            pytest.param(["-m", "splash,pmodel,sgam,rothc"], id="comma-separated"),
            pytest.param(["-m", "splash pmodel sgam rothc"], id="space-separated"),
            pytest.param(["-m", "splash,pmodel", "-m", "sgam,rothc"], id="mixed"),
        ],
    )
    def test_equivalent_forms_all_select_four_models(self, tmp_path, flags):
        out = tmp_path / "config.toml"
        result = runner.invoke(app, ["setup", "-d", "-o", str(out), *flags])
        assert result.exit_code == 0, result.output
        assert sorted(self._models_in(out)) == sorted(self.FOUR)

    def test_order_is_preserved_and_repeats_collapse(self, tmp_path):
        out = tmp_path / "config.toml"
        result = runner.invoke(
            app, ["setup", "-d", "-o", str(out), "-m", "rothc,splash,rothc"]
        )
        assert result.exit_code == 0, result.output
        assert self._models_in(out) == ["rothc", "splash"]

    def test_empty_selection_is_rejected(self, tmp_path):
        """It used to write a config with no models at all, and exit 0."""
        out = tmp_path / "config.toml"
        result = runner.invoke(app, ["setup", "-d", "-o", str(out), "-m", ""])
        assert result.exit_code != 0
        assert not out.exists()

    def test_unknown_model_lists_the_valid_ones(self, tmp_path):
        out = tmp_path / "config.toml"
        result = runner.invoke(app, ["setup", "-d", "-o", str(out), "-m", "nosuch"])
        assert result.exit_code != 0
        assert "nosuch" in result.output
        assert "splash" in result.output  # names the alternatives


class TestCustomModuleImportCheck:
    """A mistyped module path used to be accepted in silence.

    `get_model_config` swallows the ImportError and returns {}, so a typo was
    reported as "no configurable settings found" and only failed later, when
    the pipeline ran.
    """

    def test_importable_module_is_added_without_a_prompt(self):
        assert _import_error("satterc.models.rothc") is None

    def test_unimportable_module_reports_why(self):
        error = _import_error("nosuch.module")
        assert error is not None
        assert "ModuleNotFoundError" in error

    def test_declining_leaves_the_typo_out(self, tmp_path):
        out = tmp_path / "config.toml"
        # builtin splash; then a typo'd custom module, declined; defaults; no data
        result = runner.invoke(
            app,
            ["setup", "-o", str(out)],
            input="1\n0\nnosuch.module\nn\n\ny\n\nn\n",
        )
        assert result.exit_code == 0, result.output
        assert "Cannot import 'nosuch.module'" in result.output
        assert "nosuch.module" not in out.read_text()

    def test_accepting_anyway_still_adds_it(self, tmp_path):
        """The escape hatch: a module may be installed before the config is run."""
        out = tmp_path / "config.toml"
        result = runner.invoke(
            app,
            ["setup", "-o", str(out)],
            input="1\n0\nnosuch.module\ny\n\ny\n\nn\n",
        )
        assert result.exit_code == 0, result.output
        assert '_import_path = "nosuch.module"' in out.read_text()


class TestSetupCommandNonInteractive:
    def test_defaults_creates_toml(self, tmp_path):
        out = tmp_path / "config.toml"
        result = runner.invoke(
            app,
            ["setup", "--defaults", "--models", "rothc", "--output", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_generated_toml_is_loadable(self, tmp_path):
        out = tmp_path / "config.toml"
        runner.invoke(
            app,
            ["setup", "--defaults", "--models", "rothc", "--output", str(out)],
        )
        # Should parse without error.
        load_config(out)

    def test_generated_toml_contains_model_params(self, tmp_path):
        out = tmp_path / "config.toml"
        runner.invoke(
            app,
            ["setup", "--defaults", "--models", "rothc", "--output", str(out)],
        )
        with open(out, "rb") as f:
            data = tomllib.load(f)
        assert "_import_path" not in data["rothc"]
        assert "n_years_spinup" in data["rothc"]

    def test_defaults_without_models_fails(self):
        result = runner.invoke(app, ["setup", "--defaults"])
        assert result.exit_code != 0

    def test_invalid_model_name_fails(self, tmp_path):
        out = tmp_path / "config.toml"
        result = runner.invoke(
            app,
            ["setup", "--defaults", "--models", "notamodel", "--output", str(out)],
        )
        assert result.exit_code != 0

    def test_existing_output_with_defaults_exits_with_error(self, tmp_path):
        out = tmp_path / "config.toml"
        out.write_text("# existing")
        result = runner.invoke(
            app,
            ["setup", "--defaults", "--models", "rothc", "--output", str(out)],
        )
        assert result.exit_code == 1
        assert str(out) in result.output


# ---------------------------------------------------------------------------
# setup command — interactive
# ---------------------------------------------------------------------------


class TestSetupCommandInteractive:
    def test_models_option_with_interactive_prompts_creates_config(self, tmp_path):
        out = tmp_path / "config.toml"
        # Prompts in order:
        #   _select_custom_modules: module path → "\n" (finish)
        #   confirm "Use default paths?" → "\n" (accept True)
        #   prompt "Output config path" → "\n" (accept default)
        #   confirm "Generate synthetic data?" → "\n" (accept False)
        result = runner.invoke(
            app,
            ["setup", "--models", "rothc", "--output", str(out)],
            input="\n\n\n\n",
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_interactive_overwrite_confirmed(self, tmp_path):
        out = tmp_path / "config.toml"
        out.write_text("# old content")
        # First prompt: "Overwrite?" → "y"; then the 4 interactive prompts.
        result = runner.invoke(
            app,
            ["setup", "--models", "rothc", "--output", str(out)],
            input="y\n\n\n\n\n",
        )
        assert result.exit_code == 0, result.output
        # File should have been replaced with valid TOML.
        with open(out, "rb") as f:
            tomllib.load(f)

    def test_interactive_overwrite_declined(self, tmp_path):
        out = tmp_path / "config.toml"
        original = "# old content"
        out.write_text(original)
        result = runner.invoke(
            app,
            ["setup", "--models", "rothc", "--output", str(out)],
            input="n\n",
        )
        assert result.exit_code == 0
        assert out.read_text() == original

    def test_fully_interactive_model_selection(self, tmp_path):
        out = tmp_path / "config.toml"
        # Prompts in order:
        #   _select_builtin_models: "1\n" (select splash), "0\n" (done)
        #   _select_custom_modules: "\n" (finish)
        #   confirm "Use default paths?" → "\n"
        #   prompt "Output config path" → "\n"
        #   confirm "Generate synthetic data?" → "\n"
        result = runner.invoke(
            app,
            ["setup", "--output", str(out)],
            input="1\n0\n\n\n\n\n",
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_interactive_with_data_generation(self, tmp_path):
        out = tmp_path / "config.toml"
        # Prompts in order:
        #   custom modules → "\n"
        #   use default paths → "\n"
        #   output path → "\n"  (accepts --output default)
        #   generate data → "y\n"
        #   grid → "\n" (1,1)
        #   duration → "30d\n"
        #   seed → "\n" (42)
        result = runner.invoke(
            app,
            ["setup", "--models", "splash", "--output", str(out)],
            input="\n\n\ny\n\n30d\n\n",
        )
        assert result.exit_code == 0, result.output
        # Generated config must be loadable.
        load_config(out)
        # Input data files must have been written alongside the config.
        inputs_dir = out.parent / "inputs"
        assert inputs_dir.exists()
        assert any(inputs_dir.iterdir())
