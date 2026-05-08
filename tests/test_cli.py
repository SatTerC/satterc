"""Tests for the satterc CLI commands."""

import shutil
import tomllib
from unittest.mock import MagicMock

import pytest
import xarray as xr
from typer.testing import CliRunner

from satterc._version import __version__
from satterc.cli import app
from satterc.cli.data_gen import _parse_duration, _validate_output_paths
from satterc.cli.graph import custom_style
from satterc.cli.setup import _display_models, _parse_selections, _toggle_selections
from satterc.config import load_config

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_toml(tmp_path, synthetic_data_dir):
    """Config TOML pointing to session-scoped synthetic NetCDF files."""
    content = f"""\
[models.pmodel]
method_kphio = "sandoval"
method_optchi = "lavergne20_c3"

[models.rothc]
n_years_spinup = 1

[grid]

[inputs.daily]
path = "{synthetic_data_dir / "daily.nc"}"
vars = ["precipitation_mm", "sunshine_fraction", "temperature_celcius", "lai", "gpp"]

[inputs.weekly]
path = "{synthetic_data_dir / "weekly.nc"}"
vars = ["co2_ppm", "fapar", "ppfd_umol_m2_s1", "pressure_pa", "vpd_pa"]

[inputs.monthly]
path = "{synthetic_data_dir / "monthly.nc"}"
vars = ["dummy_variable"]

[inputs.static]
path = "{synthetic_data_dir / "static.nc"}"
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
    """Config TOML and output data dir for data-gen generate tests.

    The parent directory exists but no NetCDF files have been written yet.
    """
    data_dir = tmp_path / "data"
    content = f"""\
[models.rothc]
n_years_spinup = 1

[inputs.daily]
path = "{data_dir / "daily.nc"}"
vars = ["precipitation_mm", "sunshine_fraction", "temperature_celcius"]

[inputs.weekly]
path = "{data_dir / "weekly.nc"}"
vars = ["co2_ppm", "fapar", "ppfd_umol_m2_s1", "pressure_pa", "vpd_pa"]

[inputs.monthly]
path = "{data_dir / "monthly.nc"}"
vars = ["dummy_variable"]

[inputs.static]
path = "{data_dir / "static.nc"}"
vars = ["elevation", "plant_type", "clay_content", "soil_depth", "organic_carbon_stocks"]
"""
    toml_path = tmp_path / "datagen_config.toml"
    toml_path.write_text(content)
    return toml_path, data_dir


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


class TestVersionCommand:
    def test_exits_zero(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0

    def test_shows_version_string(self):
        result = runner.invoke(app, ["version"])
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
        assert out.with_suffix(".dot").exists()

    def test_missing_config_fails(self, tmp_path):
        result = runner.invoke(app, ["graph", str(tmp_path / "no.toml")])
        assert result.exit_code != 0


class TestCustomStyleFunction:
    def _mock_node(self, tags=None, type_=None, name=""):
        node = MagicMock()
        node.tags = tags or {}
        node.type = type_ or object
        node.name = name
        return node

    def test_static_input_gets_aquamarine(self):
        node = self._mock_node(tags={"module": "satterc.inputs.static"})
        style, _, label = custom_style(node=node, node_class="default")
        assert style["fillcolor"] == "aquamarine"
        assert label == "static inputs"

    def test_daily_dataarray_gets_orange(self):
        node = self._mock_node(type_=xr.DataArray, name="gpp_daily")
        style, _, _ = custom_style(node=node, node_class="default")
        assert style["fillcolor"] == "orange"

    def test_weekly_dataarray_gets_yellow(self):
        node = self._mock_node(type_=xr.DataArray, name="gpp_weekly")
        style, _, _ = custom_style(node=node, node_class="default")
        assert style["fillcolor"] == "yellow"

    def test_monthly_dataarray_gets_brown(self):
        node = self._mock_node(type_=xr.DataArray, name="gpp_monthly")
        style, _, _ = custom_style(node=node, node_class="default")
        assert style["fillcolor"] == "brown"

    def test_unrecognised_node_has_empty_style(self):
        node = self._mock_node(name="some_other_node")
        style, _, label = custom_style(node=node, node_class="default")
        assert style == {}
        assert label is None


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
# data-gen generate command
# ---------------------------------------------------------------------------


class TestDataGenGenerateCommand:
    def test_generate_creates_files(self, datagen_config_toml):
        toml_path, data_dir = datagen_config_toml
        result = runner.invoke(
            app,
            ["data-gen", "generate", str(toml_path), "--duration", "30d"],
        )
        assert result.exit_code == 0, result.output
        assert (data_dir / "daily.nc").exists()
        assert (data_dir / "static.nc").exists()

    def test_shows_generation_params_in_output(self, datagen_config_toml):
        toml_path, _ = datagen_config_toml
        result = runner.invoke(
            app,
            ["data-gen", "generate", str(toml_path), "--duration", "30d"],
        )
        assert "Grid dimensions" in result.output
        assert "Duration" in result.output
        assert "Random seed" in result.output

    def test_overwrite_confirmed_reruns_successfully(self, datagen_config_toml):
        toml_path, _data_dir = datagen_config_toml
        # First run creates files.
        runner.invoke(
            app, ["data-gen", "generate", str(toml_path), "--duration", "30d"]
        )
        # Second run: files exist → prompt → confirm overwrite.
        result = runner.invoke(
            app,
            ["data-gen", "generate", str(toml_path), "--duration", "30d"],
            input="y\n",
        )
        assert result.exit_code == 0, result.output

    def test_overwrite_declined_aborts(self, datagen_config_toml):
        toml_path, _data_dir = datagen_config_toml
        runner.invoke(
            app, ["data-gen", "generate", str(toml_path), "--duration", "30d"]
        )
        result = runner.invoke(
            app,
            ["data-gen", "generate", str(toml_path), "--duration", "30d"],
            input="n\n",
        )
        assert result.exit_code != 0

    def test_invalid_duration_fails(self, datagen_config_toml):
        toml_path, _ = datagen_config_toml
        result = runner.invoke(
            app, ["data-gen", "generate", str(toml_path), "--duration", "bad"]
        )
        assert result.exit_code != 0

    def test_missing_config_fails(self, tmp_path):
        result = runner.invoke(app, ["data-gen", "generate", str(tmp_path / "no.toml")])
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
        assert "models" in data
        assert "rothc" in data["models"]

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
