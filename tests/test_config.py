"""Unit tests for satterc.config."""

import tomllib
from pathlib import Path

import pytest

from satterc.config import Config, load_config


TEST_CONFIG_PATH = Path(__file__).parent / "test_config.toml"

EXPECTED_MODULES = [
    "inputs.daily",
    "inputs.weekly",
    "inputs.monthly",
    "inputs.static",
    "resample",
    "outputs.daily",
    "outputs.weekly",
    "outputs.monthly",
]

EXPECTED_TARGETS = [
    "save_daily_outputs",
    "save_weekly_outputs",
    "save_monthly_outputs",
]


@pytest.fixture(scope="module")
def parsed_config():
    return load_config(TEST_CONFIG_PATH)


class TestLoadConfig:
    """Tests for the load_config() convenience function."""

    def test_returns_dict(self, parsed_config):
        assert isinstance(parsed_config, dict)

    def test_top_level_keys(self, parsed_config):
        assert set(parsed_config.keys()) == {
            "modules",
            "extra_modules",
            "driver_config",
            "targets",
        }


class TestModules:
    """Tests for the modules list."""

    def test_modules_list(self, parsed_config):
        assert parsed_config["modules"] == EXPECTED_MODULES

    def test_extra_modules_empty(self, parsed_config):
        assert parsed_config["extra_modules"] == []


class TestDriverConfig:
    """Tests for driver_config keys and values."""

    def test_input_path_keys_present(self, parsed_config):
        dc = parsed_config["driver_config"]
        assert "daily_inputs_path" in dc
        assert "weekly_inputs_path" in dc
        assert "monthly_inputs_path" in dc
        assert "static_inputs_path" in dc

    def test_output_path_keys_present(self, parsed_config):
        dc = parsed_config["driver_config"]
        assert "daily_outputs_path" in dc
        assert "weekly_outputs_path" in dc
        assert "monthly_outputs_path" in dc

    def test_input_vars_keys_present(self, parsed_config):
        dc = parsed_config["driver_config"]
        assert "daily_inputs_vars" in dc
        assert "weekly_inputs_vars" in dc
        assert "monthly_inputs_vars" in dc
        assert "static_inputs_vars" in dc

    def test_daily_input_vars(self, parsed_config):
        vars_ = parsed_config["driver_config"]["daily_inputs_vars"]
        assert "temperature_celcius" in vars_
        assert "precipitation_mm" in vars_
        assert "sunshine_fraction" in vars_

    def test_static_input_vars(self, parsed_config):
        vars_ = parsed_config["driver_config"]["static_inputs_vars"]
        assert "elevation" in vars_
        assert "clay_content" in vars_

    def test_extra_config_merged_into_driver_config(self, parsed_config):
        dc = parsed_config["driver_config"]
        assert dc["method_kphio"] == "sandoval"
        assert dc["method_optchi"] == "lavergne20_c3"
        assert dc["n_years_spinup"] == 1


class TestTargets:
    """Tests for the targets list."""

    def test_targets(self, parsed_config):
        assert parsed_config["targets"] == EXPECTED_TARGETS


class TestValidation:
    """Tests for config validation behaviour."""

    def test_unknown_section_raises_value_error(self, tmp_path):
        """A config section not listed in modules should raise ValueError."""
        bad_toml = tmp_path / "bad.toml"
        bad_toml.write_text(
            'modules = ["inputs.daily"]\n'
            "\n"
            "[inputs.daily]\n"
            'path = "daily.nc"\n'
            "vars = []\n"
            "\n"
            "[inputs.weekly]\n"  # not in modules list
            'path = "weekly.nc"\n'
            "vars = []\n"
        )
        with pytest.raises(ValueError, match="inputs.weekly"):
            load_config(bad_toml)


class TestDump:
    """Tests for Config serialization."""

    def test_dump_roundtrip(self, tmp_path):
        """Config loaded, dumped, and reloaded should parse to the same result."""
        original = Config.load(TEST_CONFIG_PATH)
        out_path = tmp_path / "roundtrip.toml"
        original.dump(out_path)

        reloaded = Config.load(out_path).parse()
        original_parsed = original.parse()

        assert reloaded["modules"] == original_parsed["modules"]
        assert reloaded["targets"] == original_parsed["targets"]
        assert reloaded["driver_config"].keys() == original_parsed["driver_config"].keys()

    def test_dump_refuses_overwrite_by_default(self, tmp_path):
        """dump() should raise FileExistsError if file already exists."""
        out_path = tmp_path / "config.toml"
        out_path.write_text("")
        config = Config.load(TEST_CONFIG_PATH)
        with pytest.raises(FileExistsError):
            config.dump(out_path)

    def test_dump_overwrite_ok(self, tmp_path):
        """dump(overwrite_ok=True) should succeed even if file exists."""
        out_path = tmp_path / "config.toml"
        out_path.write_text("")
        config = Config.load(TEST_CONFIG_PATH)
        config.dump(out_path, overwrite_ok=True)
        assert out_path.stat().st_size > 0
