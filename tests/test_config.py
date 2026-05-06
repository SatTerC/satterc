"""Unit tests for satterc.config."""

from pathlib import Path

import pytest

from satterc.config import Config, load_config, ParsedConfig


TEST_CONFIG_PATH = Path(__file__).parent / "test_config.toml"

EXPECTED_MODULES = [
    "grid",
    "inputs.daily",
    "inputs.weekly",
    "inputs.monthly",
    "inputs.static",
    "models.pmodel",
    "models.rothc",
    # resample absent — no [[resample]] entries in test_config.toml
    # outputs absent — no [outputs.*] sections in test_config.toml
]

EXPECTED_TARGETS = []  # no output sections → no targets


@pytest.fixture(scope="module")
def parsed_config():
    return load_config(TEST_CONFIG_PATH)


class TestLoadConfig:
    """Tests for the load_config() convenience function."""

    def test_returns_parsed_config(self, parsed_config):
        assert isinstance(parsed_config, ParsedConfig)

    def test_has_expected_fields(self, parsed_config):
        assert hasattr(parsed_config, "modules")
        assert hasattr(parsed_config, "driver_config")
        assert hasattr(parsed_config, "targets")


class TestModules:
    """Tests for the modules list derived from config sections."""

    def test_modules_list(self, parsed_config):
        assert parsed_config.modules == EXPECTED_MODULES


class TestDriverConfig:
    """Tests for driver_config keys and values."""

    def test_input_path_keys_present(self, parsed_config):
        dc = parsed_config.driver_config
        assert "daily_inputs_path" in dc
        assert "weekly_inputs_path" in dc
        assert "monthly_inputs_path" in dc
        assert "static_inputs_path" in dc

    def test_input_format_keys_derived(self, parsed_config):
        dc = parsed_config.driver_config
        assert dc["daily_inputs_format"] == "netcdf"
        assert dc["weekly_inputs_format"] == "netcdf"
        assert dc["monthly_inputs_format"] == "netcdf"
        assert dc["static_inputs_format"] == "netcdf"

    def test_input_vars_keys_present(self, parsed_config):
        dc = parsed_config.driver_config
        assert "daily_inputs_vars" in dc
        assert "weekly_inputs_vars" in dc
        assert "monthly_inputs_vars" in dc
        assert "static_inputs_vars" in dc

    def test_daily_input_vars(self, parsed_config):
        vars_ = parsed_config.driver_config["daily_inputs_vars"]
        assert "temperature_celcius" in vars_
        assert "precipitation_mm" in vars_
        assert "sunshine_fraction" in vars_

    def test_static_input_vars(self, parsed_config):
        vars_ = parsed_config.driver_config["static_inputs_vars"]
        assert "elevation" in vars_
        assert "clay_content" in vars_

    def test_model_params_merged_into_driver_config(self, parsed_config):
        dc = parsed_config.driver_config
        assert dc["method_kphio"] == "sandoval"
        assert dc["method_optchi"] == "lavergne20_c3"
        assert dc["n_years_spinup"] == 1


class TestFlatFormat:
    """Tests for flat-file format inference."""

    def test_csv_path_sets_flat_format(self, tmp_path):
        config = Config(
            {"inputs": {"daily": {"path": str(tmp_path / "daily.csv"), "vars": ["x"]}}}
        )
        parsed = config.parse()
        assert parsed.driver_config["daily_inputs_format"] == "flat"

    def test_parquet_path_sets_flat_format(self, tmp_path):
        config = Config(
            {
                "inputs": {
                    "daily": {"path": str(tmp_path / "daily.parquet"), "vars": ["x"]}
                }
            }
        )
        parsed = config.parse()
        assert parsed.driver_config["daily_inputs_format"] == "flat"

    def test_nc_path_sets_netcdf_format(self, tmp_path):
        config = Config(
            {"inputs": {"daily": {"path": str(tmp_path / "daily.nc"), "vars": ["x"]}}}
        )
        parsed = config.parse()
        assert parsed.driver_config["daily_inputs_format"] == "netcdf"

    def test_output_csv_path_sets_flat_format(self, tmp_path):
        config = Config(
            {"outputs": {"daily": {"path": str(tmp_path / "out.csv"), "vars": ["gpp"]}}}
        )
        parsed = config.parse()
        assert parsed.driver_config["daily_outputs_format"] == "flat"

    def test_unknown_extension_raises(self, tmp_path):
        config = Config(
            {"inputs": {"daily": {"path": str(tmp_path / "data.xyz"), "vars": ["x"]}}}
        )
        with pytest.raises(ValueError, match="Cannot determine format"):
            config.parse()


class TestTargets:
    """Tests for the targets list."""

    def test_targets_empty_when_no_output_sections(self, parsed_config):
        assert parsed_config.targets == []

    def test_targets_populated_when_vars_present(self, tmp_path):
        config = Config(
            {"outputs": {"daily": {"path": str(tmp_path / "out.nc"), "vars": ["gpp"]}}}
        )
        parsed = config.parse()
        assert "save_daily_outputs" in parsed.targets


class TestPathResolution:
    """Tests for path resolution relative to the config file location."""

    def test_input_paths_are_absolute(self, parsed_config):
        dc = parsed_config.driver_config
        for key in (
            "daily_inputs_path",
            "weekly_inputs_path",
            "monthly_inputs_path",
            "static_inputs_path",
        ):
            assert Path(dc[key]).is_absolute(), f"{key} should be absolute"

    def test_input_paths_resolve_relative_to_config(self, parsed_config):
        dc = parsed_config.driver_config
        assert Path(dc["daily_inputs_path"]) == TEST_CONFIG_PATH.parent / "daily.nc"

    def test_direct_construction_paths_unchanged(self):
        """Config() constructed directly should not modify paths."""
        config = Config(
            {"inputs": {"daily": {"path": "relative/path.nc", "vars": ["x"]}}}
        )
        assert config._data["inputs"]["daily"]["path"] == "relative/path.nc"


class TestValidation:
    """Tests for config validation behaviour."""

    def test_unknown_model_raises_value_error(self, tmp_path):
        config = Config({"models": {"unknown_model": {"param": "value"}}})
        with pytest.raises(ValueError, match="unknown_model"):
            from satterc.driver import build_driver

            parsed = config.parse()
            build_driver(parsed.modules, parsed.driver_config)

    def test_duplicate_model_params_raise(self, tmp_path):
        config = Config(
            {
                "models": {
                    "pmodel": {"shared_param": "a"},
                    "splash": {"shared_param": "b"},
                }
            }
        )
        with pytest.raises(ValueError, match="shared_param"):
            config.parse()

    def test_external_module_missing_import_path_raises(self):
        config = Config({"my_section": {"param": "value"}})
        with pytest.raises(ValueError, match="_import_path"):
            config.parse()

    def test_external_module_invalid_import_path_raises(self):
        config = Config({"my_section": {"_import_path": "not a.valid..path"}})
        with pytest.raises(ValueError, match="not a valid dotted module path"):
            config.parse()

    def test_external_module_import_path_accepted(self):
        config = Config(
            {"my_section": {"_import_path": "mypackage.mymodule", "param": 42}}
        )
        parsed = config.parse()
        assert "mypackage.mymodule" in parsed.modules
        assert parsed.driver_config["param"] == 42

    def test_input_section_missing_path_raises(self):
        config = Config({"inputs": {"daily": {"vars": ["x"]}}})
        with pytest.raises(ValueError, match=r"\[inputs\.daily\].*'path'"):
            config.parse()

    def test_output_section_empty_vars_raises(self, tmp_path):
        config = Config(
            {"outputs": {"daily": {"path": str(tmp_path / "out.nc"), "vars": []}}}
        )
        with pytest.raises(ValueError, match=r"\[outputs\.daily\].*no 'vars'"):
            config.parse()

    def test_output_section_missing_vars_raises(self, tmp_path):
        config = Config({"outputs": {"daily": {"path": str(tmp_path / "out.nc")}}})
        with pytest.raises(ValueError, match=r"\[outputs\.daily\].*no 'vars'"):
            config.parse()

    def test_output_section_missing_path_raises(self, tmp_path):
        config = Config({"outputs": {"daily": {"vars": ["gpp"]}}})
        with pytest.raises(ValueError, match=r"\[outputs\.daily\].*'path'"):
            config.parse()


class TestDump:
    """Tests for Config serialization."""

    def test_dump_roundtrip(self, tmp_path):
        """Config loaded, dumped, and reloaded should parse to the same result."""
        original = Config.load(TEST_CONFIG_PATH)
        out_path = tmp_path / "roundtrip.toml"
        original.dump(out_path)

        reloaded = Config.load(out_path).parse()
        original_parsed = original.parse()

        assert reloaded.modules == original_parsed.modules
        assert reloaded.targets == original_parsed.targets
        assert reloaded.driver_config.keys() == original_parsed.driver_config.keys()

    def test_format_keys_not_in_dump(self, tmp_path):
        """Format keys derived at parse time should not appear in the serialized TOML."""
        original = Config.load(TEST_CONFIG_PATH)
        out_path = tmp_path / "config.toml"
        original.dump(out_path)
        content = out_path.read_text()
        assert "_inputs_format" not in content
        assert "_outputs_format" not in content

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
