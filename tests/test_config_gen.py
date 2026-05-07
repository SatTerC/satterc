"""Tests for setup_utils/config_gen.py."""

from satterc.setup_utils import generate_config, get_model_params


PATH_DEFAULTS = {
    "inputs_daily": "inputs/daily.nc",
    "inputs_weekly": "inputs/weekly.nc",
    "inputs_monthly": "inputs/monthly.nc",
    "inputs_static": "inputs/static.nc",
    "outputs_daily": "outputs/daily.nc",
    "outputs_weekly": "outputs/weekly.nc",
    "outputs_monthly": "outputs/monthly.nc",
}


class TestGetModelParams:
    def test_builtin_returns_defaults(self):
        params = get_model_params("rothc")
        assert "n_years_spinup" in params
        assert isinstance(params["n_years_spinup"], int)

    def test_unknown_module_returns_empty(self):
        assert get_model_params("nonexistent.module") == {}

    def test_custom_path_uses_last_component_as_func_prefix(self):
        # Passing the full dotted path to a known module must find the same
        # _parameters() function as passing the short name.
        params_short = get_model_params("rothc")
        params_full = get_model_params("satterc.dag.rothc")
        assert params_full == params_short

    def test_module_without_parameters_func_returns_empty(self):
        # resample has no keyword-only parameters with defaults
        assert get_model_params("satterc.dag.resample") == {}


class TestGenerateConfigCustomModules:
    def test_custom_module_with_params_written_to_config(self):
        config = generate_config(
            builtin_models=[],
            custom_modules=["satterc.dag.rothc"],
            paths=PATH_DEFAULTS,
        )
        # The nested dict for the custom module must contain the rothc params
        data = config._data
        assert "satterc" in data
        assert "dag.rothc" in data["satterc"]
        params = data["satterc"]["dag.rothc"]
        assert "n_years_spinup" in params

    def test_custom_module_without_params_written_as_empty_section(self):
        config = generate_config(
            builtin_models=[],
            custom_modules=["satterc.dag.resample"],
            paths=PATH_DEFAULTS,
        )
        data = config._data
        assert "satterc" in data
        assert "dag.resample" in data["satterc"]
        assert data["satterc"]["dag.resample"] == {}

    def test_custom_module_params_appear_in_toml_output(self):
        config = generate_config(
            builtin_models=[],
            custom_modules=["satterc.dag.rothc"],
            paths=PATH_DEFAULTS,
        )
        toml_str = str(config)
        assert '[satterc."dag.rothc"]' in toml_str
        assert "n_years_spinup" in toml_str
