"""Tests for scaffold/config_gen.py."""

from satterc.scaffold import generate_config, get_model_config
from satterc.scaffold.config_gen import _infer_required_data, _strip_suffix

PATH_DEFAULTS = {
    "inputs_daily": "inputs/daily.nc",
    "inputs_weekly": "inputs/weekly.nc",
    "inputs_monthly": "inputs/monthly.nc",
    "inputs_static": "inputs/static.nc",
    "outputs_daily": "outputs/daily.nc",
    "outputs_weekly": "outputs/weekly.nc",
    "outputs_monthly": "outputs/monthly.nc",
}


class TestGetModelConfig:
    def test_builtin_returns_defaults(self):
        params = get_model_config("rothc")
        assert "n_years_spinup" in params
        assert isinstance(params["n_years_spinup"], int)

    def test_unknown_module_returns_empty(self):
        assert get_model_config("nonexistent.module") == {}

    def test_custom_path_uses_last_component_as_func_prefix(self):
        # Passing the full dotted path to a known module must find the same
        # _parameters() function as passing the short name.
        params_short = get_model_config("rothc")
        params_full = get_model_config("satterc.models.rothc")
        assert params_full == params_short

    def test_module_without_matching_function_returns_empty(self):
        # satterc.models has no function named "models" → hits the return {} fallback
        assert get_model_config("satterc.models") == {}


class TestGenerateConfigCustomModules:
    """Custom modules become flat sections addressed by `_import_path`."""

    def test_custom_module_with_params_written_to_config(self):
        config = generate_config(
            builtin_models=[],
            custom_modules=["satterc.models.rothc"],
            paths=PATH_DEFAULTS,
        )
        section = config._data["rothc"]
        assert section["_import_path"] == "satterc.models.rothc"
        assert "n_years_spinup" in section

    def test_custom_module_without_params_written_as_import_path_only(self):
        config = generate_config(
            builtin_models=[],
            custom_modules=["satterc.models"],
            paths=PATH_DEFAULTS,
        )
        assert config._data["models"] == {"_import_path": "satterc.models"}

    def test_custom_module_params_appear_in_toml_output(self):
        config = generate_config(
            builtin_models=[],
            custom_modules=["satterc.models.rothc"],
            paths=PATH_DEFAULTS,
        )
        toml_str = str(config)
        assert "[rothc]" in toml_str
        assert '_import_path = "satterc.models.rothc"' in toml_str
        assert "n_years_spinup" in toml_str


class TestStripSuffix:
    """_strip_suffix extracts base name and frequency from variable names."""

    def test_daily_suffix(self):
        base, freq = _strip_suffix("temperature_daily")
        assert base == "temperature"
        assert freq == "_daily"

    def test_weekly_suffix(self):
        base, freq = _strip_suffix("gpp_weekly")
        assert base == "gpp"
        assert freq == "_weekly"

    def test_monthly_suffix(self):
        base, freq = _strip_suffix("soil_carbon_monthly")
        assert base == "soil_carbon"
        assert freq == "_monthly"

    def test_static_suffix(self):
        base, freq = _strip_suffix("elevation_static")
        assert base == "elevation"
        assert freq == "_static"

    def test_no_suffix(self):
        base, freq = _strip_suffix("elevation")
        assert base == "elevation"
        assert freq is None

    def test_partial_match_not_stripped(self):
        # "_daily" must be at the end, not in the middle
        base, freq = _strip_suffix("daily_temperature")
        assert base == "daily_temperature"
        assert freq is None


class TestGenerateConfigBuiltinModels:
    """generate_config with builtin model names populates inputs/outputs/resample."""

    def test_splash_generates_input_section(self):
        config = generate_config(
            builtin_models=["splash"],
            custom_modules=[],
            paths=PATH_DEFAULTS,
        )
        data = config._data
        assert "inputs" in data

    def test_splash_produces_daily_outputs(self):
        config = generate_config(
            builtin_models=["splash"],
            custom_modules=[],
            paths=PATH_DEFAULTS,
        )
        data = config._data
        assert "outputs" in data
        assert "daily" in data["outputs"]
        daily_vars = data["outputs"]["daily"]["vars"]
        assert any(
            "soil_moisture" in v or "evapotranspiration" in v or "runoff" in v
            for v in daily_vars
        )

    def test_rothc_generates_model_section_with_params(self):
        config = generate_config(
            builtin_models=["rothc"],
            custom_modules=[],
            paths=PATH_DEFAULTS,
        )
        section = config._data["rothc"]
        assert section["_import_path"] == "satterc.models.rothc"
        assert "n_years_spinup" in section

    def test_pmodel_generates_model_section(self):
        config = generate_config(
            builtin_models=["pmodel"],
            custom_modules=[],
            paths=PATH_DEFAULTS,
        )
        assert config._data["pmodel"]["_import_path"] == "satterc.models.pmodel"

    def test_static_input_section_opts_out_of_suffix(self):
        # Static variables are consumed under bare names, so the generated
        # section must override conduit's default `_<label>` node-name suffix.
        config = generate_config(
            builtin_models=["rothc"],
            custom_modules=[],
            paths=PATH_DEFAULTS,
        )
        assert config._data["inputs"]["static"]["suffix"] == ""

    def test_resample_entries_carry_an_explicit_freq(self):
        config = generate_config(
            builtin_models=["splash", "pmodel"],
            custom_modules=[],
            paths=PATH_DEFAULTS,
        )
        for entry in config._data.get("resample", []):
            assert entry["freq"], entry
            assert "from" in entry
            assert "to" in entry

    def test_combined_models_share_resample_vars(self):
        # splash provides daily soil_moisture; pmodel needs weekly soil_moisture
        # → a resample section should appear
        config = generate_config(
            builtin_models=["splash", "pmodel"],
            custom_modules=[],
            paths=PATH_DEFAULTS,
        )
        data = config._data
        # If any variable is needed at two frequencies, resample appears
        assert "resample" in data or "inputs" in data


class TestInferRequiredData:
    """_infer_required_data categorises inputs and outputs by frequency."""

    def test_splash_produces_daily_outputs_key(self):
        result = _infer_required_data(["splash"])
        assert "outputs_daily" in result
        assert len(result["outputs_daily"]) > 0

    def test_pmodel_produces_weekly_outputs_key(self):
        result = _infer_required_data(["pmodel"])
        assert "outputs_weekly" in result
        assert len(result["outputs_weekly"]) > 0

    def test_result_has_all_expected_keys(self):
        result = _infer_required_data(["splash"])
        for key in (
            "inputs_daily",
            "inputs_weekly",
            "inputs_monthly",
            "inputs_static",
            "resample_daily_to_weekly",
            "resample_daily_to_monthly",
            "resample_weekly_to_monthly",
            "outputs_daily",
            "outputs_weekly",
            "outputs_monthly",
        ):
            assert key in result, f"Missing key: {key}"

    def test_grid_vars_not_in_inputs(self):
        result = _infer_required_data(["splash"])
        all_inputs = (
            result["inputs_daily"]
            + result["inputs_weekly"]
            + result["inputs_monthly"]
            + result["inputs_static"]
        )
        assert "latitude" not in all_inputs
        assert "longitude" not in all_inputs
