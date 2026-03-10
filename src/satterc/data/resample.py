import xarray as xr
from hamilton.function_modifiers import parameterize_sources

# 1. Define your base variables
DAILY_VARIABLES = [
    "aridity_index",
    "co2_ppm",
    "dpm_rpm_ratio",
    "farmyard_manure_input",
    "fapar",
    "mean_growth_temperature",
    "plant_cover",
    "precipitation_mm",
    "pressure_pa",
    "ppfd_umol_m2_s1",
    "sunshine_fraction",
    "temperature_celcius",
    "vpd_pa",
    # Splash outputs
    "actual_evapotranspiration",
    "soil_moisture",
    "runoff",
]

# 2. Daily to Weekly Mapping
# Structure: {output_node_name: {function_parameter: source_node_name}}
WEEKLY_MAP = {f"{v}_weekly": {"daily_da": f"{v}_daily"} for v in DAILY_VARIABLES}


@parameterize_sources(**WEEKLY_MAP)
def aggregate_daily_to_weekly(daily_da: xr.DataArray) -> xr.DataArray:
    """Resamples daily xarray data to weekly mean."""
    return daily_da.resample(time="1W").mean()


WEEKLY_VARIABLES = [
    # Pmodel
    "gpp",
    "lue",
    "iwue",
    # Sgam
    "leaf_pool_size",
    "stem_pool_size",
    "root_pool_size",
    "leaf_respiration_loss",
    "stem_respiration_loss",
    "root_respiration_loss",
    "litter_to_soil",
    "disturbance_loss",
    "leaf_area_index",
    "npp",
    "cue",
]

# 3. Weekly to Monthly Mapping
# Note: This depends on the outputs of the previous function
MONTHLY_MAP = {
    f"{v}_monthly": {"weekly_da": f"{v}_weekly"}
    for v in DAILY_VARIABLES + WEEKLY_VARIABLES
}


@parameterize_sources(**MONTHLY_MAP)
def aggregate_weekly_to_monthly(weekly_da: xr.DataArray) -> xr.DataArray:
    """Aggregates weekly nodes into monthly nodes."""
    return weekly_da.resample(time="1ME").mean()
