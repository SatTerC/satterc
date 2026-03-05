import xarray as xr
from hamilton.function_modifiers import parameterize_sources

# 1. Define your base variables
VARIABLES = [
    "aridity_index",
    "carbon_input",
    "co2_ppm",
    "dates",
    "dpm_rpm_ratio",
    "evaporation",
    "farmyard_manure_input",
    "fapar",
    "mean_growth_temperature",
    "plant_cover",
    "precipitation_mm",
    "pressure_pa",
    "ppfd_umol_m2_s1",
    "soil_moisture",
    "sunshine_fraction",
    "temperature_celcius",
    "vpd_pa",
]

# 2. Daily to Weekly Mapping
# Structure: {output_node_name: {function_parameter: source_node_name}}
WEEKLY_MAP = {f"{v}_weekly": {"daily_da": f"{v}_daily"} for v in VARIABLES}


@parameterize_sources(**WEEKLY_MAP)
def aggregate_daily_to_weekly(daily_da: xr.DataArray) -> xr.DataArray:
    """Resamples daily xarray data to weekly mean."""
    return daily_da.resample(time="1W").mean()


# 3. Weekly to Monthly Mapping
# Note: This depends on the outputs of the previous function
MONTHLY_MAP = {f"{v}_monthly": {"weekly_da": f"{v}_weekly"} for v in VARIABLES}


@parameterize_sources(**MONTHLY_MAP)
def aggregate_weekly_to_monthly(weekly_da: xr.DataArray) -> xr.DataArray:
    """Aggregates weekly nodes into monthly nodes."""
    return weekly_da.resample(time="1M").mean()
