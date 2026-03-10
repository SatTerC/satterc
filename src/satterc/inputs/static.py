from hamilton.function_modifiers import extract_fields
import xarray as xr

STATIC_INPUT_VARIABLES = [
    "elevation",
    "plant_type",
    "max_soil_moisture",
]


@extract_fields(STATIC_INPUT_VARIABLES)
def static_inputs(static_inputs_dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    return {
        str(var): static_inputs_dataset[var] for var in static_inputs_dataset.data_vars
    }
