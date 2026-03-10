from pathlib import Path

from hamilton.function_modifiers import extract_fields
import xarray as xr

from ._utils import load_dataset, stack_spatial_dims

STATIC_INPUTS = [
    "elevation",
    "plant_type",
    "max_soil_moisture",
    "clay_content",
    "soil_depth",
    "organic_carbon_stocks",
]


def static_inputs(static_inputs_path: Path) -> xr.Dataset:
    return load_dataset(static_inputs_path)


def static_inputs_stacked(static_inputs: xr.Dataset) -> xr.Dataset:
    return stack_spatial_dims(static_inputs)


@extract_fields(STATIC_INPUTS)
def unpack_static_inputs(static_inputs_stacked: xr.Dataset) -> dict[str, xr.DataArray]:
    return {
        str(var): static_inputs_stacked[var] for var in static_inputs_stacked.data_vars
    }
