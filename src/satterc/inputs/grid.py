from hamilton.function_modifiers import unpack_fields
import xarray as xr
import rioxarray as rioxarray  # only used indirectly via dataset.rio; suppress linter errors
from pyproj import Transformer


@unpack_fields("latitude", "longitude")
def coordinate_grid(
    daily_inputs_dataset: xr.Dataset,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Computes both latitude and longitude grids and unpacks them into individual nodes.

    Parameters
    ----------
    daily_inputs_dataset : xr.Dataset
        The loaded dataset with coordinate reference system information.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        Tuple of "latitude" and "longitude" DataArrays.
    """
    crs = daily_inputs_dataset.rio.crs
    if crs is None:
        raise ValueError("No CRS found.")

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    # 1. Get the names of the spatial dims (usually 'x' and 'y')
    x_dim = daily_inputs_dataset.rio.x_dim
    y_dim = daily_inputs_dataset.rio.y_dim

    # 2. Extract values. If stacked, these are already 1D arrays of length 'pixel'
    # No meshgrid required!
    x_values = daily_inputs_dataset[x_dim].values
    y_values = daily_inputs_dataset[y_dim].values

    # 3. Transform directly
    lons, lats = transformer.transform(x_values, y_values)

    # 4. Wrap back into DataArrays
    # We reuse the existing 'pixel' coordinate from the input for perfect alignment
    lat_da = xr.DataArray(
        lats,
        coords={"pixel": daily_inputs_dataset.pixel},
        dims=("pixel",),
        name="latitude",
    )
    lon_da = xr.DataArray(
        lons,
        coords={"pixel": daily_inputs_dataset.pixel},
        dims=("pixel",),
        name="longitude",
    )

    return lat_da, lon_da
