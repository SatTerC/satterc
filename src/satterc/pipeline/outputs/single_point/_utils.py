from os import PathLike
from pathlib import Path

import pandas as pd
import xarray as xr


def dataset_to_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """Convert an output Dataset to a pandas DataFrame.

    Squeezes the size-1 'pixel' dimension if present, then calls to_dataframe().
    Any JAX-backed arrays are materialised to numpy in that call (via .values),
    which is the same implicit conversion that occurs when saving to NetCDF.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with dims (time, pixel) or (pixel,) where pixel has size 1,
        or dims (time,) / () for already-squeezed data.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by time (for temporal data) or a single-row DataFrame
        (for static data).
    """
    if "pixel" in ds.dims:
        ds = ds.squeeze("pixel", drop=True)
    return ds.to_dataframe()


def save_timeseries(df: pd.DataFrame, path: str | PathLike) -> None:
    """Save a DataFrame to CSV or Parquet, auto-detected by extension.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save. The index (typically a DatetimeIndex) is included.
    path : str | PathLike
        Destination path. Extension must be '.csv', '.parquet', or '.pq'.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        df.to_csv(path)
    elif suffix in (".parquet", ".pq"):
        df.to_parquet(path)
    else:
        raise ValueError(
            f"Unsupported format: '{suffix}'. Use '.csv', '.parquet', or '.pq'."
        )
