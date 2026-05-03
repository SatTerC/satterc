import json
import tomllib
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def load_timeseries(path: str | PathLike) -> xr.Dataset:
    """Load a single-point time series from CSV or Parquet, detected by extension.

    The file must have a datetime index (named 'time', or as the first/index column)
    and one column per variable. A size-1 'pixel' dimension is added so the resulting
    Dataset has dims (time, pixel), matching the shape expected by the rest of the
    pipeline.

    Parameters
    ----------
    path : str | PathLike
        Path to a .csv or .parquet (.pq) file.

    Returns
    -------
    xr.Dataset
        Dataset with dims (time, pixel) where pixel has a single coordinate value 0.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported format: '{suffix}'. Use '.csv', '.parquet', or '.pq'."
        )

    # Normalise index name so xarray assigns a 'time' dimension
    if "time" in df.columns:
        df = df.set_index("time")
    if df.index.name != "time":
        df.index.name = "time"
    df.index = pd.to_datetime(df.index)

    ds = df.to_xarray()  # dims: ('time',) per variable
    ds = ds.expand_dims({"pixel": [0]})  # ('pixel', 'time') per variable
    return ds.transpose("time", "pixel")  # -> ('time', 'pixel')


def load_static(path: str | PathLike) -> xr.Dataset:
    """Load single-point static inputs from JSON, YAML, or TOML, detected by extension.

    The file must be a flat key/value mapping of variable names to scalar numeric
    values. A size-1 'pixel' dimension is added so the resulting Dataset has
    dim (pixel,), matching the shape expected by the rest of the pipeline.

    Parameters
    ----------
    path : str | PathLike
        Path to a .json, .yaml/.yml, or .toml file.

    Returns
    -------
    xr.Dataset
        Dataset with dim (pixel,) where pixel has a single coordinate value 0.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".json":
        with open(p) as f:
            data: dict = json.load(f)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required for YAML static inputs. "
                "Install it with: uv add pyyaml"
            ) from e
        with open(p) as f:
            data = yaml.safe_load(f)
    elif suffix == ".toml":
        with open(p, "rb") as f:
            data = tomllib.load(f)
    else:
        raise ValueError(
            f"Unsupported format: '{suffix}'. Use '.json', '.yaml', '.yml', or '.toml'."
        )

    return xr.Dataset(
        {
            k: xr.DataArray(np.asarray([v], dtype=float), dims=["pixel"])
            for k, v in data.items()
        },
        coords={"pixel": [0]},
    )
