import numpy as np
from numpy.typing import NDArray


def aridity_index(
    actual_evapotranspiration: NDArray[np.float64],
    precipitation_mm: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate the aridity index.

    This is a dimensionless ratio of actual evapotranspiration to precipitation.

    This function is intended to act as a node in a Hamilton DAG.

    Parameters
    ----------
    actual_evapotranspiration
        Actual evapotranspiration (mm).
    precipitation_mm
        Precipitation (mm).

    Returns
    -------
    NDArray[np.float64]
        Aridity index (dimensionless ratio of actual evapotranspiration to precipitation).
    """
    return actual_evapotranspiration / precipitation_mm
