"""Type stub for the f2py extension built by ``just build-acm``.

f2py generates no stubs, and the extension is not committed, so without this
pyright cannot resolve the import at all. Keep it in step with the shim in
``fortran/acm_gpp_et/acm_gpp_et_f2py.f90``.
"""

import numpy as np
from numpy.typing import NDArray

def acm_pixel(
    met: NDArray[np.float64],
    pars: NDArray[np.float64],
    lat: float,
    soil_frac_clay_in: NDArray[np.float64],
    soil_frac_sand_in: NDArray[np.float64],
    nomet: int = ...,
    nodays: int = ...,
    nopars: int = ...,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]: ...
