"""Root growth dynamics calculations.

Pure physics kernels for computing root depth changes and associated
water redistribution between soil layers.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

__all__ = ["root_depth_from_kcb", "root_water_redistribution"]


@njit(cache=True, fastmath=True, parallel=True)
def root_depth_from_kcb(
    kcb: NDArray[np.float64],
    kc_min: NDArray[np.float64],
    kc_max: NDArray[np.float64],
    zr_max: NDArray[np.float64],
    zr_min: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate root depth from basal crop coefficient.

    Zr = Zr_min + (Zr_max - Zr_min) * (Kcb - Kc_min) / (Kc_max - Kc_min)

    Physical constraints:
        - Zr_min <= Zr <= Zr_max
        - Root depth scales linearly with vegetation development (Kcb)

    Parameters
    ----------
    kcb : (n_fields,)
        Basal crop coefficient
    kc_min : (n_fields,)
        Minimum crop coefficient (bare soil)
    kc_max : (n_fields,)
        Maximum crop coefficient (full cover)
    zr_max : (n_fields,)
        Maximum root depth (m)
    zr_min : (n_fields,)
        Minimum root depth (m), typically 0.1m

    Returns
    -------
    zr : (n_fields,)
        Root depth (m), bounded [Zr_min, Zr_max]

    Notes
    -----
    This assumes root development tracks above-ground vegetation development
    as indicated by Kcb. For annual crops, roots grow from planting to
    maturity. For perennials, root depth may remain relatively constant
    (use perennial flag in orchestration to handle this case).
    """
    n = kcb.shape[0]
    zr = np.empty(n, dtype=np.float64)

    for i in prange(n):
        kc_range = kc_max[i] - kc_min[i]
        if kc_range < 1e-6:
            # Avoid division by zero
            zr[i] = zr_min[i]
        else:
            # Linear interpolation based on Kcb
            frac = (kcb[i] - kc_min[i]) / kc_range

            # Clip fraction to [0, 1]
            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0

            zr[i] = zr_min[i] + (zr_max[i] - zr_min[i]) * frac

            # Ensure within bounds
            if zr[i] < zr_min[i]:
                zr[i] = zr_min[i]
            elif zr[i] > zr_max[i]:
                zr[i] = zr_max[i]

    return zr


@njit(cache=True, fastmath=True, parallel=True)
def root_water_redistribution(
    zr_new: NDArray[np.float64],
    zr_prev: NDArray[np.float64],
    zr_max: NDArray[np.float64],
    aw: NDArray[np.float64],
    depl_root: NDArray[np.float64],
    daw3: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Redistribute water between root zone and layer 3 during root growth.

    When roots grow deeper, they "capture" water from layer 3.
    When roots recede (senescence), water moves back to layer 3.

    Physical constraints:
        - Mass conservation: total water unchanged by root growth alone
        - daw3 >= 0
        - daw3 <= taw3

    Parameters
    ----------
    zr_new : (n_fields,)
        New root depth (m)
    zr_prev : (n_fields,)
        Previous root depth (m)
    zr_max : (n_fields,)
        Maximum root depth (m)
    aw : (n_fields,)
        Available water capacity (mm/m)
    depl_root : (n_fields,)
        Current root zone depletion (mm)
    daw3 : (n_fields,)
        Current available water in layer 3 (mm)

    Returns
    -------
    depl_root_new : (n_fields,)
        Updated root zone depletion (mm)
    daw3_new : (n_fields,)
        Updated layer 3 available water (mm)
    taw3_new : (n_fields,)
        Updated layer 3 total available water capacity (mm)

    Notes
    -----
    The redistribution logic:
    - If roots grow (delta_zr > 0): water from layer 3 is "absorbed" into
      the expanding root zone, reducing layer 3 storage
    - If roots recede (delta_zr < 0): some root zone water moves to layer 3

    This ensures mass conservation as the boundary between root zone and
    layer 3 shifts with root depth changes.
    """
    n = zr_new.shape[0]
    depl_root_new = np.empty(n, dtype=np.float64)
    daw3_new = np.empty(n, dtype=np.float64)
    taw3_new = np.empty(n, dtype=np.float64)

    for i in prange(n):
        delta_zr = zr_new[i] - zr_prev[i]

        # Calculate new layer 3 capacity
        layer3_depth = zr_max[i] - zr_new[i]
        if layer3_depth < 0.0:
            layer3_depth = 0.0
        taw3_new[i] = aw[i] * layer3_depth

        if abs(delta_zr) < 1e-6:
            # No change in root depth
            depl_root_new[i] = depl_root[i]
            daw3_new[i] = daw3[i]
        elif delta_zr > 0.0:
            # Roots growing deeper - capture water from layer 3

            # Calculate fraction of layer 3 being absorbed
            layer3_prev_depth = zr_max[i] - zr_prev[i]
            if layer3_prev_depth < 1e-6:
                frac_absorbed = 0.0
            else:
                frac_absorbed = delta_zr / layer3_prev_depth
                if frac_absorbed > 1.0:
                    frac_absorbed = 1.0

            # Water transferred from layer 3 to root zone
            water_from_layer3 = daw3[i] * frac_absorbed

            # Adjust root zone depletion (adding soil volume with its water)
            # New capacity = aw * delta_zr, water added = water_from_layer3
            added_capacity = aw[i] * delta_zr
            added_depletion = added_capacity - water_from_layer3
            depl_root_new[i] = depl_root[i] + added_depletion

            # Reduce layer 3 water
            daw3_new[i] = daw3[i] - water_from_layer3
            if daw3_new[i] < 0.0:
                daw3_new[i] = 0.0

        else:
            # Roots receding - water moves to layer 3

            # Calculate water content of root zone before recession
            rt_water_prev = (aw[i] * zr_prev[i]) - depl_root[i]

            # Calculate what fraction of root zone is being "released"
            frac_released = -delta_zr / zr_prev[i] if zr_prev[i] > 1e-6 else 0.0
            if frac_released > 1.0:
                frac_released = 1.0

            # Water transferred to layer 3
            water_to_layer3 = rt_water_prev * frac_released

            # Adjust root zone depletion (losing soil volume proportionally)
            # Scale depletion by the remaining fraction
            remaining_frac = 1.0 - frac_released
            depl_root_new[i] = depl_root[i] * remaining_frac

            # Add water to layer 3
            daw3_new[i] = daw3[i] + water_to_layer3

        # Ensure layer 3 water doesn't exceed capacity
        if daw3_new[i] > taw3_new[i]:
            daw3_new[i] = taw3_new[i]

    return depl_root_new, daw3_new, taw3_new
