"""Water balance calculations.

Pure physics kernels for computing deep percolation, layer-3 storage,
and soil water content updates.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

__all__ = [
    "deep_percolation",
    "layer3_storage",
    "root_zone_depletion",
    "total_soil_water",
    "actual_et",
]


@njit(cache=True, fastmath=True, parallel=True)
def deep_percolation(
    depl_root: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate deep percolation from negative root zone depletion.

    When depletion goes negative (excess water), the excess percolates
    to deeper layers.

    Physical constraints:
        - dperc >= 0
        - dperc = -depl_root when depl_root < 0
        - Updated depl_root >= 0 after percolation

    Parameters
    ----------
    depl_root : (n_fields,)
        Root zone depletion (mm), may be negative if excess water

    Returns
    -------
    dperc : (n_fields,)
        Deep percolation (mm) leaving the root zone
    depl_root_updated : (n_fields,)
        Updated root zone depletion (mm), non-negative

    Notes
    -----
    Negative depletion indicates the root zone has more water than
    field capacity. This excess drains to deeper layers.
    """
    n = depl_root.shape[0]
    dperc = np.empty(n, dtype=np.float64)
    depl_root_updated = np.empty(n, dtype=np.float64)

    for i in prange(n):
        if depl_root[i] < 0.0:
            # Excess water percolates
            dperc[i] = -depl_root[i]
            depl_root_updated[i] = 0.0
        else:
            dperc[i] = 0.0
            depl_root_updated[i] = depl_root[i]

    return dperc, depl_root_updated


@njit(cache=True, fastmath=True, parallel=True)
def layer3_storage(
    daw3: NDArray[np.float64],
    taw3: NDArray[np.float64],
    gross_dperc: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Update layer-3 (below root zone) water storage.

    Layer 3 is the soil between current root depth and maximum root depth.
    It accumulates deep percolation and can overflow when full.

    Physical constraints:
        - 0 <= daw3 <= taw3
        - Overflow occurs when daw3 + inflow > taw3

    Parameters
    ----------
    daw3 : (n_fields,)
        Current available water in layer 3 (mm)
    taw3 : (n_fields,)
        Total available water capacity in layer 3 (mm)
        taw3 = AWC * (zr_max - zr)
    gross_dperc : (n_fields,)
        Gross deep percolation entering layer 3 (mm)
        Includes dperc + 10% of irrigation

    Returns
    -------
    daw3_updated : (n_fields,)
        Updated available water in layer 3 (mm)
    dperc_out : (n_fields,)
        Deep percolation leaving layer 3 (overflow) (mm)

    Notes
    -----
    Layer 3 serves as a reservoir that can supply water to roots as
    they grow deeper. When layer 3 is full, excess water percolates
    below the maximum root zone and is lost from the system.
    """
    n = daw3.shape[0]
    daw3_updated = np.empty(n, dtype=np.float64)
    dperc_out = np.empty(n, dtype=np.float64)

    for i in prange(n):
        # Add incoming percolation
        daw3_new = daw3[i] + gross_dperc[i]

        # Ensure non-negative
        if daw3_new < 0.0:
            daw3_new = 0.0

        # Check for overflow
        if daw3_new > taw3[i]:
            dperc_out[i] = daw3_new - taw3[i]
            daw3_updated[i] = taw3[i]
        else:
            dperc_out[i] = 0.0
            daw3_updated[i] = daw3_new

    return daw3_updated, dperc_out


@njit(cache=True, fastmath=True, parallel=True)
def root_zone_depletion(
    depl_root_prev: NDArray[np.float64],
    etc_act: NDArray[np.float64],
    ppt_inf: NDArray[np.float64],
    irr_sim: NDArray[np.float64],
    gw_sim: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Update root zone depletion based on ET and water inputs.

    Dr_new = Dr_prev + ETc_act - Ppt_inf - Irr - GW

    Physical constraints:
        - Depletion increases with ET
        - Depletion decreases with precipitation, irrigation, groundwater

    Parameters
    ----------
    depl_root_prev : (n_fields,)
        Previous day's root zone depletion (mm)
    etc_act : (n_fields,)
        Actual evapotranspiration (mm)
    ppt_inf : (n_fields,)
        Infiltrating precipitation (rain + melt - runoff) (mm)
    irr_sim : (n_fields,)
        Simulated irrigation (mm)
    gw_sim : (n_fields,)
        Groundwater subsidy (mm)

    Returns
    -------
    depl_root : (n_fields,)
        Updated root zone depletion (mm)
        Note: May be negative (excess water) - caller should handle

    Notes
    -----
    This is the core water balance equation. Negative depletion indicates
    excess water that should percolate to deeper layers.
    """
    n = depl_root_prev.shape[0]
    depl_root = np.empty(n, dtype=np.float64)

    for i in prange(n):
        depl_root[i] = depl_root_prev[i] + etc_act[i] - ppt_inf[i] - irr_sim[i] - gw_sim[i]

    return depl_root


@njit(cache=True, fastmath=True, parallel=True)
def total_soil_water(
    aw: NDArray[np.float64],
    zr: NDArray[np.float64],
    depl_root: NDArray[np.float64],
    daw3: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate total soil water in root zone plus layer 3.

    SW = AWC * Zr - Dr + DAW3

    Parameters
    ----------
    aw : (n_fields,)
        Available water capacity (mm/m)
    zr : (n_fields,)
        Root depth (m)
    depl_root : (n_fields,)
        Root zone depletion (mm)
    daw3 : (n_fields,)
        Available water in layer 3 (mm)

    Returns
    -------
    soil_water : (n_fields,)
        Total soil water content (mm)
    """
    n = aw.shape[0]
    soil_water = np.empty(n, dtype=np.float64)

    for i in prange(n):
        soil_water[i] = (aw[i] * zr[i]) - depl_root[i] + daw3[i]

    return soil_water


@njit(cache=True, fastmath=True, parallel=True)
def actual_et(
    ks: NDArray[np.float64],
    kcb: NDArray[np.float64],
    fc: NDArray[np.float64],
    ke: NDArray[np.float64],
    kc_max: NDArray[np.float64],
    refet: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate actual evapotranspiration using dual crop coefficient.

    Kc_act = min(fc * Ks * Kcb + Ke, Kc_max)
    ETc_act = Kc_act * RefET

    Physical constraints:
        - 0 <= Kc_act <= Kc_max
        - ETc_act = 0 when RefET = 0
        - Transpiration reduced by Ks (stress) and fc (cover)

    Parameters
    ----------
    ks : (n_fields,)
        Water stress coefficient [0, 1]
    kcb : (n_fields,)
        Basal crop coefficient
    fc : (n_fields,)
        Fractional vegetation cover [0, 1]
    ke : (n_fields,)
        Soil evaporation coefficient
    kc_max : (n_fields,)
        Maximum crop coefficient
    refet : (n_fields,)
        Reference evapotranspiration (mm/day)

    Returns
    -------
    kc_act : (n_fields,)
        Actual crop coefficient
    etc_act : (n_fields,)
        Actual evapotranspiration (mm/day)
    """
    n = ks.shape[0]
    kc_act = np.empty(n, dtype=np.float64)
    etc_act = np.empty(n, dtype=np.float64)

    for i in prange(n):
        # Kc_act = fc * Ks * Kcb + Ke (FAO-56 dual crop coefficient)
        kc_raw = fc[i] * ks[i] * kcb[i] + ke[i]

        # Cap at maximum
        if kc_raw > kc_max[i]:
            kc_act[i] = kc_max[i]
        elif kc_raw < 0.0:
            kc_act[i] = 0.0
        else:
            kc_act[i] = kc_raw

        # Calculate actual ET
        etc_act[i] = kc_act[i] * refet[i]

    return kc_act, etc_act
