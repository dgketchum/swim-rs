"""Soil evaporation coefficient calculations.

Pure physics kernels for computing Kr (evaporation reduction coefficient)
and Ke (soil evaporation coefficient) based on surface layer moisture status.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

__all__ = ["kr_reduction", "kr_damped", "ke_coefficient"]


@njit(cache=True, fastmath=True, parallel=True)
def kr_reduction(
    tew: NDArray[np.float64],
    depl_ze: NDArray[np.float64],
    rew: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate evaporation reduction coefficient from surface layer depletion.

    Kr = (TEW - De) / (TEW - REW)

    Physical constraints:
        - 0 <= Kr <= 1
        - Kr = 1 when surface is wet (De <= REW)
        - Kr = 0 when surface is dry (De >= TEW)
        - Kr decreases linearly as surface dries between REW and TEW

    Parameters
    ----------
    tew : (n_fields,)
        Total evaporable water in surface layer (mm), typically [15, 35]
    depl_ze : (n_fields,)
        Depletion of evaporation layer (mm), [0, TEW]
    rew : (n_fields,)
        Readily evaporable water (mm), typically [8, 12]
        Depletion threshold below which evaporation is unrestricted

    Returns
    -------
    kr : (n_fields,)
        Evaporation reduction coefficient, bounded [0, 1]

    Notes
    -----
    When the surface layer depletion is less than REW, soil evaporation
    proceeds at the maximum rate (Kr = 1). When depletion exceeds REW,
    evaporation becomes increasingly limited by the drying surface.

    References
    ----------
    FAO-56 Eq. 74: Kr = (TEW - De) / (TEW - REW) for De > REW
    """
    n = tew.shape[0]
    kr = np.empty(n, dtype=np.float64)

    for i in prange(n):
        denom = tew[i] - rew[i]
        if denom < 1e-6:
            # Avoid division by zero; if TEW ≈ REW, Kr = 1 if not depleted
            if depl_ze[i] < tew[i]:
                kr[i] = 1.0
            else:
                kr[i] = 0.0
        else:
            kr_raw = (tew[i] - depl_ze[i]) / denom

            # Clip to [0, 1]
            if kr_raw > 1.0:
                kr[i] = 1.0
            elif kr_raw < 0.0:
                kr[i] = 0.0
            else:
                kr[i] = kr_raw

    return kr


@njit(cache=True, fastmath=True, parallel=True)
def kr_damped(
    kr_current: NDArray[np.float64],
    kr_prev: NDArray[np.float64],
    kr_alpha: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Apply damping to Kr transitions for smoother day-to-day changes.

    Kr_new = Kr_prev + alpha * (Kr_current - Kr_prev)

    Physical constraints:
        - 0 <= Kr <= 1
        - Damping smooths rapid wetting/drying transitions
        - alpha controls responsiveness (1 = no damping, 0 = no change)

    Parameters
    ----------
    kr_current : (n_fields,)
        Current-day Kr calculated from depletion state
    kr_prev : (n_fields,)
        Previous-day Kr value
    kr_alpha : (n_fields,)
        Damping factor [0, 1], higher = faster response

    Returns
    -------
    kr : (n_fields,)
        Damped evaporation reduction coefficient, bounded [0, 1]

    Notes
    -----
    Damping prevents unrealistic jumps in evaporation rates that can occur
    with daily data. This is especially important when precipitation events
    cause rapid surface rewetting.
    """
    n = kr_current.shape[0]
    kr = np.empty(n, dtype=np.float64)

    for i in prange(n):
        kr_change = kr_current[i] - kr_prev[i]
        damped_change = kr_change * kr_alpha[i]
        kr_raw = kr_prev[i] + damped_change

        # Clip to [0, 1]
        if kr_raw > 1.0:
            kr[i] = 1.0
        elif kr_raw < 0.0:
            kr[i] = 0.0
        else:
            kr[i] = kr_raw

    return kr


@njit(cache=True, fastmath=True, parallel=True)
def ke_coefficient(
    kr: NDArray[np.float64],
    kc_max: NDArray[np.float64],
    kcb: NDArray[np.float64],
    few: NDArray[np.float64],
    ke_max: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate soil evaporation coefficient.

    Ke = min(Kr * (Kc_max - Kcb), few * Kc_max)
    Ke = min(Ke, Ke_max)

    Physical constraints:
        - 0 <= Ke <= Ke_max
        - Ke limited by available energy (Kc_max - Kcb)
        - Ke limited by exposed soil fraction (few * Kc_max)
        - Ke limited by maximum soil evaporation rate (Ke_max)

    Parameters
    ----------
    kr : (n_fields,)
        Evaporation reduction coefficient [0, 1]
    kc_max : (n_fields,)
        Maximum crop coefficient, typically [1.0, 1.3]
    kcb : (n_fields,)
        Basal crop coefficient (current transpiration demand)
    few : (n_fields,)
        Fraction of soil exposed and wetted [0, 1]
    ke_max : (n_fields,)
        Maximum soil evaporation coefficient, typically [1.0, 1.2]

    Returns
    -------
    ke : (n_fields,)
        Soil evaporation coefficient, bounded [0, Ke_max]

    Notes
    -----
    The dual constraint ensures that:
    1. Evaporation doesn't exceed the energy available after transpiration
    2. Evaporation is proportional to exposed soil area

    References
    ----------
    FAO-56 Eq. 71: Ke = min(Kr * (Kc_max - Kcb), few * Kc_max)
    """
    n = kr.shape[0]
    ke = np.empty(n, dtype=np.float64)

    for i in prange(n):
        # Energy-limited evaporation
        ke_energy = kr[i] * (kc_max[i] - kcb[i])

        # Area-limited evaporation
        ke_area = few[i] * kc_max[i]

        # Take minimum of both constraints
        ke_raw = ke_energy
        if ke_area < ke_raw:
            ke_raw = ke_area

        # Apply maximum Ke constraint
        if ke_raw > ke_max[i]:
            ke_raw = ke_max[i]

        # Ensure non-negative
        if ke_raw < 0.0:
            ke[i] = 0.0
        else:
            ke[i] = ke_raw

    return ke
