"""Transpiration stress coefficient calculations.

Pure physics kernels for computing Ks (water stress coefficient) based
on root zone depletion and plant-available water thresholds.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

__all__ = ["ks_stress", "ks_damped"]


@njit(cache=True, fastmath=True, parallel=True)
def ks_stress(
    taw: NDArray[np.float64],
    depl_root: NDArray[np.float64],
    raw: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate water stress coefficient from root zone depletion.

    Ks = (TAW - Dr) / (TAW - RAW)  when Dr > RAW
    Ks = 1                          when Dr <= RAW

    Physical constraints:
        - 0 <= Ks <= 1
        - Ks = 1 when root zone is adequately watered (Dr <= RAW)
        - Ks = 0 when root zone is fully depleted (Dr >= TAW)
        - Ks decreases linearly as root zone dries below RAW threshold

    Parameters
    ----------
    taw : (n_fields,)
        Total available water in root zone (mm)
        TAW = AWC * root_depth
    depl_root : (n_fields,)
        Root zone depletion (mm), [0, TAW]
    raw : (n_fields,)
        Readily available water (mm)
        RAW = MAD * TAW, where MAD is management allowable depletion

    Returns
    -------
    ks : (n_fields,)
        Water stress coefficient, bounded [0, 1]

    Notes
    -----
    When root zone depletion is less than RAW, transpiration proceeds at
    the potential rate (Ks = 1). When depletion exceeds RAW, plants begin
    experiencing water stress and reduce transpiration to conserve water.

    The RAW threshold represents the depletion level at which plants begin
    to close stomata. This varies by crop and is typically 40-60% of TAW.

    References
    ----------
    FAO-56 Eq. 84: Ks = (TAW - Dr) / (TAW - RAW) for Dr > RAW
    """
    n = taw.shape[0]
    ks = np.empty(n, dtype=np.float64)

    for i in prange(n):
        # No stress if depletion below RAW
        if depl_root[i] <= raw[i]:
            ks[i] = 1.0
        else:
            denom = taw[i] - raw[i]
            if denom < 1e-6:
                # Avoid division by zero; if TAW ≈ RAW, Ks = 0 if depleted
                ks[i] = 0.0
            else:
                ks_raw = (taw[i] - depl_root[i]) / denom

                # Clip to [0, 1]
                if ks_raw > 1.0:
                    ks[i] = 1.0
                elif ks_raw < 0.0:
                    ks[i] = 0.0
                else:
                    ks[i] = ks_raw

    return ks


@njit(cache=True, fastmath=True, parallel=True)
def ks_damped(
    ks_current: NDArray[np.float64],
    ks_prev: NDArray[np.float64],
    ks_alpha: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Apply damping to Ks transitions for smoother day-to-day changes.

    Ks_new = Ks_prev + alpha * (Ks_current - Ks_prev)

    Physical constraints:
        - 0 <= Ks <= 1
        - Damping smooths rapid stress/recovery transitions
        - alpha controls responsiveness (1 = no damping, 0 = no change)

    Parameters
    ----------
    ks_current : (n_fields,)
        Current-day Ks calculated from depletion state
    ks_prev : (n_fields,)
        Previous-day Ks value
    ks_alpha : (n_fields,)
        Damping factor [0, 1], higher = faster response

    Returns
    -------
    ks : (n_fields,)
        Damped water stress coefficient, bounded [0, 1]

    Notes
    -----
    Plants don't respond instantaneously to changes in soil moisture.
    Stomatal response has inertia - plants may remain partially stressed
    even after irrigation, or may not show stress immediately when
    depletion exceeds RAW. Damping captures this physiological lag.

    Lower alpha values (slower response) may be appropriate for:
    - Deep-rooted crops with large water reserves
    - Crops with waxy leaves or other drought adaptations

    Higher alpha values (faster response) may be appropriate for:
    - Shallow-rooted crops
    - High water demand crops
    - Sandy soils with rapid drainage
    """
    n = ks_current.shape[0]
    ks = np.empty(n, dtype=np.float64)

    for i in prange(n):
        ks_change = ks_current[i] - ks_prev[i]
        damped_change = ks_change * ks_alpha[i]
        ks_raw = ks_prev[i] + damped_change

        # Clip to [0, 1]
        if ks_raw > 1.0:
            ks[i] = 1.0
        elif ks_raw < 0.0:
            ks[i] = 0.0
        else:
            ks[i] = ks_raw

    return ks
