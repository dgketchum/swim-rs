"""Runoff calculations for daily water balance.

Pure physics kernels for computing surface runoff using either:
- SCS Curve Number method with dynamic antecedent moisture adjustment
- Infiltration-excess method using hourly precipitation
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

__all__ = [
    "curve_number_adjust",
    "scs_runoff",
    "scs_runoff_smoothed",
    "infiltration_excess",
]


@njit(cache=True, fastmath=True, parallel=True)
def curve_number_adjust(
    cn2: NDArray[np.float64],
    depl_surface: NDArray[np.float64],
    rew: NDArray[np.float64],
    tew: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Adjust Curve Number for antecedent moisture condition.

    CN varies between CNI (dry) and CNIII (wet) based on surface depletion.

    Physical constraints:
        - 0 < CN <= 100
        - CN increases (more runoff) as soil gets wetter
        - CN decreases (less runoff) as soil gets drier

    Parameters
    ----------
    cn2 : (n_fields,)
        Curve Number for average antecedent moisture condition (AMC II)
        Typically [30, 95] depending on land use and soil type
    depl_surface : (n_fields,)
        Surface layer depletion (mm), typically same as depl_ze
    rew : (n_fields,)
        Readily evaporable water (mm), defines wet threshold
    tew : (n_fields,)
        Total evaporable water (mm), contributes to dry threshold

    Returns
    -------
    cn : (n_fields,)
        Adjusted Curve Number for current moisture condition

    Notes
    -----
    The adjustment uses thresholds based on REW and TEW:
    - AWCIII = 0.5 * REW (wet condition threshold)
    - AWCI = 0.7 * REW + 0.3 * TEW (dry condition threshold)

    When surface depletion < AWCIII: CN = CNIII (wet, maximum runoff)
    When surface depletion > AWCI: CN = CNI (dry, minimum runoff)
    Between thresholds: linear interpolation

    References
    ----------
    Hawkins et al. (1985) CN adjustment for antecedent moisture
    """
    n = cn2.shape[0]
    cn = np.empty(n, dtype=np.float64)

    for i in prange(n):
        # Clip CN2 to valid range
        cn2_clipped = cn2[i]
        if cn2_clipped < 10.0:
            cn2_clipped = 10.0
        elif cn2_clipped > 100.0:
            cn2_clipped = 100.0

        # Calculate CNI (dry) and CNIII (wet) from CNII
        # These are empirical relationships from Hawkins et al. (1985)
        cn1 = cn2_clipped / (2.281 - 0.01281 * cn2_clipped)
        cn3 = cn2_clipped / (0.427 + 0.00573 * cn2_clipped)

        # Antecedent moisture thresholds
        awc3 = 0.5 * rew[i]  # Wet threshold
        awc1 = 0.7 * rew[i] + 0.3 * tew[i]  # Dry threshold

        # Ensure AWC1 > AWC3
        if awc1 <= awc3:
            awc1 = awc3 + 0.01

        ds = depl_surface[i]

        # Interpolate CN based on surface depletion
        if ds < awc3:
            # Wet condition - use CNIII
            cn[i] = cn3
        elif ds > awc1:
            # Dry condition - use CNI
            cn[i] = cn1
        else:
            # Linear interpolation between wet and dry
            frac = (ds - awc3) / (awc1 - awc3)
            cn[i] = cn3 + frac * (cn1 - cn3)

    return cn


@njit(cache=True, fastmath=True, parallel=True)
def scs_runoff(
    precip: NDArray[np.float64],
    cn: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate surface runoff using SCS Curve Number method.

    Q = (P - Ia)^2 / (P - Ia + S)  when P > Ia
    Q = 0                           when P <= Ia

    where:
    - Ia = 0.2 * S (initial abstraction)
    - S = 250 * (100/CN - 1) (potential maximum retention, mm)

    Physical constraints:
        - 0 <= Q <= P (runoff cannot exceed precipitation)
        - Q = 0 when P < Ia (all precip absorbed)
        - Q approaches P as CN approaches 100

    Parameters
    ----------
    precip : (n_fields,)
        Daily precipitation (mm), including rain + snowmelt
    cn : (n_fields,)
        Adjusted Curve Number for current conditions

    Returns
    -------
    sro : (n_fields,)
        Surface runoff (mm)
    s : (n_fields,)
        Potential maximum retention S (mm), for smoothing in next time step

    Notes
    -----
    The initial abstraction (Ia) represents water retained before runoff
    begins (interception, surface storage, infiltration). The standard
    assumption is Ia = 0.2*S, though some studies suggest Ia = 0.05*S.

    References
    ----------
    USDA-SCS (1972) National Engineering Handbook, Section 4: Hydrology
    """
    n = precip.shape[0]
    sro = np.empty(n, dtype=np.float64)
    s = np.empty(n, dtype=np.float64)

    for i in prange(n):
        # Calculate potential maximum retention S
        if cn[i] >= 100.0:
            s[i] = 0.0
        elif cn[i] <= 0.0:
            s[i] = 25000.0  # Very large S (essentially no runoff)
        else:
            s[i] = 250.0 * (100.0 / cn[i] - 1.0)

        # Initial abstraction
        ia = 0.2 * s[i]

        # Calculate runoff
        if precip[i] <= ia:
            sro[i] = 0.0
        else:
            ppt_net = precip[i] - ia
            denom = precip[i] + 0.8 * s[i]
            if denom < 1e-6:
                sro[i] = 0.0
            else:
                sro[i] = (ppt_net * ppt_net) / denom

        # Ensure runoff doesn't exceed precipitation
        if sro[i] > precip[i]:
            sro[i] = precip[i]

    return sro, s


@njit(cache=True, fastmath=True)
def scs_runoff_smoothed(
    precip: NDArray[np.float64],
    s1: NDArray[np.float64],
    s2: NDArray[np.float64],
    s3: NDArray[np.float64],
    s4: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate surface runoff using smoothed S values for irrigated fields.

    Averages runoff calculated with S values from the past 4 days to smooth
    the effects of irrigation-induced soil moisture changes.

    Physical constraints:
        - 0 <= Q <= P
        - Smoothing reduces runoff variability from irrigation cycles

    Parameters
    ----------
    precip : (n_fields,)
        Daily precipitation (mm)
    s1 : (n_fields,)
        S value from 1 day ago
    s2 : (n_fields,)
        S value from 2 days ago
    s3 : (n_fields,)
        S value from 3 days ago
    s4 : (n_fields,)
        S value from 4 days ago

    Returns
    -------
    sro : (n_fields,)
        Surface runoff (mm), averaged from 4 prior S values

    Notes
    -----
    This smoothing is applied to irrigated fields to prevent artificial
    runoff spikes that would occur if CN (and thus S) changed rapidly
    due to irrigation wetting the surface.
    """
    n = precip.shape[0]
    sro = np.empty(n, dtype=np.float64)

    for i in range(n):
        # Calculate runoff for each historical S value
        sro_total = 0.0

        for s_val in [s1[i], s2[i], s3[i], s4[i]]:
            ia = 0.2 * s_val
            if precip[i] <= ia:
                sro_i = 0.0
            else:
                ppt_net = precip[i] - ia
                denom = precip[i] + 0.8 * s_val
                if denom < 1e-6:
                    sro_i = 0.0
                else:
                    sro_i = (ppt_net * ppt_net) / denom
            sro_total += sro_i

        # Average of 4 runoff calculations
        sro[i] = sro_total / 4.0

        # Ensure runoff doesn't exceed precipitation
        if sro[i] > precip[i]:
            sro[i] = precip[i]

    return sro


@njit(cache=True, fastmath=True, parallel=True)
def infiltration_excess(
    hr_precip: NDArray[np.float64],
    ksat_hourly: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate surface runoff using infiltration-excess (Hortonian) method.

    Runoff occurs when precipitation intensity exceeds infiltration capacity.

    Q = sum over hours of max(P_h - Ksat, 0)

    Physical constraints:
        - 0 <= Q <= total precipitation
        - Q > 0 only when hourly precip exceeds hourly Ksat

    Parameters
    ----------
    hr_precip : (24, n_fields)
        Hourly precipitation (mm/hr) for each hour of the day
    ksat_hourly : (n_fields,)
        Saturated hydraulic conductivity (mm/hr)
        Represents maximum infiltration rate

    Returns
    -------
    sro : (n_fields,)
        Daily surface runoff (mm), sum of hourly excess

    Notes
    -----
    This method is more physically realistic than Curve Number for
    high-intensity rainfall events, as it accounts for precipitation
    intensity rather than just daily total.

    Ksat should represent the soil's infiltration capacity, which may
    be less than saturated hydraulic conductivity due to surface crusting,
    compaction, or other factors.

    References
    ----------
    Horton (1933) infiltration-excess runoff concept
    """
    n_hours = hr_precip.shape[0]
    n_fields = hr_precip.shape[1]
    sro = np.zeros(n_fields, dtype=np.float64)

    for j in prange(n_fields):
        for h in range(n_hours):
            excess = hr_precip[h, j] - ksat_hourly[j]
            if excess > 0.0:
                sro[j] += excess

    return sro
