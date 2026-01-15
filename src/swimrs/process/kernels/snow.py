"""Snow dynamics calculations.

Pure physics kernels for computing snow accumulation, albedo evolution,
and snowmelt using temperature and radiation inputs.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

__all__ = ["partition_precip", "albedo_decay", "degree_day_melt", "snow_water_equivalent"]


@njit(cache=True, fastmath=True, parallel=True)
def partition_precip(
    precip: NDArray[np.float64],
    temp_avg: NDArray[np.float64],
    threshold: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Partition precipitation into rain and snow based on temperature.

    Physical constraints:
        - rain + snow = precip
        - snow = 0 when T >= threshold
        - rain = 0 when T < threshold

    Parameters
    ----------
    precip : (n_fields,)
        Total precipitation (mm)
    temp_avg : (n_fields,)
        Average daily temperature (°C)
    threshold : float
        Temperature threshold for rain/snow partitioning (°C)
        Default 1.0°C (slightly above freezing to account for
        precipitation forming at higher altitudes)

    Returns
    -------
    rain : (n_fields,)
        Liquid precipitation (mm)
    snow : (n_fields,)
        Frozen precipitation / snowfall (mm)

    Notes
    -----
    This simple threshold approach assumes all precipitation is either
    rain or snow. More sophisticated models use a transition zone
    (e.g., -1 to 3°C) where mixed precipitation occurs.
    """
    n = precip.shape[0]
    rain = np.empty(n, dtype=np.float64)
    snow = np.empty(n, dtype=np.float64)

    for i in prange(n):
        if temp_avg[i] < threshold:
            # Cold - all snow
            rain[i] = 0.0
            snow[i] = precip[i]
        else:
            # Warm - all rain
            rain[i] = precip[i]
            snow[i] = 0.0

    return rain, snow


@njit(cache=True, fastmath=True, parallel=True)
def albedo_decay(
    albedo_prev: NDArray[np.float64],
    snow_fall: NDArray[np.float64],
    decay_rate: float = 0.12,
    decay_rate_no_snow: float = 0.05,
    fresh_snow_threshold: float = 3.0,
    albedo_min: float = 0.45,
    albedo_max: float = 0.98,
) -> NDArray[np.float64]:
    """
    Update snow albedo based on snowfall and aging.

    Physical constraints:
        - albedo_min <= albedo <= albedo_max
        - Fresh snow has high albedo (~0.98)
        - Albedo decays exponentially with time since snowfall

    Parameters
    ----------
    albedo_prev : (n_fields,)
        Previous day's snow albedo
    snow_fall : (n_fields,)
        Fresh snowfall (mm)
    decay_rate : float
        Albedo decay rate constant when snow is present
        albedo_new = albedo_min + (albedo_prev - albedo_min) * exp(-decay_rate)
    decay_rate_no_snow : float
        Slower decay rate when no fresh snow falls
    fresh_snow_threshold : float
        Snowfall amount (mm) that resets albedo to maximum
    albedo_min : float
        Minimum albedo for old/dirty snow, typically 0.45
    albedo_max : float
        Maximum albedo for fresh snow, typically 0.98

    Returns
    -------
    albedo : (n_fields,)
        Updated snow albedo

    Notes
    -----
    Snow albedo decreases over time due to:
    - Grain coarsening (metamorphism)
    - Accumulation of dust and debris
    - Partial melting and refreezing

    Fresh snowfall "resets" the albedo to near-maximum values.

    References
    ----------
    USACE (1956) Snow Hydrology manual albedo decay formulation
    """
    n = albedo_prev.shape[0]
    albedo = np.empty(n, dtype=np.float64)

    for i in prange(n):
        if snow_fall[i] > fresh_snow_threshold:
            # Fresh snow - reset to maximum albedo
            albedo[i] = albedo_max
        elif snow_fall[i] > 0.0:
            # Some snow but not enough to fully refresh
            # Use standard decay rate
            albedo[i] = albedo_min + (albedo_prev[i] - albedo_min) * np.exp(-decay_rate)
        else:
            # No snow - slower decay
            albedo[i] = albedo_min + (albedo_prev[i] - albedo_min) * np.exp(-decay_rate_no_snow)

        # Ensure within bounds
        if albedo[i] < albedo_min:
            albedo[i] = albedo_min
        elif albedo[i] > albedo_max:
            albedo[i] = albedo_max

    return albedo


@njit(cache=True, fastmath=True, parallel=True)
def degree_day_melt(
    swe: NDArray[np.float64],
    temp_max: NDArray[np.float64],
    temp_avg: NDArray[np.float64],
    srad: NDArray[np.float64],
    albedo: NDArray[np.float64],
    swe_alpha: NDArray[np.float64],
    swe_beta: NDArray[np.float64],
    temp_threshold: float = 0.0,
    base_temp: float = 1.8,
) -> NDArray[np.float64]:
    """
    Calculate snowmelt using combined radiation and degree-day approach.

    Melt = max(0, (1 - albedo) * srad * alpha + (T_avg - base_temp) * beta)

    Physical constraints:
        - 0 <= melt <= SWE (can't melt more than available)
        - melt = 0 when T_max <= threshold (no melt below freezing)
        - Radiation component depends on albedo and incoming solar
        - Temperature component uses degree-day factor

    Parameters
    ----------
    swe : (n_fields,)
        Snow water equivalent (mm) before melt
    temp_max : (n_fields,)
        Maximum daily temperature (°C)
    temp_avg : (n_fields,)
        Average daily temperature (°C)
    srad : (n_fields,)
        Incoming shortwave radiation (MJ/m²/day)
    albedo : (n_fields,)
        Snow albedo [0.45, 0.98]
    swe_alpha : (n_fields,)
        Radiation melt coefficient (mm/MJ/m²)
        Calibration parameter, typically [-0.5, 1.0]
    swe_beta : (n_fields,)
        Degree-day melt factor (mm/°C/day)
        Calibration parameter, typically [0.5, 2.5]
    temp_threshold : float
        Temperature threshold for any melt to occur (°C)
    base_temp : float
        Base temperature for degree-day calculation (°C)

    Returns
    -------
    melt : (n_fields,)
        Snowmelt (mm), bounded [0, SWE]

    Notes
    -----
    The combined approach captures both:
    - Radiation-driven melt (important in spring with high sun angles)
    - Temperature-driven melt (sensible heat exchange)

    The radiation term is modulated by (1 - albedo) to account for
    the fact that fresh snow reflects most incoming radiation.

    References
    ----------
    Adapted from US Army Corps of Engineers (1956) Snow Hydrology
    """
    n = swe.shape[0]
    melt = np.empty(n, dtype=np.float64)

    for i in prange(n):
        # No melt if T_max below threshold or no snow present
        if temp_max[i] <= temp_threshold or swe[i] <= 0.0:
            melt[i] = 0.0
        else:
            # Radiation melt component
            rad_melt = (1.0 - albedo[i]) * srad[i] * swe_alpha[i]

            # Degree-day melt component
            dd_melt = (temp_avg[i] - base_temp) * swe_beta[i]

            # Total potential melt
            melt_potential = rad_melt + dd_melt

            # Ensure non-negative and bounded by available SWE
            if melt_potential < 0.0:
                melt[i] = 0.0
            elif melt_potential > swe[i]:
                melt[i] = swe[i]
            else:
                melt[i] = melt_potential

    return melt


@njit(cache=True, fastmath=True, parallel=True)
def snow_water_equivalent(
    swe_prev: NDArray[np.float64],
    snow_fall: NDArray[np.float64],
    melt: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Update snow water equivalent after accumulation and melt.

    SWE_new = SWE_prev + snowfall - melt

    Physical constraints:
        - SWE >= 0
        - SWE increases with snowfall
        - SWE decreases with melt

    Parameters
    ----------
    swe_prev : (n_fields,)
        Snow water equivalent before today (mm)
    snow_fall : (n_fields,)
        Fresh snowfall (mm)
    melt : (n_fields,)
        Snowmelt (mm)

    Returns
    -------
    swe : (n_fields,)
        Updated snow water equivalent (mm)

    Notes
    -----
    This simple mass balance assumes no sublimation, wind redistribution,
    or other snow loss mechanisms beyond melt.
    """
    n = swe_prev.shape[0]
    swe = np.empty(n, dtype=np.float64)

    for i in prange(n):
        swe_new = swe_prev[i] + snow_fall[i] - melt[i]

        # Ensure non-negative
        if swe_new < 0.0:
            swe[i] = 0.0
        else:
            swe[i] = swe_new

    return swe
