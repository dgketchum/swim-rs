"""Irrigation demand and application calculations.

Pure physics kernels for computing irrigation requirements and
simulated irrigation amounts based on depletion and management rules.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

__all__ = ["irrigation_demand", "groundwater_subsidy"]


@njit(cache=True, fastmath=True, parallel=True)
def irrigation_demand(
    depl_root: NDArray[np.float64],
    raw: NDArray[np.float64],
    max_irr_rate: NDArray[np.float64],
    irr_flag: NDArray[np.bool_],
    temp_avg: NDArray[np.float64],
    irr_continue: NDArray[np.float64],
    next_day_irr: NDArray[np.float64],
    temp_threshold: float = 5.0,
    refill_factor: float = 1.1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate irrigation demand and simulated application.

    Irrigation is triggered when:
    1. It's an irrigation day (irr_flag = True) AND depletion > RAW, OR
    2. There is carryover irrigation from previous day (irr_continue > 0)
    3. Temperature is above threshold (no irrigation when frozen)

    Physical constraints:
        - irr_sim <= max_irr_rate (daily application limit)
        - irr_sim targets refilling to slightly above field capacity
        - Excess demand carries over to next day

    Parameters
    ----------
    depl_root : (n_fields,)
        Root zone depletion (mm)
    raw : (n_fields,)
        Readily available water threshold (mm)
    max_irr_rate : (n_fields,)
        Maximum irrigation rate (mm/day), e.g., 25 mm/day for sprinkler
    irr_flag : (n_fields,)
        Boolean flag for irrigation day (from schedule)
    temp_avg : (n_fields,)
        Average daily temperature (°C)
    irr_continue : (n_fields,)
        Previous day's continuation flag (leftover irrigation needed)
    next_day_irr : (n_fields,)
        Carryover irrigation amount from previous day (mm)
    temp_threshold : float
        Minimum temperature for irrigation (°C)
    refill_factor : float
        Factor to overfill slightly (1.1 = 110% of depletion)

    Returns
    -------
    irr_sim : (n_fields,)
        Simulated irrigation amount (mm)
    irr_continue_new : (n_fields,)
        Continuation flag for next day
    next_day_irr_new : (n_fields,)
        Carryover amount for next day (mm)

    Notes
    -----
    The continuation logic matches legacy model behavior:
    - Continuation is set when irr_flag AND target_amount > max_irr_rate
      (independent of whether depl > RAW)
    - This allows irrigation to continue over multiple days even after
      depletion drops below RAW

    The refill_factor (typically 1.1) ensures a small buffer above field
    capacity to account for immediate drainage and provide a margin.
    """
    n = depl_root.shape[0]
    irr_sim = np.empty(n, dtype=np.float64)
    irr_continue_new = np.empty(n, dtype=np.float64)
    next_day_irr_new = np.empty(n, dtype=np.float64)

    for i in prange(n):
        irr_sim[i] = 0.0
        irr_continue_new[i] = 0.0
        next_day_irr_new[i] = 0.0

        # Skip if temperature too cold
        if temp_avg[i] < temp_threshold:
            continue

        # Check if new irrigation is needed (depl > RAW on irrigation day)
        needs_irrigation = irr_flag[i] and (depl_root[i] > raw[i])
        has_carryover = irr_continue[i] > 0.0

        # Calculate target refill amount
        target_amount = depl_root[i] * refill_factor

        # First, handle carryover from previous day
        # (Legacy: reduce next_day_irr if it exceeds max_irr_rate)
        irr_waiting = next_day_irr[i]
        if irr_waiting > max_irr_rate[i]:
            next_day_irr_new[i] = irr_waiting - max_irr_rate[i]
        else:
            next_day_irr_new[i] = 0.0

        # Then, check if new irrigation creates carryover
        # (Legacy: irr_day AND depl > raw AND target > max_irr_rate)
        if needs_irrigation and target_amount > max_irr_rate[i]:
            next_day_irr_new[i] = target_amount - max_irr_rate[i]

        # Calculate irrigation amount
        if has_carryover:
            # Apply carryover irrigation (regardless of current depletion)
            potential_irr = irr_waiting
            if potential_irr > max_irr_rate[i]:
                potential_irr = max_irr_rate[i]
            irr_sim[i] = potential_irr
        elif needs_irrigation:
            # Apply new irrigation
            potential_irr = target_amount
            if potential_irr > max_irr_rate[i]:
                potential_irr = max_irr_rate[i]
            irr_sim[i] = potential_irr

        # Set continuation flag for next day
        # Legacy behavior: irr_flag AND (max_irr_rate < depl_root * refill_factor)
        # This is independent of whether depl > RAW!
        if irr_flag[i] and (max_irr_rate[i] < target_amount):
            irr_continue_new[i] = 1.0

    return irr_sim, irr_continue_new, next_day_irr_new


@njit(cache=True, fastmath=True, parallel=True)
def groundwater_subsidy(
    depl_root: NDArray[np.float64],
    raw: NDArray[np.float64],
    gw_status: NDArray[np.bool_],
    f_sub: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate groundwater subsidy to root zone.

    For fields with shallow water tables, groundwater can supply water
    to the root zone when depletion exceeds RAW.

    Physical constraints:
        - gw_sim >= 0
        - gw_sim only applied when depl_root > RAW
        - gw_sim fills back to RAW level (not full capacity)

    Parameters
    ----------
    depl_root : (n_fields,)
        Root zone depletion (mm)
    raw : (n_fields,)
        Readily available water threshold (mm)
    gw_status : (n_fields,)
        Boolean flag for groundwater subsidy availability
    f_sub : (n_fields,)
        Fractional subsidy factor [0, 1]
        Uses threshold logic: if f_sub > 0.2, full subsidy applied
        (matches legacy model compute_field_et.py behavior)

    Returns
    -------
    gw_sim : (n_fields,)
        Groundwater subsidy amount (mm)

    Notes
    -----
    Groundwater subsidy represents capillary rise from a shallow water
    table. It's typically applied in areas with high water tables where
    crops can access groundwater directly.

    The subsidy fills the root zone back to the RAW level, not to field
    capacity, as the water table rise is limited by capillary forces.

    Legacy behavior: gwsub_status = 1 if f_sub > 0.2, and when active,
    the full deficit is applied (not fractional).
    """
    n = depl_root.shape[0]
    gw_sim = np.empty(n, dtype=np.float64)

    # Threshold for activating groundwater subsidy (matches legacy model)
    FSUB_THRESHOLD = 0.2

    for i in prange(n):
        # Apply full subsidy if gw_status AND f_sub > threshold AND depletion > RAW
        if gw_status[i] and f_sub[i] > FSUB_THRESHOLD and depl_root[i] > raw[i]:
            # Full deficit applied (legacy behavior)
            gw_sim[i] = depl_root[i] - raw[i]
        else:
            gw_sim[i] = 0.0

    return gw_sim
