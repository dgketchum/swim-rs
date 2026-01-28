"""Irrigation fraction tracking for consumptive use accounting.

Tracks the fraction of soil water that originated from irrigation to enable
legally defensible accounting of consumptive use for water rights purposes.

Conceptual Model
----------------
1. Well-mixed reservoir: Water in each pool is instantaneously mixed
2. Proportional withdrawal: ET draws proportionally from all sources
3. Source tracking: Each input has a known irrigation fraction:
   - Precipitation (rain, melt): frac = 0 (natural)
   - Irrigation (irr_sim): frac = 1 (irrigation)
   - Groundwater subsidy (gw_sim): frac = 0 (natural)

Key Outputs
-----------
- et_irr: eta * irr_frac_root (consumptive use of irrigation water)
- dperc_irr: dperc_out * irr_frac_l3 (irrigation water leaving as deep perc)

Conservation Law
----------------
For any period: sum(irr_sim) = sum(et_irr) + sum(dperc_irr) + delta_storage_irr
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

__all__ = [
    "update_irrigation_fraction_root",
    "update_irrigation_fraction_l3",
    "transfer_fraction_with_water",
]


@njit(cache=True, fastmath=True, parallel=True)
def update_irrigation_fraction_root(
    awc: NDArray[np.float64],
    zr: NDArray[np.float64],
    depl_root: NDArray[np.float64],
    irr_frac_root: NDArray[np.float64],
    infiltration: NDArray[np.float64],
    irr_sim: NDArray[np.float64],
    gw_sim: NDArray[np.float64],
    eta: NDArray[np.float64],
    dperc: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Update irrigation fraction in root zone after daily fluxes.

    The mixing rule: when adding inflow with frac_in to a pool with frac_pool:
        frac_new = (frac_pool * water_pool + frac_in * inflow) / water_new

    Outflows (ET, dperc) carry the pool's current fraction (proportional withdrawal).

    Parameters
    ----------
    awc : (n_fields,)
        Available water capacity (mm/m)
    zr : (n_fields,)
        Root depth (m)
    depl_root : (n_fields,)
        Root zone depletion BEFORE today's fluxes (mm)
    irr_frac_root : (n_fields,)
        Irrigation fraction in root zone BEFORE today's fluxes [0, 1]
    infiltration : (n_fields,)
        Infiltrating precipitation (rain + melt - runoff) (mm), frac = 0
    irr_sim : (n_fields,)
        Simulated irrigation (mm), frac = 1
    gw_sim : (n_fields,)
        Groundwater subsidy (mm), frac = 0; negative values are withdrawals
    eta : (n_fields,)
        Actual ET (mm) - withdrawal
    dperc : (n_fields,)
        Deep percolation from root zone (mm) - withdrawal

    Returns
    -------
    irr_frac_root_new : (n_fields,)
        Updated irrigation fraction in root zone [0, 1]
    et_irr : (n_fields,)
        ET from irrigation water (mm)

    Notes
    -----
    Outflow is calculated BEFORE inflow mixing, so ET carries the pre-mix
    irrigation fraction. This is conservative (doesn't attribute today's
    irrigation to today's ET).
    """
    n = awc.shape[0]
    irr_frac_root_new = np.empty(n, dtype=np.float64)
    et_irr = np.empty(n, dtype=np.float64)

    for i in prange(n):
        # Water content before today's fluxes
        water_before = awc[i] * zr[i] - depl_root[i]
        if water_before < 0.0:
            water_before = 0.0

        frac_before = irr_frac_root[i]

        # ET carries current fraction (before mixing)
        et_irr[i] = eta[i] * frac_before

        # Total outflow (all carry current fraction)
        # gw_sim can be negative (withdrawal from root zone)
        gw_withdrawal = 0.0
        if gw_sim[i] < 0.0:
            gw_withdrawal = -gw_sim[i]

        outflow = eta[i] + dperc[i] + gw_withdrawal

        # Water after outflow
        water_mid = water_before - outflow
        if water_mid < 0.0:
            water_mid = 0.0

        # Inflows and their fractions
        # infiltration: frac = 0, irr_sim: frac = 1, gw_sim (positive): frac = 0
        gw_inflow = 0.0
        if gw_sim[i] > 0.0:
            gw_inflow = gw_sim[i]

        inflow_natural = infiltration[i] + gw_inflow  # frac = 0
        inflow_irr = irr_sim[i]  # frac = 1
        total_inflow = inflow_natural + inflow_irr

        # Water after inflow
        water_after = water_mid + total_inflow

        # Mix to get new fraction
        if water_after < 1e-6:
            # Pool essentially empty
            irr_frac_root_new[i] = 0.0
        else:
            # Irrigation water in pool after mixing
            irr_water_mid = frac_before * water_mid
            irr_water_after = irr_water_mid + inflow_irr  # inflow_natural has frac=0
            irr_frac_root_new[i] = irr_water_after / water_after

        # Clamp to valid range
        if irr_frac_root_new[i] < 0.0:
            irr_frac_root_new[i] = 0.0
        elif irr_frac_root_new[i] > 1.0:
            irr_frac_root_new[i] = 1.0

    return irr_frac_root_new, et_irr


@njit(cache=True, fastmath=True, parallel=True)
def update_irrigation_fraction_l3(
    daw3: NDArray[np.float64],
    irr_frac_l3: NDArray[np.float64],
    gross_dperc: NDArray[np.float64],
    irr_frac_inflow: NDArray[np.float64],
    dperc_out: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Update irrigation fraction in layer 3 after deep percolation.

    Parameters
    ----------
    daw3 : (n_fields,)
        Layer 3 available water BEFORE today's fluxes (mm)
    irr_frac_l3 : (n_fields,)
        Irrigation fraction in layer 3 BEFORE today's fluxes [0, 1]
    gross_dperc : (n_fields,)
        Gross deep percolation entering layer 3 (mm)
        Includes dperc + IRR_BYPASS_FRAC of applied irrigation (bypasses root zone)
    irr_frac_inflow : (n_fields,)
        Irrigation fraction of inflowing water (root zone's current fraction)
    dperc_out : (n_fields,)
        Deep percolation leaving layer 3 (overflow) (mm)

    Returns
    -------
    irr_frac_l3_new : (n_fields,)
        Updated irrigation fraction in layer 3 [0, 1]
    dperc_irr : (n_fields,)
        Deep percolation of irrigation water leaving system (mm)

    Notes
    -----
    Layer 3 outflow (dperc_out) carries the MIXED fraction, computed after
    adding inflow. This is because overflow happens when the layer fills up,
    which conceptually occurs after mixing.
    """
    n = daw3.shape[0]
    irr_frac_l3_new = np.empty(n, dtype=np.float64)
    dperc_irr = np.empty(n, dtype=np.float64)

    for i in prange(n):
        water_before = daw3[i]
        if water_before < 0.0:
            water_before = 0.0

        frac_before = irr_frac_l3[i]
        inflow = gross_dperc[i]

        # Water after adding inflow (before overflow)
        water_after_inflow = water_before + inflow

        # Calculate mixed fraction after inflow
        if water_after_inflow < 1e-6:
            frac_mixed = 0.0
        else:
            irr_water_before = frac_before * water_before
            irr_water_inflow = irr_frac_inflow[i] * inflow
            frac_mixed = (irr_water_before + irr_water_inflow) / water_after_inflow

        # Overflow (dperc_out) carries the mixed fraction
        dperc_irr[i] = dperc_out[i] * frac_mixed

        # Final water after overflow
        water_final = water_after_inflow - dperc_out[i]
        if water_final < 1e-6:
            irr_frac_l3_new[i] = 0.0
        else:
            # Fraction unchanged by proportional withdrawal
            irr_frac_l3_new[i] = frac_mixed

        # Clamp to valid range
        if irr_frac_l3_new[i] < 0.0:
            irr_frac_l3_new[i] = 0.0
        elif irr_frac_l3_new[i] > 1.0:
            irr_frac_l3_new[i] = 1.0

    return irr_frac_l3_new, dperc_irr


@njit(cache=True, fastmath=True, parallel=True)
def transfer_fraction_with_water(
    water_from: NDArray[np.float64],
    frac_from: NDArray[np.float64],
    water_to: NDArray[np.float64],
    frac_to: NDArray[np.float64],
    transfer: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Transfer irrigation fraction when water moves between pools (root growth).

    Used when roots grow (absorb L3 water) or recede (release water to L3).

    Parameters
    ----------
    water_from : (n_fields,)
        Water in source pool before transfer (mm)
    frac_from : (n_fields,)
        Irrigation fraction in source pool [0, 1]
    water_to : (n_fields,)
        Water in destination pool before transfer (mm)
    frac_to : (n_fields,)
        Irrigation fraction in destination pool [0, 1]
    transfer : (n_fields,)
        Water transferred from source to destination (mm), >= 0

    Returns
    -------
    frac_from_new : (n_fields,)
        Updated irrigation fraction in source pool [0, 1]
    frac_to_new : (n_fields,)
        Updated irrigation fraction in destination pool [0, 1]

    Notes
    -----
    Source pool fraction unchanged by proportional withdrawal.
    Destination pool fraction updated by mixing rule.
    """
    n = water_from.shape[0]
    frac_from_new = np.empty(n, dtype=np.float64)
    frac_to_new = np.empty(n, dtype=np.float64)

    for i in prange(n):
        # Source: proportional withdrawal, fraction unchanged
        water_from_after = water_from[i] - transfer[i]
        if water_from_after < 1e-6:
            frac_from_new[i] = 0.0
        else:
            frac_from_new[i] = frac_from[i]

        # Destination: mix with incoming water
        water_to_after = water_to[i] + transfer[i]
        if water_to_after < 1e-6:
            frac_to_new[i] = 0.0
        else:
            irr_water_to = frac_to[i] * water_to[i]
            irr_water_transfer = frac_from[i] * transfer[i]
            frac_to_new[i] = (irr_water_to + irr_water_transfer) / water_to_after

        # Clamp to valid range
        if frac_from_new[i] < 0.0:
            frac_from_new[i] = 0.0
        elif frac_from_new[i] > 1.0:
            frac_from_new[i] = 1.0

        if frac_to_new[i] < 0.0:
            frac_to_new[i] = 0.0
        elif frac_to_new[i] > 1.0:
            frac_to_new[i] = 1.0

    return frac_from_new, frac_to_new
