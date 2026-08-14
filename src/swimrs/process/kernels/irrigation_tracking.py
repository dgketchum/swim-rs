"""Irrigation-source tracking for consumptive-use accounting.

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

The root-zone tracer carries the hydrologic depletion within a conservative
mixing capacity. A fixed natural buffer represents the shallow-root TAW floor,
so root-depth changes do not create or destroy tracer volume.

Daily Ordering
--------------
1. Natural infiltration mixes into the root zone.
2. ET withdraws water proportionally from that mixture.
3. Gross irrigation mixes into the root zone; groundwater subsidy is natural.
4. Irrigation-forced and saturation drainage withdraw proportionally.
5. Root drainage mixes into layer 3 before bottom-boundary overflow.

The fixed irrigation-forced drainage flux is therefore not assumed to consist
of newly applied irrigation. It can displace precipitation-derived water that
was already stored in the soil column.

Conservation Law
----------------
For any period: sum(irr_sim) = sum(et_irr) + sum(dperc_irr) + delta_storage_irr
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

__all__ = [
    "redistribute_irrigation_fractions",
    "update_irrigation_fraction_root",
    "update_irrigation_fraction_l3",
    "transfer_fraction_with_water",
]

SOURCE_WATER_EPS = 1e-12


@njit(cache=True, fastmath=True, parallel=True)
def update_irrigation_fraction_root(
    root_water: NDArray[np.float64],
    irr_frac_root: NDArray[np.float64],
    infiltration: NDArray[np.float64],
    irr_sim: NDArray[np.float64],
    gw_sim: NDArray[np.float64],
    eta: NDArray[np.float64],
    root_drainage: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Update irrigation fraction in root zone after daily fluxes.

    The mixing rule: when adding inflow with frac_in to a pool with frac_pool:
        frac_new = (frac_pool * water_pool + frac_in * inflow) / water_new

    Fractions are converted to irrigation-water mass before applying fluxes.
    Infiltration mixes before ET, while gross irrigation and natural groundwater
    subsidy mix after ET. Root drainage then carries the resulting mixed
    fraction. This ordering matches the daily water-balance semantics and
    conserves irrigation-source mass exactly.

    Parameters
    ----------
    root_water : (n_fields,)
        Water in the conservative root-zone source-tracking reservoir before
        today's fluxes (mm)
    irr_frac_root : (n_fields,)
        Irrigation fraction in root zone BEFORE today's fluxes [0, 1]
    infiltration : (n_fields,)
        Infiltrating precipitation (rain + melt - runoff) (mm), frac = 0
    irr_sim : (n_fields,)
        Gross simulated irrigation (mm), frac = 1
    gw_sim : (n_fields,)
        Groundwater subsidy (mm), frac = 0
    eta : (n_fields,)
        Actual ET (mm) - withdrawal
    root_drainage : (n_fields,)
        Total drainage transferred from the root zone to layer 3 (mm),
        including irrigation-forced and saturation drainage

    Returns
    -------
    irr_frac_root_new : (n_fields,)
        Updated irrigation fraction in root zone [0, 1]
    et_irr : (n_fields,)
        ET from irrigation water (mm)
    root_drainage_irr : (n_fields,)
        Irrigation-source drainage transferred from the root zone (mm)

    Notes
    -----
    Today's natural infiltration can supply today's ET. Gross irrigation is
    mixed only after ET, so irrigation scheduled today cannot supply today's
    ET. All root drainage is withdrawn after irrigation mixing.
    """
    n = root_water.shape[0]
    irr_frac_root_new = np.empty(n, dtype=np.float64)
    et_irr = np.empty(n, dtype=np.float64)
    root_drainage_irr = np.empty(n, dtype=np.float64)

    for i in prange(n):
        # Water content before today's fluxes. The hydrologic loop supplies
        # this directly so source accounting uses the same conservative mixing
        # reservoir for daily fluxes and root-depth redistribution.
        water_before = root_water[i]
        if water_before < 0.0:
            water_before = 0.0

        frac_before = irr_frac_root[i]
        if frac_before < 0.0:
            frac_before = 0.0
        elif frac_before > 1.0:
            frac_before = 1.0
        irr_water_before = frac_before * water_before

        # Natural infiltration mixes before ET. The physical loop caps ET by
        # this water availability, so eta cannot materially exceed this pool.
        water_before_et = water_before + infiltration[i]
        if water_before_et < SOURCE_WATER_EPS:
            frac_before_et = 0.0
        else:
            frac_before_et = irr_water_before / water_before_et
        et_irr[i] = eta[i] * frac_before_et
        water_after_et = water_before_et - eta[i]
        irr_water_after_et = irr_water_before - et_irr[i]

        # Gross irrigation is mixed before the fixed irrigation-forced
        # drainage is withdrawn. Groundwater subsidy is a natural source.
        water_before_drainage = water_after_et + irr_sim[i] + gw_sim[i]
        irr_water_before_drainage = irr_water_after_et + irr_sim[i]
        if water_before_drainage < SOURCE_WATER_EPS:
            frac_before_drainage = 0.0
        else:
            frac_before_drainage = irr_water_before_drainage / water_before_drainage

        root_drainage_irr[i] = root_drainage[i] * frac_before_drainage
        water_after = water_before_drainage - root_drainage[i]
        irr_water_after = irr_water_before_drainage - root_drainage_irr[i]

        if water_after < SOURCE_WATER_EPS:
            irr_frac_root_new[i] = 0.0
        else:
            irr_frac_root_new[i] = irr_water_after / water_after

        # Clamp to valid range
        if irr_frac_root_new[i] < 0.0:
            irr_frac_root_new[i] = 0.0
        elif irr_frac_root_new[i] > 1.0:
            irr_frac_root_new[i] = 1.0

    return irr_frac_root_new, et_irr, root_drainage_irr


@njit(cache=True, fastmath=True, parallel=True)
def update_irrigation_fraction_l3(
    daw3: NDArray[np.float64],
    irr_frac_l3: NDArray[np.float64],
    gross_dperc: NDArray[np.float64],
    irrigation_inflow: NDArray[np.float64],
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
        Includes irrigation-forced and saturation drainage from the root zone
    irrigation_inflow : (n_fields,)
        Irrigation-source mass in gross_dperc (mm)
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
        if water_after_inflow < SOURCE_WATER_EPS:
            frac_mixed = 0.0
        else:
            irr_water_before = frac_before * water_before
            irr_water_inflow = irrigation_inflow[i]
            frac_mixed = (irr_water_before + irr_water_inflow) / water_after_inflow

        # Overflow (dperc_out) carries the mixed fraction
        dperc_irr[i] = dperc_out[i] * frac_mixed

        # Final water after overflow
        water_final = water_after_inflow - dperc_out[i]
        if water_final < SOURCE_WATER_EPS:
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
def redistribute_irrigation_fractions(
    root_water_before: NDArray[np.float64],
    irr_frac_root_before: NDArray[np.float64],
    l3_water_before: NDArray[np.float64],
    irr_frac_l3_before: NDArray[np.float64],
    root_water_after: NDArray[np.float64],
    l3_water_after: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Redistribute source fractions after the root-zone boundary moves.

    Actual layer-3 storage change defines the nominal physical transfer between
    pools, and water leaving a donor pool carries its current source fraction.
    The caller's fixed natural shallow-root buffer makes changes in root mixing
    capacity conservative as the boundary moves. As a final numerical guard,
    allocation is projected to the nearest feasible two-pool split if a nominal
    destination cannot represent the transferred irrigation mass. This
    preserves source mass and fraction bounds without changing hydrology.

    Parameters
    ----------
    root_water_before, root_water_after : (n_fields,)
        Conservative root-zone mixing water before and after redistribution (mm)
    irr_frac_root_before : (n_fields,)
        Root-zone irrigation fraction before redistribution [0, 1]
    l3_water_before, l3_water_after : (n_fields,)
        Layer-3 water before and after redistribution (mm)
    irr_frac_l3_before : (n_fields,)
        Layer-3 irrigation fraction before redistribution [0, 1]

    Returns
    -------
    irr_frac_root_after, irr_frac_l3_after : (n_fields,)
        Source fractions after redistribution. Values are intentionally not
        clipped; the caller's invariants should expose an impossible state
        rather than silently destroy source mass.
    """
    n = root_water_before.shape[0]
    irr_frac_root_after = np.empty(n, dtype=np.float64)
    irr_frac_l3_after = np.empty(n, dtype=np.float64)

    for i in prange(n):
        root_before = root_water_before[i]
        if root_before < 0.0:
            root_before = 0.0
        l3_before = l3_water_before[i]
        if l3_before < 0.0:
            l3_before = 0.0
        root_after = root_water_after[i]
        if root_after < 0.0:
            root_after = 0.0
        l3_after = l3_water_after[i]
        if l3_after < 0.0:
            l3_after = 0.0

        root_fraction = irr_frac_root_before[i]
        if root_fraction < 0.0:
            root_fraction = 0.0
        elif root_fraction > 1.0:
            root_fraction = 1.0
        l3_fraction = irr_frac_l3_before[i]
        if l3_fraction < 0.0:
            l3_fraction = 0.0
        elif l3_fraction > 1.0:
            l3_fraction = 1.0

        root_irrigation = root_before * root_fraction
        l3_irrigation = l3_before * l3_fraction
        l3_change = l3_after - l3_before

        if l3_change > SOURCE_WATER_EPS:
            # Roots receded: actual water added to L3 came from the root pool.
            transferred = l3_change
            if transferred > root_before:
                transferred = root_before
            irrigation_transferred = transferred * root_fraction
            root_irrigation -= irrigation_transferred
            l3_irrigation += irrigation_transferred
        elif l3_change < -SOURCE_WATER_EPS:
            # Roots grew: actual water removed from L3 entered the root pool.
            transferred = -l3_change
            if transferred > l3_before:
                transferred = l3_before
            irrigation_transferred = transferred * l3_fraction
            l3_irrigation -= irrigation_transferred
            root_irrigation += irrigation_transferred

        # Project the nominal proportional allocation onto the feasible interval
        # while preserving total irrigation-source mass across the two pools.
        total_irrigation = root_before * root_fraction + l3_before * l3_fraction
        minimum_root_irrigation = total_irrigation - l3_after
        if minimum_root_irrigation < 0.0:
            minimum_root_irrigation = 0.0
        maximum_root_irrigation = total_irrigation
        if maximum_root_irrigation > root_after:
            maximum_root_irrigation = root_after
        if root_irrigation < minimum_root_irrigation:
            root_irrigation = minimum_root_irrigation
        elif root_irrigation > maximum_root_irrigation:
            root_irrigation = maximum_root_irrigation
        l3_irrigation = total_irrigation - root_irrigation

        if root_after < SOURCE_WATER_EPS:
            irr_frac_root_after[i] = 0.0
        else:
            irr_frac_root_after[i] = root_irrigation / root_after

        if l3_after < SOURCE_WATER_EPS:
            irr_frac_l3_after[i] = 0.0
        else:
            irr_frac_l3_after[i] = l3_irrigation / l3_after

    return irr_frac_root_after, irr_frac_l3_after


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
        if water_from_after < SOURCE_WATER_EPS:
            frac_from_new[i] = 0.0
        else:
            frac_from_new[i] = frac_from[i]

        # Destination: mix with incoming water
        water_to_after = water_to[i] + transfer[i]
        if water_to_after < SOURCE_WATER_EPS:
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
