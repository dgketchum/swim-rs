"""Numba-accelerated daily loop for SWIM-RS.

Provides a JIT-compiled version of run_daily_loop that keeps the entire
simulation loop inside numba, avoiding Python-numba boundary crossing overhead.

Uses vectorized array operations for O(n_days) scaling independent of field count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:
    from swimrs.process.input import SwimInput
    from swimrs.process.state import CalibrationParameters, FieldProperties

from swimrs.process.cover_modes import resolve_cover_mode, transpiration_cover_factor
from swimrs.process.kcb_modes import KCB_MODE_LINEAR, resolve_kcb_mode
from swimrs.process.loop import (
    DailyOutput,
    _check_finite_state_arrays,
    _enforce_post_redistribution_invariants,
)
from swimrs.process.state import WaterBalanceState

__all__ = ["run_daily_loop_fast"]


@njit(cache=True)
def _run_loop_jit(
    n_days: int,
    n_fields: int,
    # Time series: (n_days, n_fields)
    all_ndvi: np.ndarray,
    all_etr: np.ndarray,
    all_prcp: np.ndarray,
    all_tmin: np.ndarray,
    all_tmax: np.ndarray,
    all_srad: np.ndarray,
    all_irr_flag: np.ndarray,
    all_f_sub: np.ndarray,
    all_prescribed_irr: np.ndarray,
    # Properties: (n_fields,)
    awc: np.ndarray,
    rew: np.ndarray,
    tew: np.ndarray,
    cn2: np.ndarray,
    zr_max: np.ndarray,
    zr_min: np.ndarray,
    mad: np.ndarray,
    refill_frac: np.ndarray,
    min_irr_days: np.ndarray,
    irr_depth: np.ndarray,
    irr_status: np.ndarray,
    perennial: np.ndarray,
    gw_status: np.ndarray,
    ke_max: np.ndarray,
    # Parameters: (n_fields,)
    kc_max: np.ndarray,
    kc_min: np.ndarray,
    ndvi_k: np.ndarray,
    ndvi_0: np.ndarray,
    ndvi_alpha: np.ndarray,
    ndvi_beta: np.ndarray,
    swe_alpha: np.ndarray,
    swe_beta: np.ndarray,
    kr_damp: np.ndarray,
    ks_damp: np.ndarray,
    max_irr_rate: np.ndarray,
    # Scalar config: FAO-56 stress depletion fraction p (WP-C7 mad split).
    # < 0 is the sentinel for "reuse mad" -> raw_stress == raw == mad*taw
    # (bit-for-bit legacy). >= 0 fixes p so the Ks stress onset uses p*taw while
    # the irrigation trigger and gw subsidy keep the calibrated mad*taw.
    stress_depl_frac: float,
    # Initial state: (n_fields,)
    depl_root_init: np.ndarray,
    depl_ze_init: np.ndarray,
    swe_init: np.ndarray,
    albedo_init: np.ndarray,
    kr_init: np.ndarray,
    ks_init: np.ndarray,
    zr_init: np.ndarray,
    s_init: np.ndarray,
    s1_init: np.ndarray,
    s2_init: np.ndarray,
    s3_init: np.ndarray,
    s4_init: np.ndarray,
    daw3_init: np.ndarray,
    taw3_init: np.ndarray,
    # Transpiration cover weight: integer mode code (swimrs.process.cover_modes) plus
    # the optional explicit linear-ramp endpoints (lo < 0 -> sigmoid-matched).
    cover_mode: int,
    cover_lin_lo: float,
    cover_lin_hi: float,
    # NDVI->Kcb curve: integer mode code (swimrs.process.kcb_modes). 0 = sigmoid
    # on (ndvi_k, ndvi_0); 1 = linear on (ndvi_alpha, ndvi_beta).
    kcb_mode: int,
):
    """JIT-compiled daily loop using vectorized array operations.

    Scales as O(n_days) with near-constant time for field count.
    """
    # Output arrays
    out_eta = np.zeros((n_days, n_fields), dtype=np.float64)
    out_evap = np.zeros((n_days, n_fields), dtype=np.float64)
    out_etf = np.zeros((n_days, n_fields), dtype=np.float64)
    out_kcb = np.zeros((n_days, n_fields), dtype=np.float64)
    out_ke = np.zeros((n_days, n_fields), dtype=np.float64)
    out_ks = np.zeros((n_days, n_fields), dtype=np.float64)
    out_kr = np.zeros((n_days, n_fields), dtype=np.float64)
    out_runoff = np.zeros((n_days, n_fields), dtype=np.float64)
    out_rain = np.zeros((n_days, n_fields), dtype=np.float64)
    out_melt = np.zeros((n_days, n_fields), dtype=np.float64)
    out_swe = np.zeros((n_days, n_fields), dtype=np.float64)
    out_depl_root = np.zeros((n_days, n_fields), dtype=np.float64)
    out_dperc = np.zeros((n_days, n_fields), dtype=np.float64)
    out_irr_sim = np.zeros((n_days, n_fields), dtype=np.float64)
    out_gw_sim = np.zeros((n_days, n_fields), dtype=np.float64)
    out_daw3 = np.zeros((n_days, n_fields), dtype=np.float64)
    out_zr = np.zeros((n_days, n_fields), dtype=np.float64)
    out_depl_ze = np.zeros((n_days, n_fields), dtype=np.float64)

    # State arrays (copy to avoid modifying inputs)
    depl_root = depl_root_init.copy()
    depl_ze = depl_ze_init.copy()
    swe = swe_init.copy()
    albedo = albedo_init.copy()
    kr = kr_init.copy()
    ks = ks_init.copy()
    zr = zr_init.copy()
    s = s_init.copy()
    s1 = s1_init.copy()
    s2 = s2_init.copy()
    s3 = s3_init.copy()
    s4 = s4_init.copy()
    daw3 = daw3_init.copy()
    taw3 = taw3_init.copy()
    irr_continue = np.zeros(n_fields, dtype=np.float64)
    next_day_irr = np.zeros(n_fields, dtype=np.float64)
    prev_irr_sim = np.zeros(n_fields, dtype=np.float64)
    # WP-C1 minimum-return-interval tracking: days since the last new irrigation
    # event per field. Initialized large so the first event is never blocked.
    days_since_event = np.full(n_fields, 1.0e6, dtype=np.float64)

    # Constants
    albedo_min = 0.45
    albedo_max = 0.98
    fresh_snow_threshold = 3.0
    snow_temp_threshold = 1.0
    melt_base_temp = 1.8
    irr_forced_drainage_frac = 0.1

    for day_idx in range(n_days):
        # Get daily inputs for all fields
        ndvi = all_ndvi[day_idx, :]
        etr = all_etr[day_idx, :]
        prcp = all_prcp[day_idx, :]
        tmin = all_tmin[day_idx, :]
        tmax = all_tmax[day_idx, :]
        srad = all_srad[day_idx, :]
        irr_flag = all_irr_flag[day_idx, :]
        temp_avg = (tmin + tmax) * 0.5

        # ================================================================
        # 1. SNOW PARTITIONING AND MELT (vectorized)
        # ================================================================
        # Partition precip
        rain = np.where(temp_avg >= snow_temp_threshold, prcp, 0.0)
        snow = np.where(temp_avg < snow_temp_threshold, prcp, 0.0)

        # Albedo decay (vectorized with np.where chains)
        decay_fast = np.exp(-0.12)
        decay_slow = np.exp(-0.05)
        # Fresh snow resets albedo
        albedo = np.where(
            snow > fresh_snow_threshold,
            albedo_max,
            np.where(
                snow > 0.0,
                albedo_min + (albedo - albedo_min) * decay_fast,
                albedo_min + (albedo - albedo_min) * decay_slow,
            ),
        )
        albedo = np.maximum(albedo_min, np.minimum(albedo_max, albedo))

        # Degree-day snowmelt
        # NOTE: Legacy model updates SWE with today's snowfall BEFORE computing melt
        # (compute_snow.py: foo.swe += sf, then melt = min(foo.swe, melt_potential))
        swe_before_melt = swe + snow
        rad_melt = (1.0 - albedo) * srad * swe_alpha
        dd_melt = (temp_avg - melt_base_temp) * swe_beta
        melt_potential = rad_melt + dd_melt
        melt_potential = np.maximum(melt_potential, 0.0)
        # Melt only when SWE > 0 and tmax > 0
        can_melt = (swe_before_melt > 0.0) & (tmax > 0.0)
        melt = np.where(can_melt, np.minimum(melt_potential, swe_before_melt), 0.0)

        actual_melt = melt
        swe = np.maximum(0.0, swe_before_melt - melt)

        # Effective precipitation
        precip_eff = rain + actual_melt

        # ================================================================
        # 2. RUNOFF (SCS Curve Number with antecedent moisture)
        # ================================================================
        # Adjust CN for antecedent moisture (matches curve_number_adjust kernel)
        # Clip CN2 to valid range
        cn2_clipped = np.maximum(10.0, np.minimum(100.0, cn2))

        # Calculate CNI (dry) and CNIII (wet) from CNII
        cn1 = cn2_clipped / (2.281 - 0.01281 * cn2_clipped)
        cn3 = cn2_clipped / (0.427 + 0.00573 * cn2_clipped)

        # Antecedent moisture thresholds
        awc3 = 0.5 * rew  # Wet threshold
        awc1 = 0.7 * rew + 0.3 * tew  # Dry threshold
        awc1 = np.where(awc1 <= awc3, awc3 + 0.01, awc1)

        # Interpolate CN based on surface depletion
        frac = np.where(
            awc1 > awc3, np.maximum(0.0, np.minimum(1.0, (depl_ze - awc3) / (awc1 - awc3))), 0.0
        )
        cn_adj = np.where(
            depl_ze < awc3,
            cn3,  # Wet condition
            np.where(depl_ze > awc1, cn1, cn3 + frac * (cn1 - cn3)),  # Dry or interpolated
        )

        # Calculate S value (S = 250 * (100/CN - 1) in mm)
        s_new = np.where((cn_adj > 0.0) & (cn_adj < 100.0), 250.0 * (100.0 / cn_adj - 1.0), 0.0)

        # Standard SCS runoff
        ia_std = 0.2 * s_new
        runoff_std = np.where(
            (precip_eff > ia_std) & (s_new > 0.0),
            (precip_eff - ia_std) ** 2 / (precip_eff - ia_std + s_new),
            0.0,
        )

        # Smoothed runoff for irrigated fields (average of runoffs from 4 historical S values)
        # Matches scs_runoff_smoothed kernel: compute runoff for each S, then average
        runoff_smooth = np.zeros(n_fields, dtype=np.float64)
        for s_hist in [s1, s2, s3, s4]:
            ia_hist = 0.2 * s_hist
            sro_hist = np.where(
                (precip_eff > ia_hist) & (s_hist > 0.0),
                (precip_eff - ia_hist) ** 2 / (precip_eff + 0.8 * s_hist),
                0.0,
            )
            runoff_smooth = runoff_smooth + sro_hist
        runoff_smooth = runoff_smooth / 4.0
        runoff_smooth = np.minimum(runoff_smooth, precip_eff)

        # Use smoothed for irrigated, standard for others
        runoff = np.where(irr_status > 0.5, runoff_smooth, runoff_std)

        # Update S history (newest to oldest: s_new -> s1 -> s2 -> s3 -> s4)
        # Legacy model sets s1 to today's S (runoff.py: foo.s1 = foo.s)
        s4 = s3.copy()
        s3 = s2.copy()
        s2 = s1.copy()
        s1 = s_new.copy()
        s = s_new

        infiltration = precip_eff - runoff

        # ================================================================
        # 3. CROP COEFFICIENT (Kcb from NDVI)
        # sigmoid: Kcb = Kc_max / (1 + exp(-k * (NDVI - NDVI_0)))
        # linear:  Kcb = ndvi_beta * NDVI + ndvi_alpha
        # kcb_norm = Kcb / Kc_max feeds the cover weight; under the sigmoid it
        # IS the logistic, taken directly rather than by division so the
        # default path stays bit-for-bit identical to prior runs.
        # ================================================================
        if kcb_mode == KCB_MODE_LINEAR:
            kcb = ndvi_beta * ndvi + ndvi_alpha
            kcb = np.maximum(0.0, np.minimum(kc_max, kcb))
            kcb_norm = np.where(kc_max > 1e-6, kcb / kc_max, 0.0)
        else:
            exp_val = -ndvi_k * (ndvi - ndvi_0)
            exp_val = np.maximum(-20.0, np.minimum(20.0, exp_val))
            sigmoid = 1.0 / (1.0 + np.exp(exp_val))
            kcb = kc_max * sigmoid
            kcb = np.maximum(0.0, np.minimum(kc_max, kcb))
            kcb_norm = sigmoid

        # ================================================================
        # 4. FRACTIONAL COVER from Kcb (FAO-56)
        # ================================================================
        kc_range = kc_max - kc_min
        fc = np.where(kc_range > 1e-6, (kcb - kc_min) / kc_range, 0.0)
        fc = np.maximum(0.0, np.minimum(0.99, fc))
        few = 1.0 - fc

        # ================================================================
        # 5. ROOT DEPTH
        # ================================================================
        zr_prev = zr.copy()
        kcb_ratio = np.where(kc_range > 1e-6, (kcb - kc_min) / kc_range, 0.0)
        kcb_ratio = np.maximum(0.0, kcb_ratio)
        zr_new = zr_min + (zr_max - zr_min) * kcb_ratio
        zr_new = np.maximum(zr_min, np.minimum(zr_max, zr_new))
        # Perennials keep max root depth
        zr_new = np.where(perennial > 0.5, zr_max, zr_new)

        # ================================================================
        # 6. TAW and RAW (using previous day's root depth)
        # ================================================================
        taw = awc * zr
        taw = np.maximum(taw, np.maximum(tew, 0.001))
        # raw = mad * taw is the MANAGEMENT depletion: it drives the irrigation
        # trigger (step 13) and the groundwater subsidy (step 14).
        raw = mad * taw
        # WP-C7: the FAO-56 stress-onset threshold p is a distinct (physiological)
        # quantity. When stress_depl_frac >= 0 it is fixed independently of the
        # calibrated management mad; the < 0 sentinel reuses raw so the legacy
        # single-mad behavior is reproduced bit-for-bit.
        if stress_depl_frac < 0.0:
            raw_stress = raw
        else:
            raw_stress = stress_depl_frac * taw

        # ================================================================
        # 7. UPDATE SURFACE LAYER (Ze)
        # ================================================================
        depl_ze = depl_ze - (actual_melt + rain + prev_irr_sim)
        depl_ze = np.maximum(depl_ze, 0.0)

        # ================================================================
        # 8. Kr AND Ks COEFFICIENTS WITH DAMPING
        # ================================================================
        # Kr base
        denom_kr = tew - rew
        kr_base = np.where(
            denom_kr > 1e-6,
            np.maximum(0.0, (tew - depl_ze) / denom_kr),
            np.where(depl_ze < tew, 1.0, 0.0),
        )
        kr_base = np.minimum(1.0, kr_base)

        # Ks base (stress onset governed by raw_stress = p*taw, not the trigger's mad*taw)
        denom_ks = taw - raw_stress
        ks_base = np.where(
            denom_ks > 1e-6,
            np.maximum(0.0, (taw - depl_root) / denom_ks),
            np.where(depl_root < taw, 1.0, 0.0),
        )
        ks_base = np.minimum(1.0, ks_base)

        # Apply damping
        kr = kr + kr_damp * (kr_base - kr)
        ks = ks + ks_damp * (ks_base - ks)

        # ================================================================
        # 9. EVAPORATION COEFFICIENT (Ke)
        # ================================================================
        ke_energy = kr * (kc_max - kcb)
        ke_area = few * kc_max
        ke = np.minimum(ke_energy, ke_area)
        ke = np.minimum(ke, ke_max)
        ke = np.maximum(ke, 0.0)

        # ================================================================
        # 10. ACTUAL ET (FAO-56 dual crop coefficient)
        # Kc_act = fc_t * Ks * Kcb + Ke, capped at Kc_max
        # fc_t is the transpiration cover weight selected by cover_mode; the
        # evaporation fraction few = 1 - fc above is unaffected by the mode.
        # ================================================================
        fc_t = transpiration_cover_factor(
            cover_mode, ndvi, fc, kcb_norm, ndvi_k, ndvi_0, cover_lin_lo, cover_lin_hi
        )
        kc_act = fc_t * ks * kcb + ke
        kc_act = np.minimum(kc_max, kc_act)
        eta = kc_act * etr
        evap = ke * etr

        # Constrain ET to available water (prevents phantom ET)
        # Available water = current storage + infiltration
        available_for_et = (taw - depl_root) + infiltration
        eta_uncapped = eta
        eta = np.minimum(eta, np.maximum(0.0, available_for_et))
        # The cap reduces E and T proportionally: scale evap to the share
        # actually contained in the capped eta, so the Ze update and the TEW
        # adjustment below never account for evaporation that didn't happen
        evap = evap * (eta / np.maximum(eta_uncapped, 1e-12))

        # ================================================================
        # 11. UPDATE Ze WITH EVAPORATION
        # ================================================================
        depl_ze_prev = depl_ze.copy()
        depl_ze = depl_ze + evap
        depl_ze = np.maximum(depl_ze, 0.0)

        # Cap at TEW and adjust evap + eta
        over_tew = depl_ze > tew
        potential_e = np.maximum(depl_ze - depl_ze_prev, 1e-4)
        e_factor = np.where(
            over_tew, np.maximum(0.0, np.minimum(1.0, 1.0 - (depl_ze - tew) / potential_e)), 1.0
        )
        evap_before_cap = evap.copy()
        evap = evap * e_factor
        eta = eta - (evap_before_cap - evap)
        eta = np.maximum(eta, 0.0)
        depl_ze = np.where(over_tew, np.maximum(depl_ze_prev, 0.0) + evap, depl_ze)

        # ETf
        etf = np.where(etr > 0.0, eta / etr, 0.0)

        # ================================================================
        # 12. UPDATE ROOT ZONE DEPLETION
        # ================================================================
        depl_after_et = depl_root + eta - infiltration

        # ================================================================
        # 13. IRRIGATION DEMAND (per-field logic)
        # Matches irrigation_demand kernel exactly
        # ================================================================
        irr_sim = np.zeros(n_fields, dtype=np.float64)
        irr_continue_new = np.zeros(n_fields, dtype=np.float64)
        next_day_irr_new = np.zeros(n_fields, dtype=np.float64)
        # WP-C1: 1.0 where a NEW irrigation event fires this day (resets the
        # return-interval clock). Kept per-field so the vectorized clock update
        # after the loop is deterministic.
        event_fired = np.zeros(n_fields, dtype=np.float64)
        temp_threshold = 5.0

        for i in range(n_fields):
            # Skip if not an irrigated field
            if irr_status[i] < 0.5:
                continue

            # Skip if temperature too cold (check FIRST, matches kernel)
            if temp_avg[i] < temp_threshold:
                continue

            # WP-C1 minimum-return-interval gate: suppress a NEW demand trigger
            # within `min_irr_days` of the previous event. Carryover (an event
            # already in progress) is unaffected. The `min_irr_days[i] > 0.5`
            # guard short-circuits when the constraint is off (default 0), so
            # the value of days_since_event is never read and the legacy path
            # is reproduced bit-for-bit.
            blocked_by_interval = (min_irr_days[i] > 0.5) and (
                days_since_event[i] < min_irr_days[i]
            )

            # Check if new irrigation is needed (depl > RAW on irrigation day)
            needs_irrigation = (
                (irr_flag[i] > 0.5) and (depl_after_et[i] > raw[i]) and (not blocked_by_interval)
            )
            has_carryover = irr_continue[i] > 0.5

            # Calculate target refill amount. `irr_depth > 0` prescribes a fixed
            # discrete application depth per event; otherwise refill to
            # `refill_frac` of the current depletion (default 1.1 = legacy
            # refill-past-FC, < 1.0 leaves the profile below field capacity).
            if irr_depth[i] > 0.0:
                target_amount = irr_depth[i]
            else:
                target_amount = depl_after_et[i] * refill_frac[i]

            # First, handle carryover from previous day
            irr_waiting = next_day_irr[i]
            if irr_waiting > max_irr_rate[i]:
                next_day_irr_new[i] = irr_waiting - max_irr_rate[i]
            else:
                next_day_irr_new[i] = 0.0

            # Then, check if new irrigation creates carryover
            if needs_irrigation and target_amount > max_irr_rate[i]:
                next_day_irr_new[i] = target_amount - max_irr_rate[i]

            # Calculate irrigation amount (carryover takes priority)
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
                event_fired[i] = 1.0

            # Set continuation flag for next day
            # Legacy behavior: irr_flag AND (max_irr_rate < target_amount)
            # This is independent of whether depl > RAW!
            if (irr_flag[i] > 0.5) and (max_irr_rate[i] < target_amount):
                irr_continue_new[i] = 1.0

        # Update state for next iteration
        irr_continue = irr_continue_new
        next_day_irr = next_day_irr_new
        # WP-C1: advance the return-interval clock, resetting on new events.
        days_since_event = np.where(event_fired > 0.5, 0.0, days_since_event + 1.0)

        # ================================================================
        # 13b. PRESCRIBED-IRRIGATION OVERRIDE (diagnostic physics-bypass)
        # Where a finite observed daily irrigation is supplied, it REPLACES the
        # scheduler's computed amount for that field/day, unconditionally (the
        # irr_status / cold / RAW gates do not apply to an exogenous series).
        # NaN sentinel = "no prescription, keep the scheduler". This runs before
        # the water balance applies irr_sim (step 15) and before prev_irr_sim is
        # captured (step 19), so the override flows fully through the balance.
        # This is a DIAGNOSTIC override only, never a production/calibration input.
        # ================================================================
        for i in range(n_fields):
            pv = all_prescribed_irr[day_idx, i]
            if not np.isnan(pv):
                irr_sim[i] = pv

        # ================================================================
        # 14. GROUNDWATER SUBSIDY
        # Matches groundwater_subsidy kernel: f_sub is a threshold check (> 0.2),
        # not a multiplier. Returns full (depl - raw) when conditions are met.
        # ================================================================
        FSUB_THRESHOLD = 0.2
        gw_sim = np.where(
            (gw_status > 0.5) & (all_f_sub[day_idx] > FSUB_THRESHOLD) & (depl_after_et > raw),
            depl_after_et - raw,
            0.0,
        )

        # ================================================================
        # 15. APPLY IRRIGATION AND GW SUBSIDY
        # The net root-zone addition is 90% of gross irrigation. This is
        # algebraically equivalent to mixing the gross amount and then
        # withdrawing 10% as irrigation-forced drainage in step 17.
        # ================================================================
        irr_net_to_root = (1.0 - irr_forced_drainage_frac) * irr_sim
        depl_new = depl_after_et - irr_net_to_root - gw_sim

        # ================================================================
        # 16. DEEP PERCOLATION
        # ================================================================
        dperc = np.where(depl_new < 0.0, -depl_new, 0.0)
        depl_new = np.maximum(depl_new, 0.0)

        # Cap at TAW
        depl_root = np.minimum(depl_new, taw)

        # ================================================================
        # 17. LAYER 3 STORAGE
        # The fixed 10% irrigation-forced drainage plus the 90% net root-zone
        # addition accounts for all gross irrigation.
        # Matches layer3_storage kernel behavior exactly.
        # ================================================================
        irr_forced_drainage = irr_forced_drainage_frac * irr_sim
        gross_dperc = dperc + irr_forced_drainage

        # Match loop.py logic: only use layer3 storage if any field has taw3 > 0
        if np.any(taw3 > 0.0):
            # Layer 3 storage kernel: add inflow, check overflow
            daw3_new = daw3 + gross_dperc
            daw3_new = np.maximum(daw3_new, 0.0)  # Ensure non-negative
            # Overflow when daw3 > taw3
            dperc_out = np.where(daw3_new > taw3, daw3_new - taw3, 0.0)
            daw3 = np.minimum(daw3_new, taw3)
        else:
            # No layer 3 - all dperc passes through
            dperc_out = gross_dperc

        # ================================================================
        # 18. ROOT GROWTH WATER REDISTRIBUTION
        # ================================================================
        # Matches root_water_redistribution kernel from root_growth.py
        delta_zr = zr_new - zr_prev
        growing = delta_zr > 1e-6
        shrinking = delta_zr < -1e-6

        # Calculate new layer 3 depth and capacity
        layer3_new_depth = np.maximum(0.0, zr_max - zr_new)
        taw3_new = awc * layer3_new_depth

        # Growing: capture a bounded fraction of previous layer-3 water.
        layer3_prev_depth = np.maximum(0.0, zr_max - zr_prev)
        captured_depth = np.minimum(np.maximum(delta_zr, 0.0), layer3_prev_depth)
        added_capacity = awc * captured_depth
        safe_prev_depth = np.where(layer3_prev_depth > 1e-6, layer3_prev_depth, 1.0)
        capture_fraction = np.minimum(1.0, captured_depth / safe_prev_depth)
        water_from_l3 = np.where(
            growing & (layer3_prev_depth > 1e-6) & (daw3 > 0.0) & (added_capacity > 0.0),
            daw3 * capture_fraction,
            0.0,
        )
        water_from_l3 = np.minimum(water_from_l3, daw3)
        water_from_l3 = np.minimum(water_from_l3, added_capacity)
        added_depletion = added_capacity - water_from_l3
        depl_root = np.where(growing, depl_root + added_depletion, depl_root)
        daw3 = np.where(growing, np.maximum(0.0, daw3 - water_from_l3), daw3)

        # Shrinking: water moves from root zone to layer 3
        rt_water_prev = np.where(shrinking, awc * zr_prev - depl_root, 0.0)
        rt_water_prev = np.maximum(0.0, rt_water_prev)
        frac_released = np.where(shrinking & (zr_prev > 1e-6), np.abs(delta_zr) / zr_prev, 0.0)
        frac_released = np.minimum(1.0, frac_released)
        water_to_l3 = rt_water_prev * frac_released
        daw3 = np.where(shrinking, daw3 + water_to_l3, daw3)
        # Update depletion for reduced capacity
        new_taw = awc * zr_new
        released_capacity = awc * np.abs(delta_zr)
        depl_root = np.where(
            shrinking, np.maximum(0.0, depl_root - released_capacity + water_to_l3), depl_root
        )
        depl_root = np.where(shrinking, np.minimum(depl_root, new_taw), depl_root)

        # ALWAYS update taw3 to match kernel behavior
        # The kernel always returns taw3_new = awc * (zr_max - zr_new)
        taw3 = taw3_new

        # Enforce bounded storage before persisting daily outputs
        taw_root_new = awc * zr_new
        taw_root_new = np.maximum(taw_root_new, 0.001)
        taw_root_new = np.maximum(taw_root_new, tew)
        depl_root = np.minimum(np.maximum(depl_root, 0.0), taw_root_new)

        # Ensure daw3 doesn't exceed taw3
        taw3 = np.maximum(taw3, 0.0)
        daw3 = np.minimum(np.maximum(daw3, 0.0), taw3)

        zr = zr_new

        # ================================================================
        # 19. STORE IRR_SIM FOR NEXT DAY
        # ================================================================
        prev_irr_sim = irr_sim.copy()

        # ================================================================
        # STORE OUTPUTS
        # ================================================================
        out_eta[day_idx, :] = eta
        out_evap[day_idx, :] = evap
        out_etf[day_idx, :] = etf
        out_kcb[day_idx, :] = kcb
        out_ke[day_idx, :] = ke
        out_ks[day_idx, :] = ks
        out_kr[day_idx, :] = kr
        out_runoff[day_idx, :] = runoff
        out_rain[day_idx, :] = rain
        out_melt[day_idx, :] = actual_melt
        out_swe[day_idx, :] = swe
        out_depl_root[day_idx, :] = depl_root
        out_dperc[day_idx, :] = dperc_out
        out_irr_sim[day_idx, :] = irr_sim
        out_gw_sim[day_idx, :] = gw_sim
        out_daw3[day_idx, :] = daw3
        out_zr[day_idx, :] = zr
        out_depl_ze[day_idx, :] = depl_ze

    return (
        out_eta,
        out_evap,
        out_etf,
        out_kcb,
        out_ke,
        out_ks,
        out_kr,
        out_runoff,
        out_rain,
        out_melt,
        out_swe,
        out_depl_root,
        out_dperc,
        out_irr_sim,
        out_gw_sim,
        out_daw3,
        out_zr,
        out_depl_ze,
        # Final state
        depl_root,
        depl_ze,
        swe,
        albedo,
        kr,
        ks,
        zr,
        daw3,
        taw3,
    )


def run_daily_loop_fast(
    swim_input: SwimInput,
    parameters: CalibrationParameters | None = None,
    properties: FieldProperties | None = None,
    cover_scaling: bool | None = None,
    prescribed_irr: np.ndarray | None = None,
    cover_mode: str | int | None = None,
    kcb_mode: str | int | None = None,
) -> tuple[DailyOutput, WaterBalanceState]:
    """Run daily water balance simulation using JIT-compiled loop.

    This is a high-performance replacement for run_daily_loop() that keeps
    the entire simulation loop inside numba, avoiding the overhead of
    crossing the Python-numba boundary on each day.

    Uses vectorized array operations that scale as O(n_days) with near-constant
    time for field count (up to hardware limits).

    Parameters
    ----------
    swim_input : SwimInput
        Input data container (HDF5-backed)
    parameters : CalibrationParameters, optional
        Calibration parameters. If not provided, uses swim_input.parameters.
    properties : FieldProperties, optional
        Field properties. If not provided, uses swim_input.properties.
        Pass custom properties to use PEST++ calibrated values (awc, mad).
    prescribed_irr : np.ndarray, optional
        Diagnostic exogenous daily irrigation, shape ``(n_days, n_fields)`` in
        mm/day. Where a value is finite it REPLACES the internal scheduler's
        ``irr_sim`` for that field/day; ``NaN`` means "use the scheduler". If not
        given here, it falls back to ``swim_input.get_prescribed_irr()`` (the h5
        ``prescribed_irrigation/irr_mm`` group), and finally to an all-NaN array
        (scheduler everywhere, i.e. baseline). This is a physics-bypass for
        attribution experiments only — never a production or calibration input.
    cover_mode : str | int, optional
        Transpiration cover-weight formulation (``none``/``kcb``/``sigmoid``/
        ``linear``; see :mod:`swimrs.process.cover_modes`). Defaults to the mode baked
        into ``swim_input`` at build time, itself defaulting to ``kcb`` — the
        historical ``transpiration_cover_scaling=True`` behavior.
    kcb_mode : str | int, optional
        NDVI→Kcb curve (``sigmoid``/``linear``; see
        :mod:`swimrs.process.kcb_modes`). Defaults to the mode baked into
        ``swim_input`` at build time, itself defaulting to ``sigmoid``.
    Returns
    -------
    output : DailyOutput
        Daily output arrays with shape (n_days, n_fields)
    final_state : WaterBalanceState
        Final state after simulation
    """
    if cover_mode is None and cover_scaling is None:
        cover_mode = getattr(swim_input, "cover_mode", None)
        cover_scaling = getattr(swim_input, "cover_scaling", None)
    cover_mode_code = resolve_cover_mode(cover_mode, cover_scaling)

    lin_lo, lin_hi = getattr(swim_input, "cover_linear_ndvi_range", (None, None))
    cover_lin_lo = -1.0 if lin_lo is None else float(lin_lo)
    cover_lin_hi = -1.0 if lin_hi is None else float(lin_hi)

    if kcb_mode is None:
        kcb_mode = getattr(swim_input, "kcb_mode", None)
    kcb_mode_code = resolve_kcb_mode(kcb_mode)

    # WP-C7 mad split: the FAO-56 stress depletion fraction p. None (the default
    # for every existing project) -> -1.0 sentinel -> the loop reuses mad*taw for
    # the Ks stress onset, reproducing the legacy single-mad behavior bit-for-bit.
    sdf = getattr(swim_input, "stress_depletion_fraction", None)
    stress_depl_frac = -1.0 if sdf is None else float(sdf)

    n_days = swim_input.n_days
    n_fields = swim_input.n_fields
    props = properties if properties is not None else swim_input.properties
    params = parameters if parameters is not None else swim_input.parameters
    spinup = swim_input.spinup_state

    # Pre-load all time series (this is fast - single HDF5 read per variable)
    all_ndvi = swim_input.get_time_series("ndvi").astype(np.float64)
    all_ref_et = swim_input.get_time_series("ref_et").astype(np.float64)
    all_prcp = swim_input.get_time_series("prcp").astype(np.float64)
    all_tmin = swim_input.get_time_series("tmin").astype(np.float64)
    all_tmax = swim_input.get_time_series("tmax").astype(np.float64)
    all_srad = swim_input.get_time_series("srad").astype(np.float64)
    all_irr_flag = swim_input.get_irr_flag().astype(np.float64)

    # Prescribed-irrigation override: explicit argument wins, then the h5 group,
    # then an all-NaN array (baseline: scheduler everywhere). NaN is the sentinel
    # for "no prescription this field/day", so numba stays nopython.
    if prescribed_irr is None:
        prescribed_irr = swim_input.get_prescribed_irr()
    if prescribed_irr is None:
        all_prescribed_irr = np.full((n_days, n_fields), np.nan, dtype=np.float64)
    else:
        all_prescribed_irr = np.asarray(prescribed_irr, dtype=np.float64)
        if all_prescribed_irr.shape != (n_days, n_fields):
            raise ValueError(
                "prescribed_irr must have shape (n_days, n_fields) = "
                f"{(n_days, n_fields)}, got {all_prescribed_irr.shape}"
            )

    # Extract property arrays
    awc = props.awc.astype(np.float64)
    rew = props.rew.astype(np.float64)
    tew = props.tew.astype(np.float64)
    cn2 = props.cn2.astype(np.float64)
    zr_max = props.zr_max.astype(np.float64)
    zr_min = props.zr_min.astype(np.float64)
    mad = props.mad.astype(np.float64)
    # WP-C1 scheduler-realism knobs. getattr fallbacks keep older FieldProperties
    # (or hand-built ones) working with the legacy defaults (refill 1.1, no min
    # interval, no fixed depth).
    refill_frac_attr = getattr(props, "refill_frac", None)
    refill_frac = (
        refill_frac_attr.astype(np.float64)
        if refill_frac_attr is not None
        else np.full(n_fields, 1.1)
    )
    min_irr_days_attr = getattr(props, "min_irr_days", None)
    min_irr_days = (
        min_irr_days_attr.astype(np.float64)
        if min_irr_days_attr is not None
        else np.zeros(n_fields)
    )
    irr_depth_attr = getattr(props, "irr_depth", None)
    irr_depth = (
        irr_depth_attr.astype(np.float64) if irr_depth_attr is not None else np.zeros(n_fields)
    )
    irr_status = props.irr_status.astype(np.float64)
    perennial = props.perennial.astype(np.float64)
    gw_status = props.gw_status.astype(np.float64)
    ke_max = props.ke_max.astype(np.float64) if props.ke_max is not None else np.ones(n_fields)
    kc_max = (
        props.kc_max.astype(np.float64) if props.kc_max is not None else np.full(n_fields, 1.25)
    )
    # Daily f_sub: year-specific values when available (matches run_daily_loop),
    # else the static property broadcast across all days
    f_sub_static = props.f_sub.astype(np.float64) if props.f_sub is not None else np.zeros(n_fields)
    all_f_sub = np.empty((n_days, n_fields), dtype=np.float64)
    if swim_input.has_year_specific_gwsub():
        current_year = None
        for day_idx in range(n_days):
            year = swim_input.get_date(day_idx).year
            if year != current_year:
                current_year = year
                f_sub_year = swim_input.get_f_sub_for_year(year).astype(np.float64)
            all_f_sub[day_idx, :] = f_sub_year
    else:
        all_f_sub[:] = f_sub_static

    # Extract parameter arrays
    kc_min = params.kc_min.astype(np.float64)
    ndvi_k = params.ndvi_k.astype(np.float64)
    ndvi_0 = params.ndvi_0.astype(np.float64)
    ndvi_alpha = params.ndvi_alpha.astype(np.float64)
    ndvi_beta = params.ndvi_beta.astype(np.float64)
    swe_alpha = params.swe_alpha.astype(np.float64)
    swe_beta = params.swe_beta.astype(np.float64)
    kr_damp = params.kr_damp.astype(np.float64)
    ks_damp = params.ks_damp.astype(np.float64)
    max_irr_rate = params.max_irr_rate.astype(np.float64)

    # Extract initial state arrays
    depl_root_init = spinup.depl_root.astype(np.float64)
    depl_ze_init = (
        spinup.depl_ze.astype(np.float64) if spinup.depl_ze is not None else np.zeros(n_fields)
    )
    swe_init = spinup.swe.astype(np.float64)
    albedo_init = (
        spinup.albedo.astype(np.float64) if spinup.albedo is not None else np.full(n_fields, 0.45)
    )
    kr_init = spinup.kr.astype(np.float64)
    ks_init = spinup.ks.astype(np.float64)
    zr_init = spinup.zr.astype(np.float64)

    # S history for smoothed CN runoff
    default_s = 84.7  # Default S from CN2=75
    s_init = spinup.s.astype(np.float64) if spinup.s is not None else np.full(n_fields, default_s)
    s1_init = (
        spinup.s1.astype(np.float64) if spinup.s1 is not None else np.full(n_fields, default_s)
    )
    s2_init = (
        spinup.s2.astype(np.float64) if spinup.s2 is not None else np.full(n_fields, default_s)
    )
    s3_init = (
        spinup.s3.astype(np.float64) if spinup.s3 is not None else np.full(n_fields, default_s)
    )
    s4_init = (
        spinup.s4.astype(np.float64) if spinup.s4 is not None else np.full(n_fields, default_s)
    )

    # Layer 3 storage
    daw3_init = spinup.daw3.astype(np.float64) if spinup.daw3 is not None else np.zeros(n_fields)
    taw3_init = spinup.taw3.astype(np.float64) if spinup.taw3 is not None else np.zeros(n_fields)

    # Run the JIT-compiled loop
    (
        out_eta,
        out_evap,
        out_etf,
        out_kcb,
        out_ke,
        out_ks,
        out_kr,
        out_runoff,
        out_rain,
        out_melt,
        out_swe,
        out_depl_root,
        out_dperc,
        out_irr_sim,
        out_gw_sim,
        out_daw3,
        out_zr,
        out_depl_ze,
        final_depl_root,
        final_depl_ze,
        final_swe,
        final_albedo,
        final_kr,
        final_ks,
        final_zr,
        final_daw3,
        final_taw3,
    ) = _run_loop_jit(
        n_days,
        n_fields,
        all_ndvi,
        all_ref_et,
        all_prcp,
        all_tmin,
        all_tmax,
        all_srad,
        all_irr_flag,
        all_f_sub,
        all_prescribed_irr,
        awc,
        rew,
        tew,
        cn2,
        zr_max,
        zr_min,
        mad,
        refill_frac,
        min_irr_days,
        irr_depth,
        irr_status,
        perennial,
        gw_status,
        ke_max,
        kc_max,
        kc_min,
        ndvi_k,
        ndvi_0,
        ndvi_alpha,
        ndvi_beta,
        swe_alpha,
        swe_beta,
        kr_damp,
        ks_damp,
        max_irr_rate,
        stress_depl_frac,
        depl_root_init,
        depl_ze_init,
        swe_init,
        albedo_init,
        kr_init,
        ks_init,
        zr_init,
        s_init,
        s1_init,
        s2_init,
        s3_init,
        s4_init,
        daw3_init,
        taw3_init,
        cover_mode_code,
        cover_lin_lo,
        cover_lin_hi,
        kcb_mode_code,
    )

    # Package outputs into DailyOutput dataclass
    output = DailyOutput(n_days=n_days, n_fields=n_fields)
    output.eta = out_eta
    output.evap = out_evap
    output.etf = out_etf
    output.kcb = out_kcb
    output.ke = out_ke
    output.ks = out_ks
    output.kr = out_kr
    output.runoff = out_runoff
    output.rain = out_rain
    output.melt = out_melt
    output.swe = out_swe
    output.depl_root = out_depl_root
    output.dperc = out_dperc
    output.irr_sim = out_irr_sim
    output.gw_sim = out_gw_sim
    output.daw3 = out_daw3
    output.zr = out_zr
    output.depl_ze = out_depl_ze
    _check_finite_state_arrays(
        "run_daily_loop_fast/output",
        {
            "eta": output.eta,
            "etf": output.etf,
            "runoff": output.runoff,
            "swe": output.swe,
            "depl_root": output.depl_root,
            "dperc": output.dperc,
        },
    )

    # Package final state
    taw_root_final = props.compute_taw(final_zr)
    final_depl_root, final_daw3, final_taw3 = _enforce_post_redistribution_invariants(
        context="run_daily_loop_fast/final_state",
        depl_root=final_depl_root,
        taw_root=taw_root_final,
        daw3=final_daw3,
        taw3=final_taw3,
        extra_arrays={
            "zr": final_zr,
            "kr": final_kr,
            "ks": final_ks,
            "depl_ze": final_depl_ze,
        },
    )
    final_state = WaterBalanceState.from_spinup(
        n_fields=n_fields,
        depl_root=final_depl_root,
        swe=final_swe,
        kr=final_kr,
        ks=final_ks,
        zr=final_zr,
        depl_ze=final_depl_ze,
        albedo=final_albedo,
        daw3=final_daw3,
        taw3=final_taw3,
    )

    return output, final_state
