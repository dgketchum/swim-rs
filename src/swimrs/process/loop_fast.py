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

from swimrs.process.loop import DailyOutput
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
    # Properties: (n_fields,)
    awc: np.ndarray,
    rew: np.ndarray,
    tew: np.ndarray,
    cn2: np.ndarray,
    zr_max: np.ndarray,
    zr_min: np.ndarray,
    p_depletion: np.ndarray,
    irr_status: np.ndarray,
    perennial: np.ndarray,
    gw_status: np.ndarray,
    ke_max: np.ndarray,
    f_sub: np.ndarray,
    # Parameters: (n_fields,)
    kc_max: np.ndarray,
    kc_min: np.ndarray,
    ndvi_k: np.ndarray,
    ndvi_0: np.ndarray,
    swe_alpha: np.ndarray,
    swe_beta: np.ndarray,
    kr_damp: np.ndarray,
    ks_damp: np.ndarray,
    max_irr_rate: np.ndarray,
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
):
    """JIT-compiled daily loop using vectorized array operations.

    Scales as O(n_days) with near-constant time for field count.
    """
    # Output arrays
    out_eta = np.zeros((n_days, n_fields), dtype=np.float64)
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

    # Constants
    albedo_min = 0.45
    albedo_max = 0.98
    fresh_snow_threshold = 3.0
    snow_temp_threshold = 1.0
    melt_base_temp = 1.8

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
        albedo = np.where(snow > fresh_snow_threshold, albedo_max,
                  np.where(snow > 0.0,
                           albedo_min + (albedo - albedo_min) * decay_fast,
                           albedo_min + (albedo - albedo_min) * decay_slow))
        albedo = np.maximum(albedo_min, np.minimum(albedo_max, albedo))

        # Degree-day snowmelt
        rad_melt = (1.0 - albedo) * srad * swe_alpha
        dd_melt = (temp_avg - melt_base_temp) * swe_beta
        melt_potential = rad_melt + dd_melt
        melt_potential = np.maximum(melt_potential, 0.0)
        # Melt only when SWE > 0 and tmax > 0
        can_melt = (swe > 0.0) & (tmax > 0.0)
        melt = np.where(can_melt, np.minimum(melt_potential, swe), 0.0)

        actual_melt = melt
        swe = np.maximum(0.0, swe + snow - melt)

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
            awc1 > awc3,
            np.maximum(0.0, np.minimum(1.0, (depl_ze - awc3) / (awc1 - awc3))),
            0.0
        )
        cn_adj = np.where(
            depl_ze < awc3,
            cn3,  # Wet condition
            np.where(depl_ze > awc1, cn1, cn3 + frac * (cn1 - cn3))  # Dry or interpolated
        )

        # Calculate S value (S = 250 * (100/CN - 1) in mm)
        s_new = np.where((cn_adj > 0.0) & (cn_adj < 100.0),
                         250.0 * (100.0 / cn_adj - 1.0), 0.0)

        # Standard SCS runoff
        ia_std = 0.2 * s_new
        runoff_std = np.where(
            (precip_eff > ia_std) & (s_new > 0.0),
            (precip_eff - ia_std) ** 2 / (precip_eff - ia_std + s_new),
            0.0
        )

        # Smoothed runoff for irrigated fields (4-day S average)
        s_avg = (s + s1 + s2 + s3 + s4) / 5.0
        ia_avg = 0.2 * s_avg
        runoff_smooth = np.where(
            (precip_eff > ia_avg) & (s_avg > 0.0),
            (precip_eff - ia_avg) ** 2 / (precip_eff - ia_avg + s_avg),
            0.0
        )

        # Use smoothed for irrigated, standard for others
        runoff = np.where(irr_status > 0.5, runoff_smooth, runoff_std)

        # Update S history
        s4 = s3.copy()
        s3 = s2.copy()
        s2 = s1.copy()
        s1 = s.copy()
        s = s_new

        infiltration = precip_eff - runoff

        # ================================================================
        # 3. CROP COEFFICIENT (Kcb from NDVI - sigmoid)
        # ================================================================
        exp_val = -ndvi_k * (ndvi - ndvi_0)
        exp_val = np.maximum(-20.0, np.minimum(20.0, exp_val))
        kcb = kc_max / (1.0 + np.exp(exp_val))
        kcb = np.maximum(0.0, np.minimum(kc_max, kcb))

        # ================================================================
        # 4. FRACTIONAL COVER
        # ================================================================
        kc_range = kc_max - kc_min
        kcb_for_fc = np.maximum(kcb, kc_min)  # Clip kcb for fc only
        fc = np.where(kc_range > 1e-6, (kcb_for_fc - kc_min) / kc_range, 0.0)
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
        raw = p_depletion * taw

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
            np.where(depl_ze < tew, 1.0, 0.0)
        )
        kr_base = np.minimum(1.0, kr_base)

        # Ks base
        denom_ks = taw - raw
        ks_base = np.where(
            denom_ks > 1e-6,
            np.maximum(0.0, (taw - depl_root) / denom_ks),
            np.where(depl_root < taw, 1.0, 0.0)
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
        # 10. ACTUAL ET
        # ================================================================
        kc_act = ks * kcb * fc + ke
        kc_act = np.minimum(kc_max, kc_act)
        eta = kc_act * etr
        evap = ke * etr

        # ================================================================
        # 11. UPDATE Ze WITH EVAPORATION
        # ================================================================
        depl_ze_prev = depl_ze.copy()
        depl_ze = depl_ze + evap
        depl_ze = np.maximum(depl_ze, 0.0)

        # Cap at TEW and adjust evap
        over_tew = depl_ze > tew
        potential_e = np.maximum(depl_ze - depl_ze_prev, 1e-4)
        e_factor = np.where(
            over_tew,
            np.maximum(0.0, np.minimum(1.0, 1.0 - (depl_ze - tew) / potential_e)),
            1.0
        )
        evap = evap * e_factor
        depl_ze = np.where(
            over_tew,
            np.maximum(depl_ze_prev, 0.0) + evap,
            depl_ze
        )

        # ETf
        etf = np.where(etr > 0.0, eta / etr, 0.0)

        # ================================================================
        # 12. UPDATE ROOT ZONE DEPLETION
        # ================================================================
        depl_after_et = depl_root + eta - infiltration

        # ================================================================
        # 13. IRRIGATION DEMAND (per-field logic)
        # ================================================================
        irr_sim = np.zeros(n_fields, dtype=np.float64)
        for i in range(n_fields):
            if (irr_flag[i] > 0.5 or irr_continue[i] > 0.5) and irr_status[i] > 0.5:
                irr_waiting = next_day_irr[i]

                if next_day_irr[i] > max_irr_rate[i]:
                    next_day_irr[i] = next_day_irr[i] - max_irr_rate[i]
                else:
                    next_day_irr[i] = 0.0

                if irr_flag[i] > 0.5 and depl_after_et[i] > raw[i]:
                    irr_needed = depl_after_et[i] * 1.1
                    if max_irr_rate[i] < irr_needed:
                        next_day_irr[i] = irr_needed - max_irr_rate[i]
                        irr_continue[i] = 1.0
                    else:
                        irr_continue[i] = 0.0
                    irr_sim[i] = min(max_irr_rate[i], irr_needed)
                elif irr_continue[i] > 0.5:
                    irr_sim[i] = min(irr_waiting, max_irr_rate[i])

                # No irrigation if too cold
                if temp_avg[i] < 5.0:
                    irr_sim[i] = 0.0

        # ================================================================
        # 14. GROUNDWATER SUBSIDY
        # ================================================================
        gw_sim = np.where(
            (gw_status > 0.5) & (depl_after_et > raw),
            (depl_after_et - raw) * f_sub,
            0.0
        )

        # ================================================================
        # 15. APPLY IRRIGATION AND GW SUBSIDY
        # ================================================================
        depl_new = depl_after_et - irr_sim - gw_sim

        # ================================================================
        # 16. DEEP PERCOLATION
        # ================================================================
        dperc = np.where(depl_new < 0.0, -depl_new, 0.0)
        depl_new = np.maximum(depl_new, 0.0)

        # Cap at TAW
        depl_root = np.minimum(depl_new, taw)

        # ================================================================
        # 17. LAYER 3 STORAGE
        # ================================================================
        gross_dperc = dperc + 0.1 * irr_sim

        has_taw3 = taw3 > 0.0
        daw3 = daw3 + gross_dperc
        dperc_out = np.where(
            has_taw3 & (daw3 > taw3),
            daw3 - taw3,
            np.where(has_taw3, 0.0, gross_dperc)
        )
        daw3 = np.where(has_taw3, np.minimum(daw3, taw3), daw3)

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

        # Growing: capture water from layer 3, increase depletion for added capacity
        # water_from_l3 = daw3 * delta_zr / (layer3_new_depth + eps)
        water_from_l3 = np.where(
            growing & (layer3_new_depth > 1e-6),
            daw3 * delta_zr / (layer3_new_depth + 1e-6),
            np.where(growing, daw3, 0.0)  # If layer3 fully absorbed, take all
        )
        added_capacity = awc * np.maximum(0.0, delta_zr)
        added_depletion = added_capacity - water_from_l3
        depl_root = np.where(growing, depl_root + added_depletion, depl_root)
        daw3 = np.where(growing, np.maximum(0.0, daw3 - water_from_l3), daw3)
        taw3 = np.where(growing, taw3_new, taw3)

        # Shrinking: water moves from root zone to layer 3
        rt_water_prev = np.where(shrinking, awc * zr_prev - depl_root, 0.0)
        rt_water_prev = np.maximum(0.0, rt_water_prev)
        frac_released = np.where(
            shrinking & (zr_prev > 1e-6),
            np.abs(delta_zr) / zr_prev,
            0.0
        )
        frac_released = np.minimum(1.0, frac_released)
        water_to_l3 = rt_water_prev * frac_released
        daw3 = np.where(shrinking, daw3 + water_to_l3, daw3)
        # Update depletion for reduced capacity
        new_taw = awc * zr_new
        released_capacity = awc * np.abs(delta_zr)
        depl_root = np.where(
            shrinking,
            np.maximum(0.0, depl_root - released_capacity + water_to_l3),
            depl_root
        )
        depl_root = np.where(shrinking, np.minimum(depl_root, new_taw), depl_root)
        taw3 = np.where(shrinking, taw3_new, taw3)

        zr = zr_new

        # ================================================================
        # 19. STORE IRR_SIM FOR NEXT DAY
        # ================================================================
        prev_irr_sim = irr_sim.copy()

        # ================================================================
        # STORE OUTPUTS
        # ================================================================
        out_eta[day_idx, :] = eta
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

    return (
        out_eta, out_etf, out_kcb, out_ke, out_ks, out_kr,
        out_runoff, out_rain, out_melt, out_swe, out_depl_root, out_dperc,
        out_irr_sim, out_gw_sim,
        # Final state
        depl_root, depl_ze, swe, albedo, kr, ks, zr,
        daw3, taw3,
    )


def run_daily_loop_fast(
    swim_input: "SwimInput",
    parameters: "CalibrationParameters | None" = None,
    properties: "FieldProperties | None" = None,
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

    Returns
    -------
    output : DailyOutput
        Daily output arrays with shape (n_days, n_fields)
    final_state : WaterBalanceState
        Final state after simulation
    """
    n_days = swim_input.n_days
    n_fields = swim_input.n_fields
    props = properties if properties is not None else swim_input.properties
    params = parameters if parameters is not None else swim_input.parameters
    spinup = swim_input.spinup_state

    # Pre-load all time series (this is fast - single HDF5 read per variable)
    all_ndvi = swim_input.get_time_series("ndvi").astype(np.float64)
    all_etr = swim_input.get_time_series("etr").astype(np.float64)
    all_prcp = swim_input.get_time_series("prcp").astype(np.float64)
    all_tmin = swim_input.get_time_series("tmin").astype(np.float64)
    all_tmax = swim_input.get_time_series("tmax").astype(np.float64)
    all_srad = swim_input.get_time_series("srad").astype(np.float64)
    all_irr_flag = swim_input.get_irr_flag().astype(np.float64)

    # Extract property arrays
    awc = props.awc.astype(np.float64)
    rew = props.rew.astype(np.float64)
    tew = props.tew.astype(np.float64)
    cn2 = props.cn2.astype(np.float64)
    zr_max = props.zr_max.astype(np.float64)
    zr_min = props.zr_min.astype(np.float64)
    p_depletion = props.p_depletion.astype(np.float64)
    irr_status = props.irr_status.astype(np.float64)
    perennial = props.perennial.astype(np.float64)
    gw_status = props.gw_status.astype(np.float64)
    ke_max = (props.ke_max.astype(np.float64)
              if props.ke_max is not None else np.ones(n_fields))
    f_sub = (props.f_sub.astype(np.float64)
             if props.f_sub is not None else np.zeros(n_fields))

    # Extract parameter arrays
    kc_max = params.kc_max.astype(np.float64)
    kc_min = params.kc_min.astype(np.float64)
    ndvi_k = params.ndvi_k.astype(np.float64)
    ndvi_0 = params.ndvi_0.astype(np.float64)
    swe_alpha = params.swe_alpha.astype(np.float64)
    swe_beta = params.swe_beta.astype(np.float64)
    kr_damp = params.kr_damp.astype(np.float64)
    ks_damp = params.ks_damp.astype(np.float64)
    max_irr_rate = params.max_irr_rate.astype(np.float64)

    # Extract initial state arrays
    depl_root_init = spinup.depl_root.astype(np.float64)
    depl_ze_init = (spinup.depl_ze.astype(np.float64)
                    if spinup.depl_ze is not None else np.zeros(n_fields))
    swe_init = spinup.swe.astype(np.float64)
    albedo_init = (spinup.albedo.astype(np.float64)
                   if spinup.albedo is not None else np.full(n_fields, 0.45))
    kr_init = spinup.kr.astype(np.float64)
    ks_init = spinup.ks.astype(np.float64)
    zr_init = spinup.zr.astype(np.float64)

    # S history for smoothed CN runoff
    default_s = 84.7  # Default S from CN2=75
    s_init = (spinup.s.astype(np.float64)
              if spinup.s is not None else np.full(n_fields, default_s))
    s1_init = (spinup.s1.astype(np.float64)
               if spinup.s1 is not None else np.full(n_fields, default_s))
    s2_init = (spinup.s2.astype(np.float64)
               if spinup.s2 is not None else np.full(n_fields, default_s))
    s3_init = (spinup.s3.astype(np.float64)
               if spinup.s3 is not None else np.full(n_fields, default_s))
    s4_init = (spinup.s4.astype(np.float64)
               if spinup.s4 is not None else np.full(n_fields, default_s))

    # Layer 3 storage
    daw3_init = (spinup.daw3.astype(np.float64)
                 if spinup.daw3 is not None else np.zeros(n_fields))
    taw3_init = (spinup.taw3.astype(np.float64)
                 if spinup.taw3 is not None else np.zeros(n_fields))

    # Run the JIT-compiled loop
    (
        out_eta, out_etf, out_kcb, out_ke, out_ks, out_kr,
        out_runoff, out_rain, out_melt, out_swe, out_depl_root, out_dperc,
        out_irr_sim, out_gw_sim,
        final_depl_root, final_depl_ze, final_swe, final_albedo,
        final_kr, final_ks, final_zr,
        final_daw3, final_taw3,
    ) = _run_loop_jit(
        n_days, n_fields,
        all_ndvi, all_etr, all_prcp, all_tmin, all_tmax, all_srad, all_irr_flag,
        awc, rew, tew, cn2, zr_max, zr_min, p_depletion,
        irr_status, perennial, gw_status, ke_max, f_sub,
        kc_max, kc_min, ndvi_k, ndvi_0, swe_alpha, swe_beta,
        kr_damp, ks_damp, max_irr_rate,
        depl_root_init, depl_ze_init, swe_init, albedo_init,
        kr_init, ks_init, zr_init,
        s_init, s1_init, s2_init, s3_init, s4_init,
        daw3_init, taw3_init,
    )

    # Package outputs into DailyOutput dataclass
    output = DailyOutput(n_days=n_days, n_fields=n_fields)
    output.eta = out_eta
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

    # Package final state
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
