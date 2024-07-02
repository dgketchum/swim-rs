"""compute_crop_et.py
Function for calculating crop et
Called by crop_cycle.py

"""

import logging
import math

import numpy as np

from model.etd import grow_root
from model.etd import runoff
from model.etd import compute_snow


def compute_field_et(config, et_cell, foo, foo_day, debug_flag=False):
    foo.height = np.maximum(0.05, foo.height)

    kc_max = np.maximum(foo.kc_max, foo.kc_bas + 0.05)

    foo.kc_bas = np.maximum(foo.kc_min, foo.kc_bas)

    foo.fc = ((foo.kc_bas - foo.kc_min) / (kc_max - foo.kc_min)) ** (1 + 0.5 * foo.height)

    # limit so that few > 0
    foo.fc = np.minimum(foo.fc, 0.99)
    if np.any(np.isnan(foo.fc)):
        mask = np.isnan(foo.fc).flatten()
        nan_ids = np.array(et_cell.input['order'])[mask]
        for nan_id in nan_ids:
            if not nan_id in foo.isnan:
                foo.isnan.append(nan_id)
                print('Found nan in foo.fc: {}'.format(nan_ids))

    # Estimate infiltrating precipitation
    # Yesterday's infiltration

    foo.ppt_inf_prev = foo.ppt_inf
    foo.ppt_inf = np.zeros_like(foo_day.precip)
    foo.sro = np.zeros_like(foo_day.precip)

    if np.any(foo_day.precip > 0) or np.any(foo.swe > 0.0):

        compute_snow.calculate_snow(foo, foo_day)

        # runoff.runoff_curve_number(foo, foo_day, debug_flag)
        runoff.runoff_infiltration_excess(foo, foo_day)

        foo.ppt_inf = (foo.melt + foo.rain) - foo.sro

    else:
        foo.rain = np.zeros_like(foo_day.precip)
        foo.snow_fall = np.zeros_like(foo_day.precip)
        foo.melt = np.zeros_like(foo_day.precip)
        foo.swe = np.zeros_like(foo_day.precip)

    # setup for water balance of evaporation layer
    # Deep percolation from Ze layer (not root zone, only surface soil)
    foo.depl_ze = foo.depl_ze - (foo.melt + foo.rain + foo.irr_sim)

    foo.few = 1 - foo.fc

    foo.kr = np.minimum((foo.tew - foo.depl_ze) / (foo.tew - foo.rew), 1.)

    foo.ke = np.minimum(foo.kr * (foo.kc_max - foo.kc_bas), foo.few * foo.kc_max)
    foo.ke = np.maximum(foo.ke, 0.0)

    # Transpiration coefficient for moisture stress
    foo.taw = foo.aw * foo.zr
    foo.taw = np.maximum(foo.taw, 0.001)
    foo.taw = np.maximum(foo.taw, foo.tew)
    foo.raw = foo.mad * foo.taw

    foo.ks = np.where(foo.depl_root > foo.raw,
                      np.maximum((foo.taw - foo.depl_root) / (foo.taw - foo.raw), 0), 1)

    if 90 > foo_day.doy > 306:
        # Calculate Kc during snow cover

        kc_mult = np.ones_like(foo_day.swe)
        condition = foo_day.swe > 0.01

        # Radiation term for reducing Kc to actCount for snow albedo
        k_rad = (
            0.000000022 * foo_day.doy ** 3 - 0.0000242 * foo_day.doy ** 2 +
            0.006 * foo_day.doy + 0.011)
        albedo_snow = 0.8
        albedo_soil = 0.25
        kc_mult[condition] = 1 - k_rad + (1 - albedo_snow) / (1 - albedo_soil) * k_rad

        # Was 0.9, reduced another 30% to account for latent heat of fusion of melting snow
        kc_mult = kc_mult * 0.7

        foo.ke *= kc_mult

    else:
        kc_mult = 1

    foo.kc_act = kc_mult * foo.ks * foo.kc_bas * foo.fc + foo.ke

    foo.t = kc_mult * foo.ks * foo.kc_bas * foo.fc

    foo.kc_pot = foo.kc_bas + foo.ke

    foo.etc_act = foo.kc_act * foo_day.refet

    foo.e = foo.ke * foo_day.refet
    depl_ze_prev = foo.depl_ze
    foo.depl_ze = depl_ze_prev + foo.e
    foo.depl_ze = np.where(foo.depl_ze < 0, 0.0, foo.depl_ze)

    if np.any(foo.depl_ze > foo.tew):
        potential_e = foo.depl_ze - depl_ze_prev
        potential_e = np.maximum(potential_e, 0.0001)
        e_factor = 1 - (foo.depl_ze - foo.tew) / potential_e
        e_factor = np.minimum(np.maximum(e_factor, 0), 1)
        foo.e *= e_factor
        foo.depl_ze = depl_ze_prev + foo.e

    foo.cum_evap_prev = foo.cum_evap_prev + foo.e - (foo.ppt_inf - depl_ze_prev)
    foo.cum_evap_prev = np.maximum(foo.cum_evap_prev, 0)

    foo.soil_water_prev = foo.soil_water.copy()
    foo.depl_root_prev = foo.depl_root.copy()
    foo.depl_root += foo.etc_act - foo.ppt_inf

    foo.irr_sim = np.zeros_like(foo.aw)

    if np.any(foo_day.irr_day) or np.any(foo.irr_continue):
        # account for the case where depletion exceeded the maximum daily irr rate yesterday
        irr_waiting = foo.next_day_irr

        foo.next_day_irr = np.where(foo.next_day_irr > foo.max_irr_rate,
                                    foo.next_day_irr - foo.max_irr_rate,
                                    0.0)

        next_day_cond = (foo_day.irr_day & (foo.depl_root > foo.raw) & (foo.max_irr_rate < foo.depl_root * 1.1))
        foo.next_day_irr = np.where(next_day_cond,
                                    foo.depl_root * 1.1 - foo.max_irr_rate,
                                    foo.next_day_irr)

        potential_irr = np.where(foo.irr_continue, np.minimum(irr_waiting, foo.max_irr_rate), 0.0)

        potential_irr = np.where((foo_day.irr_day & (foo.depl_root > foo.raw)),
                                 np.minimum(foo.max_irr_rate, foo.depl_root * 1.1), potential_irr)

        foo.irr_continue = np.where((foo_day.irr_day & (foo.max_irr_rate < foo.depl_root * 1.1)), 1, 0)

        foo.irr_sim = potential_irr

    # Update depletion of root zone

    foo.depl_root -= foo.irr_sim

    foo.cum_evap[foo.irr_sim > 0] = foo.cum_evap_prev[foo.irr_sim > 0]
    foo.cum_evap_prev[foo.irr_sim > 0] = 0.0

    foo.dperc = np.where(foo.depl_root < 0.0, -1. * foo.depl_root, np.zeros_like(foo.depl_root))
    foo.depl_root += foo.dperc
    foo.depl_root = np.where(foo.depl_root > foo.taw, foo.taw, foo.depl_root)

    gross_dperc = foo.dperc + (0.1 * foo.irr_sim)

    # aw3 is mm/m and daw3 is mm in layer 3.
    # aw3 is layer between current root depth and max root

    foo.daw3_prev = foo.daw3.copy()

    # foo.taw3 = foo.aw * (foo.zr_max - foo.zr)
    # foo.daw3 = np.maximum(foo.daw3, 0)
    # foo.taw3 = np.maximum(foo.taw3, 0)

    # Increase water in layer 3 for deep percolation from root zone

    foo.daw3 += gross_dperc
    foo.daw3 = np.maximum(foo.daw3, 0.0)

    foo.dperc = np.where(foo.daw3 > foo.taw3, foo.daw3 - foo.taw3, np.zeros_like(foo.dperc))
    foo.daw3 = np.where(foo.daw3 > foo.taw3, foo.taw3, foo.daw3)

    foo.aw3 = np.where(foo.zr_max > foo.zr, foo.daw3 / (foo.zr_max - foo.zr), np.zeros_like(foo.aw3))

    foo.niwr = np.where(foo.irr_sim > 0, foo.etc_act - ((foo.melt + foo.rain) - foo.sro),
                        foo.etc_act - ((foo.melt + foo.rain) - foo.sro - foo.dperc))

    foo.p_rz = np.where(foo.irr_sim > 0, (foo.melt + foo.rain) - foo.sro, (foo.melt + foo.rain) - foo.sro - foo.dperc)
    foo.p_rz = np.maximum(foo.p_rz, 0)

    foo.p_eft = np.where(foo.irr_sim > 0, (foo.melt + foo.rain) - foo.sro - foo.e,
                         (foo.melt + foo.rain) - foo.sro - foo.dperc - foo.e)
    foo.p_eft = np.maximum(foo.p_eft, 0)

    # Note, at end of season (harvest or death), aw3 and zr need to be reset
    #   according to depl_root at that time and zr for dormant season.
    # This is done in setup_dormant().

    # Get setup for next time step.
    # if foo.in_season:

    foo.delta_daw3 = foo.daw3 - foo.daw3_prev

    foo.soil_water = (foo.aw * foo.zr) - foo.depl_root + foo.daw3

    foo.delta_soil_water = foo.soil_water - foo.soil_water_prev

    grow_root.grow_root(foo, foo_day, debug_flag)


