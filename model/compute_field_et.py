import numpy as np

from model import grow_root
from model import runoff
from model import compute_snow


def compute_field_et(ts_data, swb, day_data):

    kc_max = np.maximum(swb.kc_max, swb.kc_bas + 0.05)

    swb.kc_bas = np.maximum(swb.kc_min, swb.kc_bas)

    # consider height removal
    swb.fc = ((swb.kc_bas - swb.kc_min) / (kc_max - swb.kc_min)) # ** (1 + 0.5 * swb.height)

    # limit so that few > 0
    swb.fc = np.minimum(swb.fc, 0.99)
    if np.any(np.isnan(swb.fc)):
        mask = np.isnan(swb.fc).flatten()
        nan_ids = np.array(ts_data.input['order'])[mask]
        for nan_id in nan_ids:
            if not nan_id in swb.isnan:
                swb.isnan.append(nan_id)
                print('Found nan in foo.fc: {}'.format(nan_ids))

    # Estimate infiltrating precipitation
    # Yesterday's infiltration

    swb.ppt_inf_prev = swb.ppt_inf
    swb.ppt_inf = np.zeros_like(day_data.precip)
    swb.sro = np.zeros_like(day_data.precip)

    if np.any(day_data.precip > 0) or np.any(swb.swe > 0.0):

        compute_snow.calculate_snow(swb, day_data)

        # runoff.runoff_curve_number(foo, foo_day, debug_flag)
        runoff.runoff_infiltration_excess(swb, day_data)

        swb.ppt_inf = (swb.melt + swb.rain) - swb.sro

    else:
        swb.rain = np.zeros_like(day_data.precip)
        swb.snow_fall = np.zeros_like(day_data.precip)
        swb.melt = np.zeros_like(day_data.precip)
        swb.swe = np.zeros_like(day_data.precip)

    # setup for water balance of evaporation layer
    # Deep percolation from Ze layer (not root zone, only surface soil)
    swb.depl_ze = swb.depl_ze - (swb.melt + swb.rain + swb.irr_sim)

    swb.few = 1 - swb.fc

    swb.kr = np.minimum((swb.tew - swb.depl_ze) / (swb.tew - swb.rew + 1e-6), 1.)

    swb.ke = np.minimum(swb.kr * (swb.kc_max - swb.kc_bas), swb.few * swb.kc_max)
    swb.ke = np.maximum(swb.ke, 0.0)
    swb.ke = np.minimum(swb.ke, swb.ke_max)

    # Transpiration coefficient for moisture stress
    swb.taw = swb.aw * swb.zr
    swb.taw = np.maximum(swb.taw, 0.001)
    swb.taw = np.maximum(swb.taw, swb.tew)
    swb.raw = swb.mad * swb.taw

    swb.ks = np.where(swb.depl_root > swb.raw,
                      np.maximum((swb.taw - swb.depl_root) / (swb.taw - swb.raw), 0), 1)

    if np.any(swb.swe > 0.0):
        # Calculate Kc during snow cover

        kc_mult = np.ones_like(swb.swe)
        condition = swb.swe > 0.01

        # Radiation term for reducing Kc to actCount for snow albedo
        k_rad = (
            0.000000022 * day_data.doy ** 3 - 0.0000242 * day_data.doy ** 2 +
            0.006 * day_data.doy + 0.011)
        albedo_snow = 0.8
        albedo_soil = 0.25
        kc_mult[condition] = 1 - k_rad + (1 - albedo_snow) / (1 - albedo_soil) * k_rad

        # Was 0.9, reduced another 30% to account for latent heat of fusion of melting snow
        kc_mult = kc_mult * 0.7

        swb.ke *= kc_mult

    else:
        kc_mult = 1

    swb.kc_act = kc_mult * swb.ks * swb.kc_bas * swb.fc + swb.ke

    swb.kc_act = np.minimum(swb.kc_max, swb.kc_act)

    swb.t = kc_mult * swb.ks * swb.kc_bas * swb.fc

    swb.kc_pot = swb.kc_bas + swb.ke

    swb.etc_act = swb.kc_act * day_data.refet

    swb.e = swb.ke * day_data.refet
    depl_ze_prev = swb.depl_ze
    swb.depl_ze = depl_ze_prev + swb.e
    swb.depl_ze = np.where(swb.depl_ze < 0, 0.0, swb.depl_ze)

    if np.any(swb.depl_ze > swb.tew):
        potential_e = swb.depl_ze - depl_ze_prev
        potential_e = np.maximum(potential_e, 0.0001)
        e_factor = 1 - (swb.depl_ze - swb.tew) / potential_e
        e_factor = np.minimum(np.maximum(e_factor, 0), 1)
        swb.e *= e_factor
        swb.depl_ze = depl_ze_prev + swb.e

    swb.cum_evap_prev = swb.cum_evap_prev + swb.e - (swb.ppt_inf - depl_ze_prev)
    swb.cum_evap_prev = np.maximum(swb.cum_evap_prev, 0)

    swb.soil_water_prev = swb.soil_water.copy()
    swb.depl_root_prev = swb.depl_root.copy()
    swb.depl_root += swb.etc_act - swb.ppt_inf

    swb.irr_sim = np.zeros_like(swb.aw)

    if np.any(day_data.irr_day) or np.any(swb.irr_continue):
        # account for the case where depletion exceeded the maximum daily irr rate yesterday
        irr_waiting = swb.next_day_irr

        swb.next_day_irr = np.where(swb.next_day_irr > swb.max_irr_rate,
                                    swb.next_day_irr - swb.max_irr_rate,
                                    0.0)

        next_day_cond = (day_data.irr_day & (swb.depl_root > swb.raw) & (swb.max_irr_rate < swb.depl_root * 1.1))
        swb.next_day_irr = np.where(next_day_cond,
                                    swb.depl_root * 1.1 - swb.max_irr_rate,
                                    swb.next_day_irr)

        potential_irr = np.where(swb.irr_continue, np.minimum(irr_waiting, swb.max_irr_rate), 0.0)

        potential_irr = np.where((day_data.irr_day & (swb.depl_root > swb.raw)),
                                 np.minimum(swb.max_irr_rate, swb.depl_root * 1.1), potential_irr)

        # if np.any(potential_irr > 0.) and foo_day.doy > 190 and foo_day.irr_day[0, 46]:
        #     a = 1

        swb.irr_continue = np.where((day_data.irr_day & (swb.max_irr_rate < swb.depl_root * 1.1)), 1, 0)

        swb.irr_sim = potential_irr

    swb.gw_sim = np.zeros_like(swb.aw)

    if np.any(day_data.gwsub_status) and np.any((swb.depl_root > swb.raw)):

        gw_subsidy = np.where(day_data.gwsub_status, swb.depl_root - swb.raw, 0.0)
        swb.gw_sim = gw_subsidy

    # Update depletion of root zone

    swb.depl_root -= (swb.irr_sim + swb.gw_sim)

    swb.cum_evap[swb.irr_sim > 0] = swb.cum_evap_prev[swb.irr_sim > 0]
    swb.cum_evap_prev[swb.irr_sim > 0] = 0.0

    swb.dperc = np.where(swb.depl_root < 0.0, -1. * swb.depl_root, np.zeros_like(swb.depl_root))
    swb.depl_root += swb.dperc
    swb.depl_root = np.where(swb.depl_root > swb.taw, swb.taw, swb.depl_root)

    gross_dperc = swb.dperc + (0.1 * swb.irr_sim)

    # aw3 is mm/m and daw3 is mm in layer 3.
    # aw3 is layer between current root depth and max root

    swb.daw3_prev = swb.daw3.copy()

    # foo.taw3 = foo.aw * (foo.zr_max - foo.zr)
    # foo.daw3 = np.maximum(foo.daw3, 0)
    # foo.taw3 = np.maximum(foo.taw3, 0)

    # Increase water in layer 3 for deep percolation from root zone

    swb.daw3 += gross_dperc
    swb.daw3 = np.maximum(swb.daw3, 0.0)

    swb.dperc = np.where(swb.daw3 > swb.taw3, swb.daw3 - swb.taw3, np.zeros_like(swb.dperc))
    swb.daw3 = np.where(swb.daw3 > swb.taw3, swb.taw3, swb.daw3)

    swb.aw3 = np.where(swb.zr_max > swb.zr, swb.daw3 / (swb.zr_max - swb.zr), np.zeros_like(swb.aw3))

    swb.niwr = np.where(swb.irr_sim > 0, swb.etc_act - ((swb.melt + swb.rain) - swb.sro),
                        swb.etc_act - ((swb.melt + swb.rain) - swb.sro - swb.dperc))

    swb.p_rz = np.where(swb.irr_sim > 0, (swb.melt + swb.rain) - swb.sro, (swb.melt + swb.rain) - swb.sro - swb.dperc)
    swb.p_rz = np.maximum(swb.p_rz, 0)

    swb.p_eft = np.where(swb.irr_sim > 0, (swb.melt + swb.rain) - swb.sro - swb.e,
                         (swb.melt + swb.rain) - swb.sro - swb.dperc - swb.e)
    swb.p_eft = np.maximum(swb.p_eft, 0)

    swb.delta_daw3 = swb.daw3 - swb.daw3_prev

    swb.soil_water = (swb.aw * swb.zr) - swb.depl_root + swb.daw3

    swb.delta_soil_water = swb.soil_water - swb.soil_water_prev

    grow_root.grow_root(swb)


