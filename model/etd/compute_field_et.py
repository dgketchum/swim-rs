"""compute_crop_et.py
Function for calculating crop et
Called by crop_cycle.py

"""

import datetime
import logging
import math
import sys

import numpy as np

from fieldET import grow_root
from fieldET import util
from fieldET import runoff


def compute_field_et(config, et_cell, foo, foo_day, debug_flag=False):

    # Maximum Kc when soil is wet.  For grass reference, kc_max = 1.2 plus climatic adj.
    # For alfalfa reference, kc_max = 1.0, with no climatic adj.
    # kc_max is set to less than 1.0 during winter to account for effects of cold soil.

    # ETo basis:  Switched over to this 12/2007  # Allen and Huntington
    # Note that U2 and RHmin were disabled when KcETr code was converted to ETr basis
    #   these have been reactivated 12/2007 by Allen, based on daily wind and TDew
    # RHmin and U2 are computed in ETCell.set_weather_data()

    # Limit height for numerical stability

    foo.height = max(0.05, foo.height)

    if et_cell.props['irr'][str(foo_day.year)] < 0.5:
        fallow = True
    else:
        fallow = False

    kc_max = max(foo.kc_max, foo.kc_bas + 0.05)

    # Use same value for both ETr or ETo bases.

    foo.kc_min = 0.1
    foo.kc_bas = max(foo.kc_min, foo.kc_bas)

    # Estimate height of vegetation for estimating fraction of ground cover
    #   for evaporation and fraction of ground covered by vegetation

    # heightcalc  #'call to heightcalc was moved to top of this subroutine 12/26/07 by Allen
    # foo.fc = ((foo.kc_bas - foo.kc_min) / (kc_max - foo.kc_min)) ** (1 + 0.5 * foo.height)

    # dgketchum hack tunable cover fraction function
    if fallow or config.field_type == 'unirrigated':
        cover_proxy = '{}_inv_irr'.format(config.cover_proxy)
    else:
        cover_proxy = '{}_irr'.format(config.cover_proxy)
    foo.ndvi = et_cell.input[foo_day.dt_string][cover_proxy]
    foo.fc = foo.ndvi_fc * foo.ndvi

    # limit so that few > 0
    foo.fc = min(foo.fc, 0.99)
    if np.isnan(foo.fc):
        raise ValueError

    # Estimate infiltrating precipitation

    # Yesterday's infiltration

    foo.ppt_inf_prev = foo.ppt_inf
    foo.ppt_inf = 0.0
    foo.sro = 0.0

    if foo_day.precip > 0:
        # Compute weighted depletion of surface from irr and precip areas
        foo.depl_surface = foo.wt_irr * foo.depl_ze + (1 - foo.wt_irr) * foo.depl_zep
        runoff.runoff(foo, foo_day, debug_flag)
        foo.ppt_inf = foo_day.precip - foo.sro

    # Compare precipitation and irrigation to determine value for fw

    # At this point, irrigation depth, Irr is based on yesterday's irrigations
    # (irrig has not yet been updated)
    # Note: In Idaho CU computations, scheduling is assumed automated according to MAD
    # Following code contains capacity to specify manual and #'special' irrigations, but isn't used here

    # irr_real is a real irrigation experienced and read in
    # irr_manual is a manually specified irrigation from an array
    # irr_special is a special irrigation for leaching or germination

    irr_real = 0.0
    irr_manual = 0.0
    irr_special = 0.0

    # Update fw of irrigation if an irrigation yesterday

    if (irr_real + foo.irr_auto) > 0:
        foo.fw_irr = foo.fw_std
    elif (irr_manual + irr_special) > 0:
        foo.fw_irr = foo.fw_spec

    # find current water in fw_irr portion of ze layer

    # [140820] changedtests below to round(watin_ze?,6) in both py/vb for consistency
    # both versions tested values ~1e-15, which were being passed through
    # and would happen inconsistently between py/vb versions

    watin_ze = foo.tew - foo.depl_ze

    # if watin_ze <= 0.:
    if round(watin_ze, 6) <= 0.:
        watin_ze = 0.001
    watin_ze = min(watin_ze, foo.tew)

    # Find current water in fwp portion of Ze layer
    # use of 'fewp' (precip) fraction of surface

    watin_zep = foo.tew - foo.depl_zep  # follows Allen et al. 2005 (ASCE JIDE) extensions

    # Modified to be consistent with python - 09/25/2014
    # if watin_zep <= 0.:

    if round(watin_zep, 6) <= 0.:
        watin_zep = 0.001
    watin_zep = min(watin_zep, foo.tew)

    # Fraction of ground that is both exposed and wet

    foo.few = 1 - foo.fc

    # Limit to fraction wetted by irrigation
    # dgketchum limit this to irrigated field type
    # and that fw = 1 - fc
    if config.field_type == 'irrigated':
        # foo.few = min(max(foo.few, 0.001), foo.fw_irr)
        foo.few = max(foo.few, 0.001)
        foo.fw_irr = 1 - foo.fc
    else:
        foo.few = 0.
        foo.fw_irr = 0.

    # Fraction of ground that is exposed and wet by precip beyond irrigation

    foo.fewp = 1 - foo.fc - foo.few
    foo.fewp = max(foo.fewp, 0.001)

    # Was "totwatin_ze = watin_ze * few + watin_zep * fewp" until 5/9/07
    # (corrected)

    foo.totwatin_ze = (watin_ze * foo.few + watin_zep * foo.fewp) / (foo.few + foo.fewp)

    # tew is total evaporable water (end of 2nd or 3rd stage)
    # rew is readily evaporable water (end of stage 1)
    # depl_ze is depletion of evaporation layer wetted by irrigation and exposed
    # depl_ze is computed here each day and retained as static
    # depl_ze should be initialized at start of each new crop in crop cycle routine
    # depl_zep is depletion of evaporation layer wetted by Precip beyond irrigation

    # setup for water balance of evaporation layer

    # Deep percolation from Ze layer (not root zone, only surface soil)

    if foo.fw_irr > 0.0001:
        # depl_ze, irr from yesterday
        # fw changed to foo.fw_irr 8/10/06

        dperc_ze = foo.ppt_inf + foo.irr_sim / foo.fw_irr - foo.depl_ze
    else:
        # depl_ze, irr from yesterday
        # fw changed to fw_irr 8/10/06

        dperc_ze = foo.ppt_inf + foo.irr_sim / 1 - foo.depl_ze
    dperc_ze = max(dperc_ze, 0)

    # depl_zep from yesterday (this was called Dpep in TP's code)

    depl_zep_prev = foo.ppt_inf - foo.depl_zep
    depl_zep_prev = max(depl_zep_prev, 0)

    # Compute initial balance of Ze layer.  E and T from Ze layer
    # will be added later.  De is depletion of Ze layer, mm
    # ppt_inf is P that infiltrates
    # This balance is here, since irr is yesterday's irrigation and
    # it is assumed to be in morning before E and T for day have occurred.
    # It is assumed that P occurs earlier in morning.

    if foo.fw_irr > 0.0001:
        # fw changed to fw_irr 8/10/06
        foo.depl_ze = foo.depl_ze - foo.ppt_inf - foo.irr_sim / foo.fw_irr + dperc_ze
    else:
        # fw changed to fw_irr 8/10/06

        foo.depl_ze = foo.depl_ze - foo.ppt_inf - foo.irr_sim / 1 + dperc_ze

    # Use TEW rather than TEW2use to conserve depl_ze
    if config.field_type == 'unirrigated':
        foo.depl_ze = 0.0
    else:
        foo.depl_ze = min(max(foo.depl_ze, 0), foo.tew)

    # Update depletion of few beyond that wetted by irrigation

    foo.depl_zep = foo.depl_zep - foo.ppt_inf + depl_zep_prev
    foo.depl_zep = min(max(foo.depl_zep, 0), foo.tew)

    # reducer coefficient for evaporation based on moisture left
    # This is set up for three stage evaporation
    # REW is depletion at end of stage 1 (energy limiting), mm
    # TEW2 is depletion at end of stage 2 (typical soil), mm
    # TEW3 is depletion at end of stage 3 (rare), mm
    # Stage 3 represents a cracking soil where cracks open on drying
    # Kr2 is value for Kr at transition from stage 2 to 3
    #   i.e., when depletion is equal to TEW2.
    # for example, for a cracking clay loam soil,
    #     REW=8 mm, TEW2=50 mm, TEW3=100 mm and Kr2=0.2
    # for a noncracking clay loam soil, REW=8 mm, TEW2=50 mm, TEW3=0, Kr2=0
    # make sure that Kr2 is 0 if TEW3=0

    if foo.tew3 < 0.1:
        foo.kr2 = 0.0

    # De is depletion of evaporation layer, mm

    # make sure De does not exceed depl_root (depl. of root zone), since depl_root includes De.
    # IF De > depl_root THEN De = depl_root   #'this causes problems during offseason.  don't use

    # For portion of surface that has been wetted by irrigation and precipitation
    #   reduce TEW (and REW) during winter when ETr drops below 4 mm/day (FAO-56)

    tew2use = foo.tew2
    tew3use = foo.tew3  # for stage 3 drying (cracking soils (not in Idaho))
    rew2use = foo.rew
    foo.etref_30 = max(0.1, foo.etref_30)  # mm/day  #'edited from ETr to ETref 12/26/2007
    if config.refet_type == 'eto':
        etr_threshold = 5  # for ETo basis #'added March 26, 2008 RGA
    elif config.refet_type == 'etr':
        etr_threshold = 4  # for ETr basis

    # Use 30 day ETr, if less than 4 or 5 mm/d to reduce TEW

    if foo.etref_30 < etr_threshold:
        tew2use = foo.tew2 * math.sqrt(foo.etref_30 / etr_threshold)
        tew3use = foo.tew3 * math.sqrt(foo.etref_30 / etr_threshold)
        if rew2use > 0.8 * tew2use:
            # Limit REW to 30% less than TEW
            # Was 0.7 until 4/16/08

            rew2use = 0.8 * tew2use

    if foo.depl_ze <= rew2use:
        kr = 1
    else:
        if foo.depl_ze <= tew2use:
            kr = foo.kr2 + (1 - foo.kr2) * (tew2use - foo.depl_ze) / (tew2use - rew2use)
        else:
            if tew3use > tew2use:  # Stage 3 drying (cracking soils)
                kr = foo.kr2 * (tew3use - foo.depl_ze) / (tew3use - tew2use)
            else:
                kr = 0.0

    # Portion of surface that has been wetted by precipitation

    if foo.depl_zep <= rew2use:
        krp = 1
    else:
        if foo.depl_zep <= tew2use:
            krp = foo.kr2 + (1 - foo.kr2) * (tew2use - foo.depl_zep) / (tew2use - rew2use)
        else:
            if tew3use > tew2use:  # Stage 3 drying (cracking soils)
                krp = foo.kr2 * (tew3use - foo.depl_zep) / (tew3use - tew2use)
            else:
                krp = 0.0

    # evaporation coefficient Ke
    # partition Ke into that from irrigation wetted and from precip wetted
    # Kelimit = (few + fewp) * kc_max
    # find weighting factor based on water in Ze layer in irrig. wetted and precip wetted

    # following conditional added July 2006 for when denominator is zero

    if (foo.few * watin_ze + foo.fewp * watin_zep) > 0.0001:
        foo.wt_irr = foo.few * watin_ze / (foo.few * watin_ze + foo.fewp * watin_zep)
    else:
        foo.wt_irr = foo.few * watin_ze
    foo.wt_irr = min(max(foo.wt_irr, 0), 1)
    if debug_flag:
        logging.debug('compute_crop_et(): wt_irr %.6f' % (foo.wt_irr))

    # Ke = Kr * (kc_max - foo.kc_bas)  # this was generic for irr + precip
    # IF Ke > few * kc_max THEN Ke = few * kc_max

    ke_irr = kr * (kc_max - foo.kc_bas) * foo.wt_irr

    ke_ppt = krp * (kc_max - foo.kc_bas) * (1 - foo.wt_irr)

    # Limit to maximum rate per unit surface area

    ke_irr = min(max(ke_irr, 0), foo.few * kc_max)

    ke_ppt = min(max(ke_ppt, 0), foo.fewp * kc_max)

    foo.ke = ke_irr + ke_ppt

    # Transpiration coefficient for moisture stress

    taw = foo.aw * foo.zr
    taw = max(taw, 0.001)

    # MAD: Management Allowable Depletion
    # MAD is set to mad_ini or mad_mid in kcb_daily sub.
    # dgketchum reimplement Allen 2005 form
    foo.raw = foo.mad * taw

    # Remember to check reset of AD and RAW each new crop season.  #####
    # AD is allowable depletion

    if foo.depl_root > foo.raw:
        foo.ks = max((taw - foo.depl_root) / (taw - foo.raw), 0)
    else:
        foo.ks = 1

    # Calculate Kc during snow cover

    kc_mult = 1
    if foo_day.snow_depth > 0.01:
        # Radiation term for reducing Kc to actCount for snow albedo

        k_rad = (
            0.000000022 * foo_day.doy ** 3 - 0.0000242 * foo_day.doy ** 2 +
            0.006 * foo_day.doy + 0.011)
        albedo_snow = 0.8
        albedo_soil = 0.25
        kc_mult = 1 - k_rad + (1 - albedo_snow) / (1 - albedo_soil) * k_rad

        # Was 0.9, reduced another 30% to account for latent heat of fusion of melting snow

        kc_mult = kc_mult * 0.7

    foo.ke *= kc_mult
    ke_irr *= kc_mult
    ke_ppt *= kc_mult

    # Don't reduce Kc_bas, since it may be held constant during non-growing periods.
    # Make adjustment to kc_act

    foo.kc_act = kc_mult * foo.ks * foo.kc_bas + foo.ke

    foo.kc_pot = foo.kc_bas + foo.ke

    # ETref is defined (to ETo or ETr) in CropCycle sub #'Allen 12/26/2007

    foo.etc_act = foo.kc_act * foo_day.etref
    foo.etc_pot = foo.kc_pot * foo_day.etref
    foo.etc_bas = foo.kc_bas * foo_day.etref

    e = foo.ke * foo_day.etref
    e_irr = ke_irr * foo_day.etref
    e_ppt = ke_ppt * foo_day.etref

    # Begin Water balance of evaporation layer and root zone

    # Transpiration from Ze layer
    # transpiration proportioning

    # CGM - For now, just set to target value

    ze = 0.0001

    # TP - ze never initialized, assume 0.0 value
    #   Also used in SetupDormant(), but value set explicitly
    #   Wonder if was meant to be global????
    # ze = 0.0   # I added this line
    # ze = max(ze, 0.0001)
    # # if ze < 0.0001:
    # #     ze = 0.0001

    if foo.zr < 0.0001:
        foo.zr = 0.01
    kt_prop = (ze / foo.zr) ** 0.6

    # if kt_prop > 1:
    #     _prop = 1

    kt_prop = min(kt_prop, 1)

    # Zr is root depth, m
    # depl_root is depletion in root zone, mm
    # AW is available water for soil, mm/m

    # For irrigation wetted fraction

    kt_reducer_denom = max(1 - foo.depl_root / taw, 0.001)

    # few added, 8/2006, that is not in Allen et al., 2005, ASCE

    kt_reducer = foo.few * (1 - foo.depl_ze / tew2use) / kt_reducer_denom
    kt_prop = kt_prop * kt_reducer

    # kt_reducer can be greater than 1

    kt_prop = min(kt_prop, 1)

    # this had a few in equation as compared to Allen et al., 2005, ASCE

    te_irr = kc_mult * foo.ks * foo.kc_bas * foo_day.etref * kt_prop

    # For precip wetted fraction beyond that irrigated
    # fewp added, 8/2006, that is not in Allen et al., 2005, ASCE

    kt_reducer = foo.fewp * (1 - foo.depl_zep / tew2use) / kt_reducer_denom
    kt_prop = kt_prop * kt_reducer

    # kt_reducer can be greater than 1

    kt_prop = min(kt_prop, 1)

    # this had a fewp in equation as compared to Allen et al., 2005, ASCE

    te_ppt = kc_mult * foo.ks * foo.kc_bas * foo_day.etref * kt_prop

    # Setup for water balance of evaporation layer

    depl_ze_prev = foo.depl_ze
    depl_zep_prev = foo.depl_zep

    # if total profile is bone dry from a dry down, then any root
    # extraction from a rain or light watering may all come from the
    # evaporating layer.  Therefore, actCount for transpiration extraction
    # of water from Ze layer that will increase depletion of that layer

    # Available water in Zr includes water in Ze layer.  Therefore limit depl_ze.
    AvailWatinTotalZr = taw - foo.depl_root

    # leave following out, for noncrop situations
    # IF Delast + Deplast < TEW - AvailWatinTotalZr THEN
    #   Delast = TEW - AvailWatinTotalZr #'soil is very dry
    #  Deplast = TEW
    # END IF

    # finish water balance of Ze evaporation layer
    # (ptt_inf, irr and dperc_ze were subtracted or added earlier)

    if config.field_type == 'irrigated':
        foo.depl_ze = depl_ze_prev + e_irr / foo.few + te_irr
        logging.debug('compute_crop_et(): depl_ze %.6f' % (foo.depl_ze))

    # This next section modified 2/21/08 to keep a days potential E from exceeding
    # Evaporable water available (for coarse soils).  Allen and Huntington

    if foo.depl_ze < 0:
        foo.depl_ze = 0.0
        logging.debug('compute_crop_et(): depl_ze %.6f' % (foo.depl_ze))
    if foo.depl_ze > foo.tew:
        # use tew here rather than tew2use to allow depl_ze to remain at tew
        # '''  probably not.  if Delast <= 0:    Delast = 0

        potential_e = foo.depl_ze - depl_ze_prev
        if potential_e < 0.0001: potential_e = 0.0001
        e_factor = 1 - (foo.depl_ze - foo.tew) / potential_e
        e_factor = min(max(e_factor, 0), 1)
        e_irr *= e_factor
        te_irr *= e_factor
        foo.depl_ze = depl_ze_prev + e_irr / foo.few + te_irr  # recalculate
        logging.debug('compute_crop_et(): depl_ze %.6f' % (foo.depl_ze))
        if foo.depl_ze > foo.tew + 0.2:
            logging.warning(
                ('Problem in keeping depl_ze water balance within TEW.' +
                 'depl_ze, TEW, e_irr, te_irr, e_factor = {} {} {} {} {}').format(
                    foo.depl_ze, foo.tew, e_irr, te_irr, e_factor))
            return
    foo.depl_zep = depl_zep_prev + e_ppt / foo.fewp + te_ppt
    foo.depl_zep = max(foo.depl_zep, 0)

    if foo.depl_zep > foo.tew:
        # 'depl_zep = TEW

        potential_e = foo.depl_zep - depl_zep_prev
        if potential_e < 0.0001:
            potential_e = 0.0001
        e_factor = 1 - (foo.depl_zep - foo.tew) / potential_e
        e_factor = min(max(e_factor, 0), 1)
        e_ppt *= e_factor
        te_ppt *= e_factor
        foo.depl_zep = depl_zep_prev + e_ppt / foo.fewp + te_ppt  # recalculate
        if foo.depl_zep > foo.tew + 0.2:
            logging.warning(
                ('Problem in keeping De water balance within TEW.  ' +
                 'De, TEW, E_irr, te_irr, e_factor = {} {} {} {} {}').format(
                    foo.depl_ze, foo.tew, e_irr, te_irr, e_factor))
            return

    # Recomputed these based on corrections above if depl_ze > TEW  2/21/08

    etref_divisor = foo_day.etref
    if etref_divisor < 0.01:
        etref_divisor = 0.01
    ke_irr = e_irr / etref_divisor
    ke_ppt = e_ppt / etref_divisor

    # limit for when ETref is super small

    ke_irr = min(max(ke_irr, 0), 1.5)
    ke_ppt = min(max(ke_ppt, 0), 1.5)
    foo.ke = ke_irr + ke_ppt
    e = foo.ke * foo_day.etref
    if kc_mult > 1:
        logging.warning("kcmult > 1.")
        return
    if foo.ks > 1:
        logging.warning("ks > 1.")
        return
    foo.kc_act = kc_mult * foo.ks * foo.kc_bas + foo.ke
    foo.kc_pot = foo.kc_bas + foo.ke

    # # CO2 correction
    # if data.co2_flag:
    #      foo.kc_bas *= foo_day.co2
    #     .kc_act *= foo_day.co2
    #     .kc_pot *= foo_day.co2
    #      debug_flag:
    #        logging.debug(
    #            ('compute_crop_et(): co2 %.6f  Kc_pot %.6f  Kc_act %.6f') %
    #            (foo_day.co2, foo.kc_pot, foo.kc_act))
    #        logging.debug(
    #            ('compute_crop_et(): ETcpot %.6f  ETcact %.6f') %
    #            ( foo.etc_pot, foo.etc_act))

    # etref is defined (to eto or etr) in crop_cycle sub'Allen 12/26/2007
    # Note that etc_act will be checked later against depl_root and TAW

    foo.etc_act = foo.kc_act * foo_day.etref
    foo.etc_pot = foo.kc_pot * foo_day.etref
    foo.etc_bas = foo.kc_bas * foo_day.etref

    # Accumulate evaporation following each irrigation event.
    # Subtract evaporation from precipitation.
    # Precipitation evaporation is set to evaporation that would have occurred
    #   from precipitation in absence of irrigation, and is estimated as
    #   infiltrated P less deep percolation of P from evaporation layer for P
    #   only (if no irrigation).
    # This was moved about 40 lines down 2/21/08 to be after adjustment to Ke, E, etc. made just above here.

    foo.cum_evap_prev = foo.cum_evap_prev + e_irr - (foo.ppt_inf - depl_zep_prev)
    foo.cum_evap_prev = max(foo.cum_evap_prev, 0)

    # Get irrigation information
    # this is done after computation of Ke, since it is assumed that
    # irrigations occur in pm of day, on average.
    # this can / should be changed (moved ahead of Ke computation) for
    # special/manual irrigations that are known to occur in am

    # irr_real is a real irrigation experienced and read in
    # irr_manual is a manually specified irrigation from an array
    # irr_special is a special irrigation for leaching or germination
    # for now, irr_real and irr_manual are place holders (zero)
    # irr_special is determined from crop data read in

    # water balance for Zr root zone (includes Ze layer)
    # Initial calculation for depl_root

    if debug_flag:
        logging.debug(
            ('compute_crop_et(): depl_root %.6f  etc_act %.6f  ppt_inf %.6f') %
            (foo.depl_root, foo.etc_act, foo.ppt_inf))
        logging.debug(
            ('compute_crop_et(): irr_real %.6f  irr_manual %.6f  irr_special %.6f') %
            (irr_real, irr_manual, irr_special))

    # Depletion ofroot zone

    foo.depl_root += foo.etc_act - foo.ppt_inf - irr_real - irr_manual - irr_special

    # irr_real is a real irrigation experienced and read in
    # irr_manual is a manually specified irrigation from an array
    # irr_special is a special irrigation for leaching or germination

    # Determine if there is a need for an automatic irrigation

    irr_sim_prev = foo.irr_sim
    foo.irr_sim = 0.0

    if config.field_type == 'irrigated':
        irr_days = et_cell.irrigation_data[str(foo_day.year)]['irr_doys']

        # TODO setup calibration for these parameters (i.e., Kcb and NDVI-based irr_doy formulation)
        if (foo_day.doy in irr_days or foo.kc_bas > 0.7) and foo.depl_root > foo.raw:
            foo.irr_sim = foo.depl_root
            foo.irr_sim = max(foo.irr_sim, foo.irr_min)

    # Update depletion ofroot zone

    foo.depl_root -= foo.irr_sim

    # Total irrigation for today

    foo.irr_auto = foo.irr_sim
    foo.irr_sim += irr_real + irr_manual + irr_special
    if foo.irr_sim > 0:
        # Ready cumulative evaporation since last irrigation for printing
        foo.cum_evap = foo.cum_evap_prev
        foo.cum_evap_prev = 0.0

    # Deep percolation from root zone
    # Evaluate irrigation and precip for today and yesterday to actCount
    #   for temporary water storage above field capacity
    # Don't allow deep perc on rainy day or if yesterday rainy if excess < 20 mm
    #   unless zr < .2 m

    if ((foo.irr_sim + irr_sim_prev + foo.ppt_inf + foo.ppt_inf_prev) <= 0.0001 or
        foo.zr < 0.2):
        if foo.depl_root < 0.0:
            foo.dperc = -foo.depl_root
        else:
            foo.dperc = 0.0
    else:
        # Allow 20 mm above FC if watered today or yesterday
        if foo.depl_root < -20:
            foo.dperc = -20.0 - foo.depl_root
        else:
            foo.dperc = 0.0

    # Final update to depl_root (depletion of root zone)

    foo.depl_root += foo.dperc
    if debug_flag:
        logging.debug(
            'compute_crop_et(): irr_sim %.6f  deep_perc %.6f ' % (foo.irr_sim, foo.dperc))
        logging.debug(
            'compute_crop_et(): depl_root %.6f  TAW %.6f  Invoke_Stress %d' %
            (foo.depl_root, taw, crop.invoke_stress))

    # 4/16/08.  if depl_root > taw, assume it is because we have overshot E+T on this day.
    # 12/23/2011.  But don't do this ifstress flag is turned off!!
    # In that case, depl_root may be computed (incidentally) as an increasing value
    # since there is no irrigation, but no stress either
    # (i.e., wetlands, cottonwoods, etc.)  (Nuts!)

    if foo.invoke_stress > 0.5 and foo.depl_root > taw:
        # Since we overshot, then just give remaining water to etc_act

        foo.etc_act -= (foo.depl_root - taw)
        foo.etc_act = max(foo.etc_act, 0)

        # Calc new kc_act

        if foo_day.etref > 0.1:
            foo.kc_act = foo.etc_act / foo_day.etref

        # Limit depletion to total available water

        foo.depl_root = taw

    # Update average Avail. Water in soil layer below current root depth
    #   and above maximum root depth.  Add gross deep percolation to it.
    # Assume a uniform soil texture.
    # First, calculate a #'gross' deep percolation that includes 10% of irrigation depth
    #   as an incidental loss

    gross_dperc = foo.dperc + 0.1 * foo.irr_sim

    # This moisture can recharge a dry profile
    # from previous year and reduce need for irrigation.
    # This is realistic, but make sure it does not reduce any #'NIR'

    # Calc total water currently in layer 3.
    # aw3 is 0 first time through and calculated in next section

    # aw3 is mm/m and daw3 is mm in layer 3.
    # aw3 is layer between current root depth and max root

    daw3 = foo.aw3 * (foo.zr_max - foo.zr)

    # taw3 is potential mm in layer 3

    taw3 = foo.aw * (foo.zr_max - foo.zr)
    daw3 = max(daw3, 0)
    taw3 = max(taw3, 0)

    # Increase water in layer 3 for deep percolation from root zone

    daw3 += gross_dperc
    if daw3 > taw3:
        foo.dperc = daw3 - taw3

        # Limit to FC

        daw3 = taw3
    else:
        # No deep percolation from potential root zone for crop (layer 3)

        foo.dperc = 0

    daw3 = max(daw3, 0)
    if foo.zr_max > foo.zr:
        # Also defined in setup_dormant.
        # Check that zr_max does not get changed during ngs

        foo.aw3 = daw3 / (foo.zr_max - foo.zr)
    else:
        # No layer present, thus, aw3 is meaningless

        foo.aw3 = 0

    # Compute NIWR
    # NIWR = ETact – (precip – runoff – deep percolation)
    # Don't include deep percolation when irrigating
    # Irrigation ON conditional accounts for assumption that 10% of irrigation goes to
    # deep percolation.
    if foo.irr_sim > 0:
        foo.niwr = foo.etc_act - (foo_day.precip - foo.sro)
    else:
        foo.niwr = foo.etc_act - (foo_day.precip - foo.sro - foo.dperc)

    # Effective Precipitation Calcs
    # p_rz = Precipitation residing in the root zone
    # p_rz = P - Runoff - DPerc, where P is gross reported precip
    if foo.irr_sim > 0:
        foo.p_rz = foo_day.precip - foo.sro
    else:
        foo.p_rz = foo_day.precip - foo.sro - foo.dperc
    if foo.p_rz <= 0:
        foo.p_rz = 0

    # p_eft = prcp residing in the root zone available for transpiration
    # p_eft = p_rz - surface evaporation losses
    # p_eft = P - Runoff - DPerc - surface evaporation losses
    # e_ppt = ke_ppt*foo_day.etref (evap component of prcp only)

    # if foo.irr_sim > 0:
    #     foo.p_eft = foo_day.precip - foo.sro - e_ppt
    # else:
    #     foo.p_eft = foo_day.precip - foo.sro - foo.dperc - e_ppt
    # if foo.p_eft <= 0:
    #     foo.p_eft = 0

    # Modified to use e for surface evaporation losses instead of e_ppt (11/2/2020)
    if foo.irr_sim > 0:
        foo.p_eft = foo_day.precip - foo.sro - e
    else:
        foo.p_eft = foo_day.precip - foo.sro - foo.dperc - e
    if foo.p_eft <= 0:
        foo.p_eft = 0

    # Note, at end of season (harvest or death), aw3 and zr need to be reset
    #   according to depl_root at that time and zr for dormant season.
    # This is done in setup_dormant().

    # Get setup for next time step.
    # if foo.in_season:
    grow_root.grow_root(foo, foo_day, debug_flag)
