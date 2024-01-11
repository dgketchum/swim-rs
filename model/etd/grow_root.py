"""grow_root.py
Defines grow_root function
Called by compute_crop_et.py

"""

import logging
import math
import datetime

import numpy as np


def grow_root(foo, foo_day, debug_flag=False):
    """Determine depth of root zone
    Parameters
    ----------
    crop :

    foo :

    debug_flag : boolean
        True : write debug level comments to debug.txt
        False

    Returns
    -------
    None

    Notes
    -----
    Uses Borg and Grimes (1986) sigmoidal function [doi: 10.13031/2013.30125]

    """

    # dlk - 10/31/2011 - added zero value tests
    # fractime = 0
    # if crop.curve_type == 1 and crop.end_of_root_growth_fraction_time != 0.0:
    #     fractime = foo.n_cgdd / crop.end_of_root_growth_fraction_time
    # elif crop.curve_type > 1 and crop.end_of_root_growth_fraction_time != 0.0:
    #     fractime = foo.n_pl_ec / crop.end_of_root_growth_fraction_time
    # fractime = min(max(fractime, 0), 1)

    # dgketchum hacks for fractime function
    # idea would be to use ETf or NDVI thresholds to integrate growth over time
    # if foo.grow_root:
    #     if foo_day.year in cell.fallow_years or data.field_type == 'unirrigated':
    #         kc_src = '{}_NO_IRR'.format(data.kc_proxy)
    #     else:
    #         kc_src = '{}_IRR'.format(data.kc_proxy)
    #
    #     gs_start = datetime.datetime(foo_day.year, 4, 1)
    #     etf_diff = cell.crop_coeffs[1].data.loc[gs_start: foo_day.date, kc_src].diff()
    #     etf_diff[etf_diff < 0] = np.nan
    #     etf_cumm = etf_diff.sum()

    zr_prev = foo.zr

    gs_start, gs_end = datetime.datetime(foo_day.year, 4, 1), datetime.datetime(foo_day.year, 10, 31)
    gs_start_doy, gs_end_doy = int(gs_start.strftime('%j')), int(gs_end.strftime('%j'))
    fractime = (foo_day.doy - gs_start_doy) / (gs_end_doy - gs_start_doy)

    if 1.0 > fractime > 0.0:
        # Borg and Grimes (1986) sigmoidal function
        foo.zr = (
            (0.5 + 0.5 * math.sin(3.03 * fractime - 1.47)) *
            (foo.zr_max - foo.zr_min) + foo.zr_min)

    elif foo_day.doy < gs_start_doy or foo_day.doy > gs_end_doy:
        foo.zr = foo.zr_min

    delta_zr = foo.zr - zr_prev

    # update depl_root for new moisture coming in bottom of root zone
    # depl_root (depletion) will increase if new portion of root zone is < FC
    if delta_zr > 0:
        # AM3 is mean moisture of maxrootzone - Zr layer
        foo.depl_root += delta_zr * (foo.aw - foo.aw3)

    # Also keep zr from #'shrinking' (for example, with multiple alfalfa cycles

    if debug_flag:
        logging.debug(
            ('grow_root(): zr %.6f  fractime %.6f  zr_max %.6f  zr_min %.6f  depl_root %.6f') %
            (foo.zr, fractime, foo.zr_max, foo.zr_min, foo.depl_root))
        logging.debug(
            ('grow_root(): delta_zr %s  AW %.6f  AW3 %.6f') %
            (delta_zr, foo.aw, foo.aw3))
        logging.debug(
            'grow_root(): n_cgdd %.6f  n_pl_ec %s' % (foo.n_cgdd, foo.n_pl_ec))
        logging.debug(
            ('grow_root(): end_of_root %s  crop_curve_type %s') %
            (crop.end_of_root_growth_fraction_time, crop.curve_type))
