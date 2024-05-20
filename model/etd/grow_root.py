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

    zr_prev = foo.zr.copy()

    gs_start, gs_end = datetime.datetime(foo_day.year, 4, 1), datetime.datetime(foo_day.year, 10, 31)
    gs_start_doy, gs_end_doy = int(gs_start.strftime('%j')), int(gs_end.strftime('%j'))
    fractime = (foo_day.doy - gs_start_doy) / (gs_end_doy - gs_start_doy)

    if fractime < 0.0:
        foo.zr = foo.zr_min

    elif 0.0 <= fractime <= 1.0:
        # Borg and Grimes (1986) sigmoidal function
        foo.zr = (
            (0.5 + 0.5 * math.sin(3.03 * fractime - 1.47)) *
            (foo.zr_max - foo.zr_min) + foo.zr_min)

    elif fractime > 1.0:
        foo.zr = foo.zr_min

    # as root zone grows or retracts, transfer the water in that column from root zone (depl_root) to/from
    # sub-root zone (daw3)

    delta_zr = foo.zr - zr_prev

    foo.taw3 = foo.aw * (foo.zr_max - foo.zr)

    prev_rt_water = ((foo.aw * zr_prev) - foo.depl_root.copy())

    foo.depl_root = np.where(delta_zr >= 0.0,
                             foo.depl_root + (delta_zr * foo.aw) - (foo.daw3 * delta_zr / (foo.zr_max - foo.zr)),
                             foo.depl_root * (1 - (-1.0 * delta_zr / zr_prev)))

    rt_water = (foo.aw * foo.zr) - foo.depl_root.copy()

    prev_daw3 = foo.daw3.copy()

    foo.daw3 = np.where(delta_zr >= 0.0,
                        foo.daw3 - (foo.daw3 * delta_zr / (foo.zr_max - foo.zr)),
                        foo.daw3 + prev_rt_water - rt_water)

    if foo.daw3 - foo.taw3 > 0.0001:
        raise NotImplementedError

    delta_daw3 = foo.daw3 - prev_daw3
    delta_rt_water = rt_water - prev_rt_water
    distribution = delta_rt_water + delta_daw3
    if distribution.sum().item() > 0.0001:
        raise NotImplementedError
