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
        foo.zr *= 0.9

    delta_zr = foo.zr - zr_prev

    prev_taw = foo.taw.copy()

    foo.taw = foo.aw * foo.zr
    foo.taw = np.maximum(foo.taw, 0.001)

    delta_taw = foo.taw - prev_taw

    # update depl_root for new moisture coming in bottom of root zone
    # depl_root (depletion) will increase if new portion of root zone is < FC

    prev_depl_root = foo.depl_root.copy()

    foo.depl_root = np.where(delta_zr > 0,
                             foo.depl_root + delta_zr / (foo.zr_max - foo.zr) * foo.daw3,
                             foo.depl_root)

    foo.depl_root = np.where(delta_zr < 0,
                             foo.depl_root - delta_zr / (foo.zr_max - foo.zr) * foo.daw3,
                             foo.depl_root)

    foo.depl_root = np.where(delta_taw < 0, foo.taw, foo.depl_root)

    foo.daw3 = np.where(delta_taw < 0,
                        foo.daw3 + delta_zr / foo.zr * (foo.taw - foo.depl_root),
                        foo.daw3 - delta_zr / foo.zr * (foo.taw - foo.depl_root))

