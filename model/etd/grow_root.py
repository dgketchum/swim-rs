import datetime

import math
import numpy as np


def grow_root(tracker, day_data):
    """"""

    zr_prev = tracker.zr.copy()

    gs_start, gs_end = datetime.datetime(day_data.year, 4, 1), datetime.datetime(day_data.year, 10, 31)
    gs_start_doy, gs_end_doy = int(gs_start.strftime('%j')), int(gs_end.strftime('%j'))
    fractime = (day_data.doy - gs_start_doy) / (gs_end_doy - gs_start_doy)

    zr_new = None
    if fractime < 0.0:
        zr_new = tracker.zr_min

    elif 0.0 <= fractime <= 1.0:
        # Borg and Grimes (1986) sigmoidal function
        zr_new = (
            (0.5 + 0.5 * math.sin(3.03 * fractime - 1.47)) *
            (tracker.zr_max - tracker.zr_min) + tracker.zr_min)

    elif fractime > 1.0:
        zr_new = tracker.zr_min

    # as root zone grows or retracts, transfer the water in that column from root zone (depl_root) to/from
    # sub-root zone (daw3)

    tracker.zr = np.where(tracker.perennial, zr_prev, zr_new)
    delta_zr = tracker.zr - zr_prev

    tracker.taw3 = tracker.aw * (tracker.zr_max - tracker.zr)

    prev_rt_water = ((tracker.aw * zr_prev) - tracker.depl_root.copy())

    tracker.depl_root = np.where(delta_zr > 0.0,
                                 tracker.depl_root + (delta_zr * tracker.aw) - (
                                         tracker.daw3 * delta_zr / (tracker.zr_max - tracker.zr + 1e-6)),
                                 tracker.depl_root * (1 - (-1.0 * delta_zr / zr_prev)))

    if np.any(np.isnan(tracker.depl_root)):
        a = 1

    rt_water = (tracker.aw * tracker.zr) - tracker.depl_root.copy()

    prev_daw3 = tracker.daw3.copy()

    tracker.daw3 = np.where(delta_zr > 0.0,
                            tracker.daw3 - (tracker.daw3 * delta_zr / (tracker.zr_max - tracker.zr + 1e-6)),
                            tracker.daw3 + prev_rt_water - rt_water)

    if np.any(tracker.daw3 - tracker.taw3 > 0.0001):
        raise NotImplementedError

    delta_daw3 = tracker.daw3 - prev_daw3
    delta_rt_water = rt_water - prev_rt_water
    distribution = delta_rt_water + delta_daw3
    if np.any(distribution > 0.0001):
        raise NotImplementedError
