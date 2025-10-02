import numpy as np


def grow_root(tracker, day_data):
    """Update effective root depth and re-partition soil water between layers.

    Root depth is tied to basal crop coefficient (NDVI-based) and capped by
    `zr_max`. The routine conserves mass between the active root zone and the
    deeper layer (daw3/taw3) as roots grow or recede.

    Parameters
    - tracker: SampleTracker with state arrays (zr, daw3, taw3, aw, depl_root).
    - day_data: DayData (unused except for error reporting on rare failures).
    """

    zr_prev = tracker.zr.copy()

    zr_new = (tracker.kc_bas - tracker.kc_min) / (tracker.kc_max - tracker.kc_min) * tracker.zr_max
    zr_new = np.clip(zr_new, np.zeros_like(tracker.zr_max), tracker.zr_max)

    tracker.zr = np.where(tracker.perennial, zr_prev, zr_new)
    delta_zr = tracker.zr - zr_prev

    tracker.taw3 = tracker.aw * (tracker.zr_max - tracker.zr)

    prev_rt_water = ((tracker.aw * zr_prev) - tracker.depl_root.copy())

    tracker.depl_root = np.where(delta_zr > 0.0,
                                 tracker.depl_root + (delta_zr * tracker.aw) - (
                                     tracker.daw3 * delta_zr / (tracker.zr_max - tracker.zr + 1e-6)),
                                 tracker.depl_root * (1 - (-1.0 * delta_zr / zr_prev)))

    rt_water = (tracker.aw * tracker.zr) - tracker.depl_root.copy()

    prev_daw3 = tracker.daw3.copy()

    tracker.daw3 = np.where(delta_zr > 0.0,
                            tracker.daw3 - (tracker.daw3 * delta_zr / (tracker.zr_max - tracker.zr + 1e-6)),
                            tracker.daw3 + prev_rt_water - rt_water)

    if np.any(tracker.daw3 - tracker.taw3 > 0.0001):
        print(day_data.dt_string)
        raise NotImplementedError(f'daw3/taw3 mass failure: zr_prev: {zr_prev[0, 0]:.4f}, zr_new {zr_new[0, 0]:.4f}')

    delta_daw3 = tracker.daw3 - prev_daw3
    delta_rt_water = rt_water - prev_rt_water
    distribution = delta_rt_water + delta_daw3
    if np.any(distribution > 0.0001):
        raise NotImplementedError
