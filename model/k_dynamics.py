import numpy as np


def ke_damper(swb):
    """Compute the soil evaporation coefficient Ke with a damping scheme.

    Damps day-to-day changes in the surface soil evaporation reduction factor
    (Kr) using `kr_alpha`, then caps Ke by `few * kc_max` and `ke_max`.
    """
    kr_current = np.clip((swb.tew - swb.depl_ze) / (swb.tew - swb.rew + 1e-6), 0.0, 1.0)

    if not hasattr(swb, 'kr_prev'):
        swb.kr_prev = kr_current

    kr_change = kr_current - swb.kr_prev
    damped_kr_change = kr_change * swb.kr_alpha
    swb.kr = np.clip(swb.kr_prev + damped_kr_change, 0.0, 1.0)
    swb.kr_prev = swb.kr

    swb.ke = np.minimum(swb.kr * (swb.kc_max - swb.kc_bas), swb.few * swb.kc_max)
    swb.ke = np.clip(swb.ke, 0.0, swb.ke_max)


def ke_momentum(swb):
    """Compute Ke using a momentum-based update of the Kr term."""
    kr_current = np.clip((swb.tew - swb.depl_ze) / (swb.tew - swb.rew + 1e-6), 0.0, 1.0)

    if not hasattr(swb, 'kr_prev_delta'):
        swb.kr_prev = kr_current
        swb.kr_prev_delta = 0.0

    kr_momentum = swb.kr_prev + swb.kr_prev_delta
    kr_momentum = np.clip(kr_momentum, 0.0, 1.0)
    swb.kr = swb.kr_alpha * kr_momentum + (1 - swb.kr_alpha) * kr_current

    swb.kr_prev_delta = swb.kr - swb.kr_prev
    swb.kr_prev = swb.kr

    swb.ke = np.minimum(swb.kr * (swb.kc_max - swb.kc_bas), swb.few * swb.kc_max)
    swb.ke = np.clip(swb.ke, 0.0, swb.ke_max)


def ke_exponential(swb):
    """Compute Ke using asymmetric exponential rise/decay parameters."""
    kr_current = np.clip((swb.tew - swb.depl_ze) / (swb.tew - swb.rew + 1e-6), 0.0, 1.0)

    if kr_current < swb.kr_prev:
        kr = kr_current + (swb.kr_prev - kr_current) * np.exp(-swb.kr_down * swb.kr_prev)
    else:
        kr_change = kr_current - swb.kr_prev
        damped_kr_change = kr_change * swb.kr_up
        kr = np.clip(swb.kr_prev + damped_kr_change, 0.0, 1.0)

    swb.kr = np.clip(kr, 0.0, 1.0)
    swb.kr_prev = swb.kr

    swb.ke = np.minimum(swb.kr * (swb.kc_max - swb.kc_bas), swb.few * swb.kc_max)
    swb.ke = np.clip(swb.ke, 0.0, swb.ke_max)


def ks_damper(swb):
    """Compute root-zone stress coefficient Ks with damping on changes.

    Applies a linear damping factor `ks_alpha` to transitions in Ks, with Ks
    defined as a function of depletion (`depl_root`), TAW, and RAW.
    """
    ks_current = np.where(swb.depl_root > swb.raw,
                          np.clip((swb.taw - swb.depl_root) / (swb.taw - swb.raw), 0.0, 1.0), 1)

    if not hasattr(swb, 'ks_prev'):
        swb.ks_prev = ks_current

    ks_change = ks_current - swb.ks_prev
    damped_ks_change = ks_change * swb.ks_alpha
    swb.ks = np.clip(swb.ks_prev + damped_ks_change, 0.0, 1.0)
    swb.ks_prev = swb.ks


def ks_momentum(swb):
    """Compute Ks using a momentum-based update of prior-day changes."""
    ks_current = np.where(swb.depl_root > swb.raw,
                          np.clip((swb.taw - swb.depl_root) / (swb.taw - swb.raw), 0.0, 1.0), 1)

    if not hasattr(swb, 'ks_prev_delta'):
        swb.ks_prev = ks_current
        swb.ks_prev_delta = 0.0

    ks_momentum_ = swb.ks_prev + swb.ks_prev_delta
    ks_momentum_ = np.clip(ks_momentum_, 0.0, 1.0)
    swb.ks = swb.ks_alpha * ks_momentum_ + (1 - swb.ks_alpha) * ks_current
    swb.ks_prev_delta = swb.ks - swb.ks_prev
    swb.ks_prev = swb.ks


def ks_exponential(swb):
    """Compute Ks with an exponential decay relative to prior state."""
    ks_current = np.where(swb.depl_root > swb.raw,
                          np.clip((swb.taw - swb.depl_root) / (swb.taw - swb.raw), 0.0, 1.0), 1)

    if not hasattr(swb, 'ks_prev'):
        swb.ks_prev = ks_current

    swb.ks = ks_current * np.exp(-swb.ks_alpha * swb.ks_prev)
    swb.ks = np.clip(swb.ks, 0.0, 1.0)
    swb.ks_prev = swb.ks
