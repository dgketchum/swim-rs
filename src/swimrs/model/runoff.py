"""Runoff formulations for the daily water balance.

Includes two alternatives:
- Infiltration-excess runoff using hourly precipitation vs. infiltration rate.
- SCS Curve Number method with dynamic antecedent moisture condition.
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)


def runoff_infiltration_excess(foo, foo_day):
    """Compute runoff as the sum over hours of (ppt - infiltration capacity).

    Parameters
    - foo: SampleTracker with `ksat_hourly` infiltration capacity (mm/h).
    - foo_day: DayData with `hr_precip` 24xN hourly precipitation (mm/h).
    """
    foo.sro = (
        np.maximum(foo_day.hr_precip - foo.ksat_hourly, np.zeros_like(foo_day.hr_precip))
        .sum(axis=0)
        .reshape(1, -1)
    )


def runoff_curve_number(foo, foo_day):
    """Compute daily runoff with SCS Curve Number (CN) method.

    Adjusts CN for antecedent moisture using surface depletion and smooths the
    potential maximum retention S over recent days when irrigated.
    """
    # CNII from crop/soil combination
    CNII = np.minimum(np.maximum(foo.cn2, 10.0), 100.0)

    # CN for other antecedent conditions (Hawkins et al., 1985)
    CNI = CNII / (2.281 - 0.01281 * CNII)
    CNIII = CNII / (0.427 + 0.00573 * CNII)

    # Antecedent moisture thresholds
    AWCIII = 0.5 * foo.rew
    AWCI = 0.7 * foo.rew + 0.3 * foo.tew

    # Ensure AWCI > AWCIII
    AWCI = np.where(AWCI <= AWCIII, AWCIII + 0.01, AWCI)

    # CN adjusted for surface depletion
    ds = foo.depl_surface  # likely maps to depl_ze in this model  # FIXME review mapping
    cn = np.where(
        ds < AWCIII,
        CNIII,
        np.where(ds > AWCI, CNI, ((ds - AWCIII) * CNI + (AWCI - ds) * CNIII) / (AWCI - AWCIII)),
    )

    # Potential maximum retention S (mm)
    foo.s = 250.0 * (100.0 / cn - 1.0)

    # Irrigated smoothing branch (average prior four S states)
    if np.any(foo.irr_flag):
        ppt_net4 = np.maximum(foo_day.precip - 0.2 * foo.s4, 0.0)
        ppt_net3 = np.maximum(foo_day.precip - 0.2 * foo.s3, 0.0)
        ppt_net2 = np.maximum(foo_day.precip - 0.2 * foo.s2, 0.0)
        ppt_net1 = np.maximum(foo_day.precip - 0.2 * foo.s1, 0.0)
        foo.sro = 0.25 * (
            (ppt_net4 * ppt_net4) / (foo_day.precip + 0.8 * foo.s4)
            + (ppt_net3 * ppt_net3) / (foo_day.precip + 0.8 * foo.s3)
            + (ppt_net2 * ppt_net2) / (foo_day.precip + 0.8 * foo.s2)
            + (ppt_net1 * ppt_net1) / (foo_day.precip + 0.8 * foo.s1)
        )
        foo.s4 = foo.s3
        foo.s3 = foo.s2
        foo.s2 = foo.s1
        foo.s1 = foo.s
    else:
        ppt_net = np.maximum(foo_day.precip - 0.2 * foo.s, 0.0)
        foo.sro = (ppt_net * ppt_net) / (foo_day.precip + 0.8 * foo.s)
