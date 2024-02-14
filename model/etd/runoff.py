"""runoff.py
Defines function runoff for computing SCS curve number runoff
called by compute_crop_et.py

"""

import logging

import numpy as np


def runoff_curve_number(foo, foo_day, config, debug_flag=False):
    """Curve number method for computing runoff
    Attributes
    ---------
    foo :

    foo_day :

    debug_flag : boolean
        True : write debug level comments to debug.txt
        False

    Returns
    -------
    None

    Notes
    -----
    ADD REFS!

    """

    # logging.debug('runoff()')

    # Bring in CNII for antecedent condition II from crop-soil combination
    # Check to insure CNII is within limits
    CNII = np.minimum(np.maximum(foo.cn2, np.ones_like(foo.cn2) * 10), np.ones_like(foo.cn2) * 100)

    # Compute CN's for other antecedent conditions
    # Hawkins et al., 1985, ASCE Irr.Drn. 11(4):330-340
    CNI = CNII / (2.281 - 0.01281 * CNII)
    CNIII = CNII / (0.427 + 0.00573 * CNII)

    # Determine antecedent condition
    # Presume that AWCIII is quite moist (when only 1/2 of REW is evaporated)
    # Be sure that REW and TEW are shared
    AWCIII = 0.5 * foo.rew

    # Presume that dry AWCI condition occurs somewhere between REW and TEW
    AWCI = 0.7 * foo.rew + 0.3 * foo.tew

    # Value for CN adjusted for soil moisture
    # Make sure AWCI>AWCIII
    AWCI = np.where(AWCI <= AWCIII, AWCIII + 0.01, AWCI)

    cn = np.where(foo.depl_surface < AWCIII, CNIII,
                  np.where(foo.depl_surface > AWCI, CNI,
                           ((foo.depl_surface - AWCIII) * CNI +
                            (AWCI - foo.depl_surface) * CNIII) / (AWCI - AWCIII)))

    foo.s = 250 * (100 / cn - 1)

    # If irrigations are automatically scheduled, base runoff on an average of
    #   conditions for prior four days to smooth results.

    if config.field_type == 'irrigated':
        # Initial abstraction
        ppt_net4 = np.maximum(foo_day.precip - 0.2 * foo.s4, 0)
        ppt_net3 = np.maximum(foo_day.precip - 0.2 * foo.s3, 0)
        ppt_net2 = np.maximum(foo_day.precip - 0.2 * foo.s2, 0)
        ppt_net1 = np.maximum(foo_day.precip - 0.2 * foo.s1, 0)
        foo.sro = 0.25 * (
            ppt_net4 ** 2 / (foo_day.precip + 0.8 * foo.s4) +
            ppt_net3 ** 2 / (foo_day.precip + 0.8 * foo.s3) +
            ppt_net2 ** 2 / (foo_day.precip + 0.8 * foo.s2) +
            ppt_net1 ** 2 / (foo_day.precip + 0.8 * foo.s1))

        foo.s4 = foo.s3
        foo.s3 = foo.s2
        foo.s2 = foo.s1
        foo.s1 = foo.s
    else:
        # Non-irrigated runoff
        # Initial abstraction
        ppt_net = np.maximum(foo_day.precip - 0.2 * foo.s, 0)
        foo.sro = ppt_net * ppt_net / (foo_day.precip + 0.8 * foo.s)


def runoff_infiltration_excess(foo, foo_day, debug_flag=False):

    foo.sro = np.maximum(foo_day.hr_precip - foo.ksat_hourly,
                         np.zeros_like(foo_day.hr_precip)).sum(axis=0).reshape(1, -1)

