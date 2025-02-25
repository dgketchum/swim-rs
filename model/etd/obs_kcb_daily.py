"""kcb_daily.py
Defines kcb_daily function
Called by crop_cycle.py

"""

import datetime
import logging

import numpy as np

def kcb_daily(foo, foo_day):
    """Compute basal ET

    Parameters
    ---------
        config :

        et_cell :
        crop :
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

    """


    foo.kc_bas = foo_day.ndvi * foo.ndvi_beta + foo.ndvi_alpha
    foo.kc_bas = np.minimum(foo.kc_max, foo.kc_bas)

    # Water has only 'kcb'
    # TODO: calculate for water
    # foo.kc_act = foo.kc_bas
    # foo.kc_pot = foo.kc_bas

    # ETr changed to ETref 12/26/2007
    foo.etc_act = foo.kc_act * foo_day.refet
    foo.etc_pot = foo.kc_pot * foo_day.refet
    foo.etc_bas = foo.kc_bas * foo_day.refet

