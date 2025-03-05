"""kcb_daily.py
Defines kcb_daily function
Called by crop_cycle.py

"""

import datetime
import logging

import numpy as np

def kcb_daily(tracker, day_data):
    """"""

    tracker.kc_bas = day_data.ndvi * tracker.ndvi_beta + tracker.ndvi_alpha
    tracker.kc_bas = np.minimum(tracker.kc_max, tracker.kc_bas)

    # Water has only 'kcb'
    # TODO: calculate for water
    # foo.kc_act = foo.kc_bas
    # foo.kc_pot = foo.kc_bas

    # ETr changed to ETref 12/26/2007
    tracker.etc_act = tracker.kc_act * day_data.refet
    tracker.etc_pot = tracker.kc_pot * day_data.refet
    tracker.etc_bas = tracker.kc_bas * day_data.refet

