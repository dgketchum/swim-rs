"""runoff.py
called by compute_crop_et.py

"""
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def runoff_infiltration_excess(foo, foo_day):
    foo.sro = np.maximum(foo_day.hr_precip - foo.ksat_hourly,
                         np.zeros_like(foo_day.hr_precip)).sum(axis=0).reshape(1, -1)
