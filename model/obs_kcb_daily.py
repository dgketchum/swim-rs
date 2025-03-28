import numpy as np


def kcb_daily(tracker, day_data):
    """"""

    # linear
    # tracker.kc_bas = day_data.ndvi * tracker.ndvi_beta + tracker.ndvi_alpha
    # sigmoid
    tracker.kc_bas = tracker.kc_max / (1 + np.exp(-tracker.ndvi_k * (day_data.ndvi - tracker.ndvi_0)))
    tracker.kc_bas = np.minimum(tracker.kc_max, tracker.kc_bas)
