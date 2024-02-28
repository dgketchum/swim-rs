import os

import numpy as np


def calculate_snow(foo, foo_day):
    temp = foo_day.temp_avg
    palb = foo.albedo

    precip = foo_day.precip

    a_min = 0.45
    a_max = 0.98

    # The threshold values here were 0.0 and were changed to 4.0, consider calibrating this
    sf = np.where(temp < 4.0, precip, 0)
    rain = np.where(temp >= 4.0, precip, 0)

    alb = np.where(sf > 3.0, a_max, palb)
    alb = np.where(sf <= 3.0, a_min + (palb - a_min) * np.exp(-0.12), alb)
    alb = np.where(sf == 0.0, a_min + (palb - a_min) * np.exp(-0.05), alb)
    alb = np.where(alb < a_min, a_min, alb)

    foo.swe += sf

    melt = np.maximum(((1 - alb) * foo_day.rg * foo.snow_alpha) + (temp - 1.8) * foo.snow_beta, 0)

    foo.melt = melt = np.minimum(foo.swe, melt)
    foo.swe -= melt

    foo.rain = rain
    foo.snow_fall = sf
    foo.albedo = alb


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
