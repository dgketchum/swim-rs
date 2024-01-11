"""calculate_height.py
Function for calculating height of crop based on Kc and height limit
Called by crop_cycle.py

"""

import logging


def calculate_height(foo):

    height_prev = foo.height

    # <----- previous (2000) and with error (Kcbmin vs Kcmin)
    # height = height_min + (kc_bas - kcb_min) / (kc_bas_mid - kc_min) * (height_max - height_min)
    # kc_bas_mid is maximum kc_bas found in kc_bas table read into program

    # Following conditionals added 12/26/07 to prevent any error
    if foo.kc_bas > foo.kc_min and foo.kc_bas_mid > foo.kc_min:
        foo.height = (
            foo.height_initial + (foo.kc_bas - foo.kc_min) / (foo.kc_bas_mid - foo.kc_min) *
            (foo.height_max - foo.height_initial))
    else:
        foo.height = foo.height_initial
    foo.height = min(max(foo.height_initial, max(height_prev, foo.height)), foo.height_max)

