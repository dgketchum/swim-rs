"""Model package constants and parameter lists used by the tracker.

`TRACKER_PARAMS` enumerates the state/parameter fields that are array-shaped
and managed by `SampleTracker` for vectorized, per-field simulation.
"""
import os


TRACKER_PARAMS = ['taw',
                  'taw3',
                  'rew',
                  'tew',
                  'melt',
                  'rain',
                  'snow_fall',
                  'daw3',
                  'daw3_prev',
                  'aw3',
                  'aw',
                  'cum_evap',
                  'cum_evap_prev',
                  'depl_ze',
                  'dperc',
                  'depl_surface',
                  'depl_root',
                  'depl_root_prev',
                  'soil_water',
                  'soil_water_prev',
                  'etc_act',
                  'fc',
                  'gw_sim',
                  'irr_sim',
                  'kc_act',
                  'kc_max',
                  'kc_min',
                  'kc_bas',
                  'kc_bas_mid',
                  'ke',
                  'ke_max',
                  'kr',
                  'kr_prev',
                  'ks',
                  'ks_prev',
                  'mad',
                  'max_irr_rate',
                  'next_day_irr',
                  'niwr',
                  'p_rz',
                  'p_eft',
                  'ppt_inf',
                  'ppt_inf_prev',
                  'sro',
                  's',
                  's1',
                  's2',
                  's3',
                  's4',
                  'swe',
                  'zr',
                  'zr_mult',
                  'zr_min',
                  'zr_max',
                  'totwatin_ze',
                  'wt_irr',
                  'irr_min']


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
