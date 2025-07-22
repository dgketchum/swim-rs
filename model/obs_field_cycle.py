from pprint import pprint

import numpy as np
import pandas as pd

from model import compute_field_et
from model import obs_kcb_daily
from model.tracker import SampleTracker, TUNABLE_PARAMS
from model.day_data import DayData

OUTPUT_FMT = ['aw',
              'et_act',
              'etref',
              'kc_act',
              'kc_bas',
              'ks',
              'ke',
              'melt',
              'rain',
              'depl_root',
              'depl_ze',
              'dperc',
              'runoff',
              'delta_soil_water',
              'wbal',
              'ppt',
              'snow_fall',
              'taw',
              'taw3',
              'daw3',
              'delta_daw3',
              'swe',
              'tavg',
              'tmax',
              'irrigation',
              'gw_sim',
              'fc',
              't',
              'e',
              'few',
              'zr',
              'p_rz',
              'p_eft',
              'soil_water',
              'niwr',
              'irr_day',
              ]


class WaterBalanceError(Exception):
    pass


def field_day_loop(config, plots, debug_flag=False, params=None):
    """"""
    etf, swe = None, None
    size = len(plots.input['order'])

    tracker = SampleTracker(config, plots, size)
    tracker.apply_initial_conditions()
    tracker.apply_parameters(params=params)
    tracker.load_root_depth()
    tracker.load_soils()

    # only set kc/ke max if they were not calibrated
    tracker.set_ke_max()
    # tracker.__setattr__('ndvi_k', np.array([[6.63]]))
    # tracker.__setattr__('ndvi_0', np.array([[0.28]]))

    if debug_flag:
        tunable_state = {k: tracker.__getattribute__(k) for k in TUNABLE_PARAMS}
        if size == 1:
            tunable_state = {k: f'{v[0, 0]:.2f}' for k, v in tunable_state.items()}
        else:
            tunable_state = {k: [f'{vv:.2f}' for vv in v.flatten()] for k, v in tunable_state.items()}
        pprint(dict(sorted(tunable_state.items())))

    targets = plots.input['order']

    if len(pd.date_range(config.start_dt, config.end_dt, freq='D')) == len(plots.input['time_series']):
        valid_data = plots.input['time_series'].copy()
    else:
        valid_data = {dt: val for dt, val in plots.input['time_series'].items() if
                      (config.start_dt <= pd.to_datetime(dt) <= config.end_dt)}

    if debug_flag:
        tracker.setup_dataframe(targets)
    else:
        time_range = pd.date_range(config.start_dt, config.end_dt, freq='D')
        empty = np.zeros((len(time_range), len(targets))) * np.nan
        etf, swe = empty.copy(), empty.copy()

    day_data = DayData()

    for j, (step_dt, vals) in enumerate(valid_data.items()):

        day_data.update_day(step_dt, size, vals['doy'])

        if day_data.doy == 1 or day_data.irr_status is None:
            day_data.update_annual_irrigation(plots)
            day_data.update_annual_groundwater_subsidy(plots)

        day_data.update_daily_irrigation(plots, vals, config)

        day_data.update_daily_inputs(vals, size)

        obs_kcb_daily.kcb_daily(tracker, day_data)

        compute_field_et.compute_field_et(tracker, day_data)

        if debug_flag:
            tracker.update_dataframe(targets, day_data, step_dt)

        else:
            if np.any(np.isnan(tracker.kc_act)):
                raise ValueError('NaN in Kc_act')

            if np.any(np.isnan(tracker.swe)):
                raise ValueError('NaN in SWE')

            etf[j, :] = tracker.kc_act
            swe[j, :] = tracker.swe

    if debug_flag:
        # pass final dataframe to calling script
        tracker.crop_df = {fid: pd.DataFrame().from_dict(tracker.crop_df[fid], orient='index')[OUTPUT_FMT]
                           for fid in targets}

        for fid in tracker.crop_df:
            tracker.crop_df[fid].index = pd.to_datetime(tracker.crop_df[fid].index)

        return tracker.crop_df

    else:
        # if not debug, just return the actual ET and SWE results as ndarray
        return etf, swe


if __name__ == '__main__':
    pass
