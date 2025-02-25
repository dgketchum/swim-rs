import numpy as np
import pandas as pd

from model.etd import calculate_height
from model.etd import compute_field_et
from model.etd import obs_kcb_daily
from model.etd.tracker import SampleTracker
from model.etd.day_data import DayData

OUTPUT_FMT = ['et_act',
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
              'season',
              'capture',
              ]


class WaterBalanceError(Exception):
    pass


def field_day_loop(config, plots, debug_flag=False, params=None):
    """"""
    etf, swe = None, None
    size = len(plots.input['order'])

    tracker = SampleTracker(size)
    tracker.load_root_depth(plots)
    tracker.load_soils(plots)
    tracker.apply_parameters(config, plots, params=params)

    targets = plots.input['order']

    valid_data = {dt: val for dt, val in plots.input['time_series'].items() if
                  (config.start_dt <= pd.to_datetime(dt) <= config.end_dt)}

    if debug_flag:
        tracker.setup_dataframe(targets)
    else:
        time_range = pd.date_range(config.start_dt, config.end_dt, freq='D')
        empty = np.zeros((len(time_range), len(targets))) * np.nan
        etf, swe = empty.copy(), empty.copy()

    tracker.set_kc_max()

    day_data = DayData()

    for j, (step_dt, vals) in enumerate(valid_data.items()):

        day_data.update_day(step_dt, size, vals['doy'])

        if day_data.doy == 1 or day_data.irr_status is None:
            day_data.update_annual_irrigation(plots)
            day_data.update_annual_groundwater_subsidy(plots)

        day_data.update_daily_irrigation(plots, vals, config)

        day_data.update_daily_inputs(vals, size)

        calculate_height.calculate_height(tracker)

        obs_kcb_daily.kcb_daily(tracker, day_data)

        compute_field_et.compute_field_et(plots, tracker, day_data)

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
