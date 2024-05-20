"""crop_cycle.py
Defines DayData class
Defines crop_cycle_mp, crop_cycle, crop_day_loop_mp, crop_day_loop,
    write_crop_output
Called by mod_crop_et.py

"""

import numpy as np
import pandas as pd

from model.etd import calculate_height
from model.etd import compute_field_et
from model.etd import obs_kcb_daily
from model.etd.initialize_tracker import PlotTracker

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
              'taw',
              'taw3',
              'daw3',
              'delta_daw3',
              'wbal',
              'ppt',
              'snow_fall',
              'swe',
              'tavg',
              'tmax',
              'irrigation',
              'fc',
              't',
              'e',
              'few',
              'zr',
              'p_rz',
              'p_eft',
              'niwr',
              'irr_day',
              'season',
              'capture',
              ]

DEFAULTS = {'ndvi_beta': 1.35,
            'ndvi_alpha': -0.44,
            'mad': 1.0,
            'swe_alpha': 0.073,
            'swe_beta': 1.38}


class DayData:

    def __init__(self):
        self.etref_array = np.zeros(30)


class WaterBalanceError(Exception):
    pass


def field_day_loop(config, plots, debug_flag=False, params=None):
    eta, swe = None, None
    size = len(plots.input['order'])
    tracker = PlotTracker(size)
    tracker.load_soils(plots)

    # apply calibration parameter updates here
    if config.calibration:
        # PEST++ hacking
        cal_arr = {k: np.zeros((1, size)) for k in config.calibration_groups}

        for k, f in config.calibration_files.items():

            group, fid = '_'.join(k.split('_')[:-1]), k.split('_')[-1]
            idx = plots.input['order'].index(fid)

            if params:
                value = params[k]
            else:
                v = pd.read_csv(f, index_col=None, header=0)
                value = v.loc[0, '1']

            cal_arr[group][0, idx] = value

        for k, v in cal_arr.items():

            tracker.__setattr__(k, v)

            if debug_flag:
                print('{}: {}'.format(k, ['{:.2f}'.format(p) for p in v.flatten()]))


    else:
        for k, v in DEFAULTS.items():
            arr = np.ones((1, size)) * v
            tracker.__setattr__(k, arr)

    print('AW: {:.3f}'.format(tracker.aw.mean()))
    targets = plots.input['order']

    # Initialize crop data frame
    if debug_flag:
        tracker.setup_dataframe(targets)
    else:
        empty = np.zeros((len(plots.input['time_series']), len(targets))) * np.nan
        eta, swe = empty.copy(), empty.copy()

    tracker.set_kc_max()

    foo_day = DayData()
    foo_day.sdays = 0
    foo_day.doy_prev = 0

    hr_ppt_keys = ['prcp_hr_{}'.format(str(i).rjust(2, '0')) for i in range(0, 24)]

    for j, (step_dt, vals) in enumerate(plots.input['time_series'].items()):

        # Track variables for each day
        # For now, cast all values to native Python types
        foo_day.sdays += 1
        foo_day.doy = vals['doy']
        foo_day.dt_string = step_dt
        dt = pd.to_datetime(step_dt)
        foo_day.year = dt.year
        foo_day.month = dt.month
        foo_day.day = dt.day

        foo_day.ndvi = np.zeros((1, size))
        foo_day.capture = np.zeros((1, size))
        foo_day.refet = np.zeros((1, size))
        foo_day.irr_day = np.zeros((1, size), dtype=int)

        if foo_day.doy == 1:

            foo_day.irr_status = np.zeros((1, len(plots.input['order'])))
            foo_day.irr_doys = []

            for i, fid in enumerate(plots.input['order']):
                try:
                    irrigated = plots.input['irr_data'][fid][str(dt.year)]['irrigated']
                    foo_day.irr_doys.append(plots.input['irr_data'][fid][str(foo_day.year)]['irr_doys'])
                    foo_day.irr_status[0, i] = irrigated
                except KeyError:
                    foo_day.irr_status[0, i] = 0
                    foo_day.irr_doys.append([])

        for i, fid in enumerate(plots.input['order']):
            irrigated = foo_day.irr_status[0, i]
            if irrigated:
                foo_day.ndvi[0, i] = vals['ndvi_irr'][i]
                foo_day.capture[0, i] = vals['etf_irr_ct'][i]
                foo_day.refet[0, i] = vals['{}_mm'.format(config.refet_type)][i]
                foo_day.irr_day[0, i] = int(foo_day.doy in foo_day.irr_doys[i])

            else:
                foo_day.ndvi[0, i] = vals['ndvi_inv_irr'][i]
                foo_day.capture[0, i] = vals['etf_inv_irr_ct'][i]
                foo_day.refet[0, i] = vals['{}_mm_uncorr'.format(config.refet_type)][i]
                foo_day.irr_day[0, i] = 0

        foo_day.ndvi = foo_day.ndvi.reshape(1, -1)
        foo_day.capture = foo_day.capture.reshape(1, -1)
        foo_day.refet = foo_day.refet.reshape(1, -1)

        foo_day.min_temp = np.array(vals['tmin_c']).reshape(1, -1)
        foo_day.max_temp = np.array(vals['tmax_c']).reshape(1, -1)
        foo_day.temp_avg = (foo_day.min_temp + foo_day.max_temp) / 2.
        foo_day.srad = np.array(vals['srad_wm2']).reshape(1, -1)
        foo_day.precip = np.array(vals['prcp_mm'])

        if np.any(foo_day.precip > 0.):
            hr_ppt = np.array([vals[k] for k in hr_ppt_keys]).reshape(24, size)
            foo_day.hr_precip = hr_ppt

        foo_day.precip = foo_day.precip.reshape(1, -1)

        if step_dt == '2005-11-12':
            a = 1

        # Calculate height of vegetation.
        # Moved up to this point 12/26/07 for use in adj. Kcb and kc_max
        calculate_height.calculate_height(tracker)

        # Interpolate Kcb and make climate adjustment (for ETo basis)
        obs_kcb_daily.kcb_daily(config, plots, tracker, foo_day)

        # Calculate Kcb, Ke, ETc
        compute_field_et.compute_field_et(config, plots, tracker, foo_day,
                                          debug_flag)

        # Retrieve values from foo_day and write to output data frame
        # Eventually let compute_crop_et() write directly to output df

        if debug_flag:
            for i, fid in enumerate(targets):
                tracker.crop_df[fid][step_dt] = {}
                sample_idx = 0, i
                tracker.crop_df[fid][step_dt]['etref'] = foo_day.refet[sample_idx]

                eta_act = tracker.etc_act[sample_idx]
                tracker.crop_df[fid][step_dt]['capture'] = foo_day.capture[sample_idx]
                tracker.crop_df[fid][step_dt]['t'] = tracker.t[sample_idx]
                tracker.crop_df[fid][step_dt]['e'] = tracker.e[sample_idx]
                tracker.crop_df[fid][step_dt]['kc_act'] = tracker.kc_act[sample_idx]
                tracker.crop_df[fid][step_dt]['ks'] = tracker.ks[sample_idx]
                tracker.crop_df[fid][step_dt]['ke'] = tracker.ke[sample_idx]

                # water balance components
                tracker.crop_df[fid][step_dt]['et_act'] = eta_act

                ppt = foo_day.precip[sample_idx]
                tracker.crop_df[fid][step_dt]['ppt'] = ppt

                melt = tracker.melt[sample_idx]
                tracker.crop_df[fid][step_dt]['melt'] = melt
                rain = tracker.rain[sample_idx]
                tracker.crop_df[fid][step_dt]['rain'] = rain

                runoff = tracker.sro[sample_idx]
                tracker.crop_df[fid][step_dt]['runoff'] = runoff
                dperc = tracker.dperc[sample_idx]
                tracker.crop_df[fid][step_dt]['dperc'] = dperc

                depl_root = tracker.depl_root[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_root'] = depl_root
                depl_root_prev = tracker.depl_root_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_root_prev'] = depl_root_prev

                daw3 = tracker.daw3[sample_idx]
                tracker.crop_df[fid][step_dt]['daw3'] = daw3
                daw3_prev = tracker.daw3_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['daw3_prev'] = daw3_prev
                delta_daw3 = daw3 - daw3_prev
                tracker.crop_df[fid][step_dt]['delta_daw3'] = delta_daw3

                soil_water = tracker.soil_water[sample_idx]
                tracker.crop_df[fid][step_dt]['soil_water'] = soil_water
                soil_water_prev = tracker.soil_water_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['soil_water_prev'] = soil_water_prev

                depl_ze = tracker.depl_ze[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_ze'] = depl_ze
                tracker.crop_df[fid][step_dt]['p_rz'] = tracker.p_rz[sample_idx]
                tracker.crop_df[fid][step_dt]['p_eft'] = tracker.p_eft[sample_idx]
                tracker.crop_df[fid][step_dt]['fc'] = tracker.fc[sample_idx]
                tracker.crop_df[fid][step_dt]['few'] = tracker.few[sample_idx]
                tracker.crop_df[fid][step_dt]['aw'] = tracker.aw[sample_idx]
                tracker.crop_df[fid][step_dt]['aw3'] = tracker.aw3[sample_idx]
                tracker.crop_df[fid][step_dt]['taw'] = tracker.taw[sample_idx]
                tracker.crop_df[fid][step_dt]['taw3'] = tracker.taw3[sample_idx]
                tracker.crop_df[fid][step_dt]['irrigation'] = tracker.irr_sim[sample_idx]
                tracker.crop_df[fid][step_dt]['irr_day'] = foo_day.irr_day[sample_idx]
                tracker.crop_df[fid][step_dt]['swe'] = tracker.swe[sample_idx]
                tracker.crop_df[fid][step_dt]['snow_fall'] = tracker.snow_fall[sample_idx]
                tracker.crop_df[fid][step_dt]['tavg'] = foo_day.temp_avg[sample_idx]
                tracker.crop_df[fid][step_dt]['tmax'] = foo_day.max_temp[sample_idx]
                tracker.crop_df[fid][step_dt]['zr'] = tracker.zr[sample_idx]
                tracker.crop_df[fid][step_dt]['kc_bas'] = tracker.kc_bas[sample_idx]
                tracker.crop_df[fid][step_dt]['niwr'] = tracker.niwr[sample_idx]
                tracker.crop_df[fid][step_dt]['et_bas'] = tracker.etc_bas
                tracker.crop_df[fid][step_dt]['season'] = tracker.in_season

                water_out = eta_act + dperc + runoff
                water_stored = soil_water - soil_water_prev
                water_in = melt + rain
                balance = water_in - water_stored - water_out

                tracker.crop_df[fid][step_dt]['wbal'] = balance

                if abs(balance) > 0.1:
                    raise WaterBalanceError('Check November water balance')

        else:
            eta[j, :] = tracker.etc_act
            swe[j, :] = tracker.swe

    if debug_flag:
        # pass final dataframe to calling script

        tracker.crop_df = {fid: pd.DataFrame().from_dict(tracker.crop_df[fid], orient='index')[OUTPUT_FMT]
                           for fid in targets}
        return tracker.crop_df

    else:
        # if not debug, just return the actual ET and SWE results as ndarray
        return eta, swe


if __name__ == '__main__':
    pass
