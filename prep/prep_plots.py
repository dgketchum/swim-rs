try:
    import ujson as json
except ImportError:
    try:
        import simplejson as json
    except ImportError:
        import json

import os

from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime

from swim.config import ProjectConfig
from swim.input import SamplePlots

REQUIRED = ['tmin_c', 'tmax_c', 'srad_wm2', 'obs_swe', 'prcp_mm', 'nld_ppt_d',
            'prcp_hr_00', 'prcp_hr_01', 'prcp_hr_02', 'prcp_hr_03', 'prcp_hr_04',
            'prcp_hr_05', 'prcp_hr_06', 'prcp_hr_07', 'prcp_hr_08', 'prcp_hr_09', 'prcp_hr_10',
            'prcp_hr_11', 'prcp_hr_12', 'prcp_hr_13', 'prcp_hr_14', 'prcp_hr_15', 'prcp_hr_16',
            'prcp_hr_17', 'prcp_hr_18', 'prcp_hr_19', 'prcp_hr_20', 'prcp_hr_21', 'prcp_hr_22',
            'prcp_hr_23']

REQ_UNIRR = ['etr_mm_uncorr',
             'eto_mm_uncorr',
             'etf_inv_irr',
             'ndvi_inv_irr',
             'etf_inv_irr_ct',
             'ndvi_inv_irr_ct']

REQ_IRR = ['etr_mm',
           'eto_mm',
           'etf_irr',
           'ndvi_irr',
           'etf_irr_ct',
           'ndvi_irr_ct']

ACCEPT_NAN = REQ_IRR + REQ_UNIRR + ['obs_swe']


def prep_fields_json(fields, input_ts, out_js, target_plots=None, irr_data=None, force_unirrigated=False):
    with open(fields, 'r') as fp:
        fields = json.load(fp)

    if target_plots is None:
        target_plots = list(fields.keys())

    dct = {'props': {i: r for i, r in fields.items() if i in target_plots}}

    missing = [x for x in target_plots if x not in dct['props'].keys()]
    missing += [x for x in target_plots if not os.path.exists(os.path.join(input_ts, '{}_daily.csv'.format(x)))]
    missing = list(set(missing))

    if missing:
        print('Target sample(s) missing: {}'.format(missing))
        [target_plots.remove(f) for f in missing]
        if not target_plots:
            return target_plots, missing

    with open(irr_data, 'r') as fp:
        irr_data = json.load(fp)

    if force_unirrigated:

        with open(force_unirrigated, 'r') as fp:
            ndvi = json.load(fp)

        unirr_ndvi = ndvi['ndvi_inv_irr'] + [ndvi['ndvi_inv_irr'][-1]]
        dt = ['{}-{}'.format(d.month, d.day) for d in pd.date_range('2000-01-01', '2000-12-31')]
        unirr_ndvi = {d: unirr_ndvi[j] for j, d in enumerate(dt)}

        if 'ndvi_inv_irr' in ACCEPT_NAN:
            ACCEPT_NAN.remove('ndvi_inv_irr')

        required_params = REQUIRED + REQ_UNIRR
        dct['irr_data'] = {fid: {'fallow_years': []} for fid, v in irr_data.items() if fid in target_plots}

    else:
        required_params = REQUIRED + REQ_IRR + REQ_UNIRR
        dct['irr_data'] = {fid: v for fid, v in irr_data.items() if fid in target_plots}

    dts, order = None, []
    first, arrays, shape = True, {r: [] for r in required_params}, None
    for fid, v in tqdm(dct['props'].items(), total=len(dct['props'])):

        if fid in missing:
            continue

        _file = os.path.join(input_ts, '{}_daily.csv'.format(fid))
        df = pd.read_csv(_file, index_col='date', parse_dates=True)
        if first:
            shape = df.shape[0]
            doys = [int(dt.strftime('%j')) for dt in df.index]
            dts = [(int(r['year']), int(r['month']), int(r['day'])) for i, r in df.iterrows()]
            dts = ['{}-{:02d}-{:02d}'.format(y, m, d) for y, m, d in dts]
            data = {dt: {'doy': doy} for dt, doy in zip(dts, doys)}
            order = [fid]
            first = False
        else:
            order.append(fid)
            if not df.shape[0] == shape:
                print('{} does not have shape {}'.format(fid, df.shape[0]))
                continue

        for p in required_params:
            a = df[p].values
            if np.any(np.isnan(a)) and p not in ACCEPT_NAN:
                if p == 'ndvi_inv_irr':
                    for i, r in df[p].copy().items():
                        if np.isnan(r):
                            df.loc[i, p] = unirr_ndvi['{}-{}'.format(i.month, i.day)]
                    a = df[p].values
                else:
                    raise ValueError
            arrays[p].append(a)

    for p in required_params:
        a = np.array(arrays[p]).T
        arrays[p] = a

    for i, dt in enumerate(dts):
        for p in required_params:
            data[dt][p] = arrays[p][i, :].tolist()

    dct.update({'order': order, 'time_series': data})

    # write large json line-by-line
    with open(out_js, 'w') as f:
        for key, value in dct.items():
            json.dump({key: value}, f)
            f.write('\n')

    print(f'wrote {out_js}')

    return target_plots, missing


def preproc(config_file, project_ws):
    ct = 0

    print('Writing observations to file...')
    config = ProjectConfig()
    config.read_config(config_file, project_ws)
    start = datetime.strftime(config.start_dt, '%Y-%m-%d')
    end = datetime.strftime(config.end_dt, '%Y-%m-%d')

    fields = SamplePlots()
    fields.initialize_plot_data(config)

    if not os.path.isdir(config.obs_folder):
        os.mkdir(config.obs_folder)

    for fid in fields.input['order']:

        data = fields.input_to_dataframe(fid)
        irr_threshold = config.irr_threshold
        irr_years = [int(k) for k, v in fields.input['irr_data'][fid].items() if k != 'fallow_years'
                     and v['f_irr'] >= irr_threshold]

        irr_index = [i for i in data.index if i.year in irr_years]

        data = data.loc[start: end]

        data['etf'] = data['etf_inv_irr']
        data.loc[irr_index, 'etf'] = data.loc[irr_index, 'etf_irr']

        print('\n{}\npreproc ETf mean: {:.2f}'.format(fid, np.nanmean(data['etf'].values)))
        _file = os.path.join(config.obs_folder, 'obs_etf_{}.np'.format(fid))
        np.savetxt(_file, data['etf'].values)

        print('preproc SWE mean: {:.2f}\n'.format(np.nanmean(data['obs_swe'].values)))
        _file = os.path.join(config.obs_folder, 'obs_swe_{}.np'.format(fid))
        np.savetxt(_file, data['obs_swe'].values)

        ct += 1

    print('Prepped {} fields input'.format(ct))


if __name__ == '__main__':
    root = '/home/dgketchum/PycharmProjects/swim-rs'

    project_ws_ = os.path.join(root, 'tutorials', '4_Flux_Network')
    data = os.path.join(project_ws_, 'data')
    landsat = os.path.join(data, 'landsat')

    properties_json = os.path.join(data, 'properties', 'calibration_properties.json')
    cuttings_json = os.path.join(landsat, 'calibration_cuttings.json')
    joined_timeseries = os.path.join(data, 'plot_timeseries')
    prepped_input = os.path.join(data, 'prepped_input.json')

    # processed_targets, excluded_targets = prep_fields_json(properties_json, joined_timeseries, prepped_input,
    #                                                        target_plots=None, irr_data=cuttings_json)

    obs_dir = os.path.join(project_ws_, 'obs')
    if not os.path.isdir(obs_dir):
        os.makedirs(obs_dir, exist_ok=True)

    project_ws_ = os.path.join(root, 'tutorials', '4_Flux_Network')
    config_path = os.path.join(data, 'config.toml')

    preproc(config_path, project_ws_)

# ========================= EOF ====================================================================
