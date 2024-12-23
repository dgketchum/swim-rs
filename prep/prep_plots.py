try:
    import ujson as json
except ImportError:
    try:
        import simplejson as json
    except ImportError:
        import json

import os

import numpy as np
import pandas as pd

# All Sites
FLUX_SELECT = ['US-ADR', 'US-Bi1', 'US-Bi2', 'US-Blo', 'US-CZ3', 'US-Fmf',
               'US-Fuf', 'US-Fwf', 'US-GLE', 'US-Hn2', 'US-Hn3', 'US-Jo2',
               'US-MC1', 'US-Me1', 'US-Me2', 'US-Me5', 'US-Me6', 'US-Mj1',
               'US-Mj2', 'US-NR1', 'US-Rwe', 'US-Rwf', 'US-Rws', 'US-SCg',
               'US-SCs', 'US-SCw', 'US-Sne', 'US-SO2', 'US-SO3', 'US-SO4',
               'US-Srr', 'US-Tw2', 'US-Tw3', 'US-Twt', 'US-Var', 'US-xJR',
               'US-xNW', 'US-xRM', 'US-xYE', 'MB_Pch', 'S2', 'Almond_Low',
               'Almond_Med', 'JPL1_JV114', 'UA1_JV187', 'UA1_KN18', 'UA2_JV330',
               'UA2_KN20', 'UA3_JV108', 'UA3_KN15', 'BAR012', 'RIP760', 'SLM001',
               'B_01', 'B_11', 'ET_1', 'ET_8', 'MOVAL', 'MR', 'TAM', 'VR', 'AFD',
               'AFS', 'BPHV', 'BPLV', 'DVDV', 'KV_1', 'KV_2', 'KV_4', 'SPV_1',
               'SPV_3', 'SV_5', 'SV_6', 'UMVW', 'UOVLO', 'UOVMD', 'UOVUP', 'WRV_1', 'WRV_2']

# Sites with clean records
# FLUX_SELECT = ['US-ADR', 'US-Blo', 'US-CZ3', 'US-Fmf', 'US-Fuf', 'US-GLE', 'US-Hn2', 'US-Hn3', 'US-Jo2', 'US-MC1',
#                'US-Me1', 'US-Me2', 'US-Me5', 'US-Me6', 'US-Mj1', 'US-Mj2', 'US-NR1', 'US-Rwe', 'US-Rwf', 'US-Rws',
#                'US-SCg', 'US-SCs', 'US-SCw', 'US-SO2', 'US-SO3', 'US-SO4', 'US-Srr', 'US-Var', 'US-xJR', 'US-xNW',
#                'US-xRM', 'US-xYE', 'MB_Pch', 'Almond_Low']

# Sites in PNW
# FLUX_SELECT = ['US-Me1', 'US-Me2', 'US-Me5', 'US-Me6', 'US-Mj1', 'US-Mj2',
#                'US-Rwe', 'US-Rwf', 'US-Rws', 'US-xYE']


# FLUX_SELECT = ['US-MC1']

TONGUE_SELECT = [str(f) for f in [1609]]

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


def prep_fields_json(fields, target_plots, input_ts, out_js, irr_data=None, force_unirrigated=False):
    with open(fields, 'r') as fp:
        fields = json.load(fp)

    dct = {'props': {i: r for i, r in fields.items() if i in target_plots}}

    missing = [x for x in target_plots if x not in dct['props'].keys()]
    missing += [x for x in target_plots if not os.path.exists(os.path.join(input_ts, '{}_daily.csv'.format(x)))]
    missing = list(set(missing))

    if missing:
        print('Target sample missing: {}'.format(missing))
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
    for fid, v in dct['props'].items():

        if fid in missing:
            continue
            
        _file = os.path.join(input_ts, '{}_daily.csv'.format(fid))
        df = pd.read_csv(_file, index_col='date', parse_dates=True)
        if first:
            shape = df.shape[0]
            print('Input shape: {}'.format(df.shape))
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
    with open(out_js, 'w') as fp:
        json.dump(dct, fp, indent=4)

    return target_plots, missing


def preproc(field_ids, src, _dir):

    ct = 0

    for fid in field_ids:
        obs_file = os.path.join(src, '{}_daily.csv'.format(fid))
        data = pd.read_csv(obs_file, index_col=0, parse_dates=True)
        data.index = list(range(data.shape[0]))

        data['etf'] = data['etf_inv_irr']
        print('\n{}\npreproc ETf mean: {:.2f}'.format(fid, np.nanmean(data['etf'].values)))
        _file = os.path.join(_dir, 'obs', 'obs_etf_{}.np'.format(fid))
        np.savetxt(_file, data['etf'].values)

        data['eta'] = data['eto_mm'] * data['etf_inv_irr']
        print('preproc ETa mean: {:.2f}'.format(np.nanmean(data['eta'].values)))
        _file = os.path.join(_dir, 'obs', 'obs_eta_{}.np'.format(fid))
        np.savetxt(_file, data['eta'].values)

        print('preproc SWE mean: {:.2f}\n'.format(np.nanmean(data['obs_swe'].values)))
        _file = os.path.join(_dir, 'obs', 'obs_swe_{}.np'.format(fid))
        np.savetxt(_file, data['obs_swe'].values)

        ct += 1

    print('Prepped {} fields input'.format(ct))


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/swim'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/swim'

    project = 'flux'
    if project == 'tongue':
        fields = TONGUE_SELECT
    elif project == 'flux':
        fields = FLUX_SELECT

    project_ws = os.path.join(d, 'examples', project)

    src_dir = os.path.join(project_ws, 'input_timeseries')

    fields_props = os.path.join(project_ws, 'properties', '{}_props.json'.format(project))
    cuttings = os.path.join(d, 'examples/{}/landsat/{}_cuttings.json'.format(project, project))

    select_fields_js = os.path.join(project_ws, 'prepped_input', '{}_input_sample.json'.format(project))

    processed_targets, excluded_targets = prep_fields_json(fields_props, fields,
                                                           src_dir, select_fields_js, irr_data=cuttings)

    project_dir = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}'.format(project)
    preproc(processed_targets, src_dir, project_dir)

# ========================= EOF ====================================================================
