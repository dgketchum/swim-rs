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
# FLUX_SELECT = ['US-ADR', 'US-Bi1', 'US-Bi2', 'US-Blo', 'US-CZ3', 'US-Fmf',
#                'US-Fuf', 'US-Fwf', 'US-GLE', 'US-Hn2', 'US-Hn3', 'US-Jo2',
#                'US-MC1', 'US-Me1', 'US-Me2', 'US-Me5', 'US-Me6', 'US-Mj1',
#                'US-Mj2', 'US-NR1', 'US-Rwe', 'US-Rwf', 'US-Rws', 'US-SCg',
#                'US-SCs', 'US-SCw', 'US-Sne', 'US-SO2', 'US-SO3', 'US-SO4',
#                'US-Srr', 'US-Tw2', 'US-Tw3', 'US-Twt', 'US-Var', 'US-xJR',
#                'US-xNW', 'US-xRM', 'US-xYE', 'MB_Pch', 'S2', 'Almond_Low',
#                'Almond_Med', 'JPL1_JV114', 'UA1_JV187', 'UA1_KN18', 'UA2_JV330',
#                'UA2_KN20', 'UA3_JV108', 'UA3_KN15', 'BAR012', 'RIP760', 'SLM001',
#                'B_01', 'B_11', 'ET_1', 'ET_8', 'MOVAL', 'MR', 'TAM', 'VR', 'AFD',
#                'AFS', 'BPHV', 'BPLV', 'DVDV', 'KV_1', 'KV_2', 'KV_4', 'SPV_1',
#                'SPV_3', 'SV_5', 'SV_6', 'UMVW', 'UOVLO', 'UOVMD', 'UOVUP', 'WRV_1', 'WRV_2']

# Sites with clean records
# FLUX_SELECT = ['US-ADR', 'US-Blo', 'US-CZ3', 'US-Fmf', 'US-Fuf', 'US-GLE', 'US-Hn2', 'US-Hn3', 'US-Jo2', 'US-MC1',
#                'US-Me1', 'US-Me2', 'US-Me5', 'US-Me6', 'US-Mj1', 'US-Mj2', 'US-NR1', 'US-Rwe', 'US-Rwf', 'US-Rws',
#                'US-SCg', 'US-SCs', 'US-SCw', 'US-SO2', 'US-SO3', 'US-SO4', 'US-Srr', 'US-Var', 'US-xJR', 'US-xNW',
#                'US-xRM', 'US-xYE', 'MB_Pch', 'Almond_Low']

# Sites in PNW
# FLUX_SELECT = ['US-Me1', 'US-Me2', 'US-Me5', 'US-Me6', 'US-Mj1', 'US-Mj2',
#                'US-Rwe', 'US-Rwf', 'US-Rws', 'US-xYE']


FLUX_SELECT = ['US-MC1']

TONGUE_SELECT = [str(i) for i in [262, 334, 340, 346, 771, 875, 1377, 1378, 1483, 1526, 1581,
                                  1698, 1815, 1851, 1865, 1872, 1881, 1888, 1901]]

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


def prep_fields_json(fields, target_plots, input_ts, out_js, irr_data=None):
    with open(fields, 'r') as fp:
        fields = json.load(fp)

    dct = {'props': {i: r for i, r in fields.items() if i in target_plots}}

    missing = [x for x in target_plots if x not in dct['props'].keys()]
    if missing:
        print('Target sample missing: {}'.format(missing))
        [target_plots.remove(f) for f in missing]

    required_params = REQUIRED + REQ_IRR + REQ_UNIRR
    with open(irr_data, 'r') as fp:
        irr_data = json.load(fp)
    dct['irr_data'] = {fid: v for fid, v in irr_data.items() if fid in target_plots}

    dts, order = None, []
    first, arrays = True, {r: [] for r in required_params}
    for fid, v in dct['props'].items():
        _file = os.path.join(input_ts, '{}_daily.csv'.format(fid))
        df = pd.read_csv(_file, index_col='date', parse_dates=True)
        if first:
            doys = [int(dt.strftime('%j')) for dt in df.index]
            dts = [(int(r['year']), int(r['month']), int(r['day'])) for i, r in df.iterrows()]
            dts = ['{}-{:02d}-{:02d}'.format(y, m, d) for y, m, d in dts]
            data = {dt: {'doy': doy} for dt, doy in zip(dts, doys)}
            order = [fid]
            first = False
        else:
            order.append(fid)

        for p in required_params:
            a = df[p].values
            if np.any(np.isnan(a)) and p not in ACCEPT_NAN:
                raise ValueError
            arrays[p].append(a)

    for p in required_params:
        a = np.array(arrays[p]).T
        arrays[p] = a

    for i, dt in enumerate(dts):
        for p in required_params:
            data[dt][p] = list(arrays[p][i, :])

    dct.update({'order': order, 'time_series': data})
    with open(out_js, 'w') as fp:
        json.dump(dct, fp, indent=4)


def preproc(field_ids, src, _dir):
    for fid in field_ids:
        obs_file = os.path.join(src, '{}_daily.csv'.format(fid))
        data = pd.read_csv(obs_file, index_col=0, parse_dates=True)
        data.index = list(range(data.shape[0]))

        data['etf'] = data['etf_inv_irr']
        print('\n{}\npreproc ETf mean: {:.2f}'.format(fid, np.nanmean(data['etf'].values)))
        _file = os.path.join(project_dir, 'obs', 'obs_etf_{}.np'.format(fid))
        np.savetxt(_file, data['etf'].values)

        data['eta'] = data['eto_mm'] * data['etf_inv_irr']
        print('preproc ETa mean: {:.2f}'.format(np.nanmean(data['eta'].values)))
        _file = os.path.join(project_dir, 'obs', 'obs_eta_{}.np'.format(fid))
        np.savetxt(_file, data['eta'].values)

        print('preproc SWE mean: {:.2f}\n'.format(np.nanmean(data['obs_swe'].values)))
        _file = os.path.join(project_dir, 'obs', 'obs_swe_{}.np'.format(fid))
        np.savetxt(_file, data['obs_swe'].values)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/swim'
    if not os.path.exists(d):
        d = d = '/home/dgketchum/data/IrrigationGIS/swim'

    project = 'tongue'
    project_ws = os.path.join(d, 'examples', project)

    src_dir = os.path.join(project_ws, 'input_timeseries')

    fields_props = os.path.join(project_ws, 'properties', '{}_props.json'.format(project))
    cuttings = os.path.join(d, 'examples/{}/landsat/{}_cuttings.json'.format(project, project))

    select_fields_js = os.path.join(project_ws, 'prepped_input', '{}_input_sample.json'.format(project))

    prep_fields_json(fields_props, TONGUE_SELECT, src_dir, select_fields_js, irr_data=cuttings)

    project_dir = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}'.format(project)
    preproc(TONGUE_SELECT, src_dir, project_dir)

# ========================= EOF ====================================================================
