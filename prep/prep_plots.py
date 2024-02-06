import json
import os

import numpy as np
import pandas as pd

REQUIRED = ['prcp_mm']

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


def prep_fields_json(fields, target_plots, input_ts, out_js, ltype='unirrigated', irr_data=None):

    with open(fields, 'r') as fp:
        fields = json.load(fp)

    dct = {'props': {i: r for i, r in fields.items() if i in target_plots}}

    required_params = None
    if ltype == 'irrigated':
        required_params = REQUIRED + REQ_IRR
        with open(irr_data, 'r') as fp:
            irr_data = json.load(fp)
        dct['irr_data'] = {fid: v for fid, v in irr_data.items() if fid in target_plots}

    elif ltype == 'unirrigated':
        required_params = REQUIRED + REQ_UNIRR

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
            if np.any(np.isnan(a)):
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
        data['eta'] = data['etr_mm'] * data['etf_inv_irr']
        data = data[['eta']]
        print('preproc mean: {}'.format(np.nanmean(data.values)))
        _file = os.path.join(project_dir, 'obs', 'obs_eta_{}.np'.format(fid))
        np.savetxt(_file, data.values)
        print('Wrote obs to {}'.format(_file))


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/swim'
    if not os.path.exists(d):
        d = d = '/home/dgketchum/data/IrrigationGIS/swim'

    project = 'tongue'
    project_ws = os.path.join(d, 'examples', project)

    src_dir = os.path.join(project_ws, 'input_timeseries')

    fields_props = os.path.join(project_ws, 'properties', '{}_props.json'.format(project))
    cuttings = '/media/research/IrrigationGIS/swim/examples/tongue/landsat/{}_cuttings.json'.format(project)

    select_fields = [str(f) for f in [1779, 1787, 1793, 1794, 1797, 1801, 1804]]
    select_fields_js = os.path.join(project_ws, 'prepped_input', '{}_input.json'.format(project))

    prep_fields_json(fields_props, select_fields, src_dir, select_fields_js, ltype='irrigated', irr_data=cuttings)

    project_dir = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}'.format(project)

    preproc(select_fields, src_dir, project_dir)

# ========================= EOF ====================================================================
