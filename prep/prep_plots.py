import os
import json

import numpy as np
import geopandas as gpd
import pandas as pd

REQUIRED = ['prcp_mm',
            'etr_mm',
            'eto_mm',
            'etr_mm_uncorr',
            'eto_mm_uncorr',
            'etf_inv_irr',
            'ndvi_inv_irr',
            'etf_irr',
            'ndvi_irr',
            'etf_inv_irr_ct',
            'ndvi_inv_irr_ct',
            'etf_irr_ct',
            'ndvi_irr_ct']


def prep_fields_json(fields, input_ts, target_plots, out_js, data_out, idx_col='FID'):
    df = gpd.read_file(fields)
    df.index = df[idx_col]
    df = df.loc[target_plots]
    df.drop(columns=['STATE', 'geometry'], inplace=True)

    dct = {i: r.to_dict() for i, r in df.iterrows()}

    dts = None
    first, arrays = True, {r: [] for r in REQUIRED}
    for fid, v in dct.items():
        _file = os.path.join(input_ts, '{}_daily.csv'.format(fid))
        df = pd.read_csv(_file, index_col='date', parse_dates=True)
        if first:
            doys = [int(dt.strftime('%j')) for dt in df.index]
            dts = [(int(r['year']), int(r['month']), int(r['day'])) for i, r in df.iterrows()]
            dts = ['{}-{:02d}-{:02d}'.format(y, m, d) for y, m, d in dts]
            data = {dt: {'doy': doy} for dt, doy in zip(dts, doys)}
            data['order'] = [fid]
            first = False
        else:
            data['order'].append(fid)

        for p in REQUIRED:
            arrays[p].append(df[p].values)

    for p in REQUIRED:
        arrays[p] = np.array(arrays[p]).T

    for i, dt in enumerate(dts):
        for p in REQUIRED:
            data[dt][p] = list(arrays[p][i, :])

    with open(out_js, 'w') as fp:
        json.dump(dct, fp, indent=4)

    with open(data_out, 'w') as fp:
        json.dump(data, fp, indent=4)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/swim'
    if not os.path.exists(d):
        d = d = '/home/dgketchum/data/IrrigationGIS/swim'

    project = 'tongue'
    project_ws = os.path.join(d, 'examples', project)

    src_dir = os.path.join(project_ws, 'input_timeseries')

    fields_shp = os.path.join(project_ws, 'gis', '{}_fields.shp'.format(project))

    select_fields = [1778, 1791, 1804, 1853, 1375]

    select_fields_js = os.path.join(project_ws, 'prepped_input', '{}_fields.json'.format(project))
    input_data = os.path.join(project_ws, 'prepped_input', '{}_data.json'.format(project))

    prep_fields_json(fields_shp, src_dir, select_fields, select_fields_js, input_data, idx_col='FID')

# ========================= EOF ====================================================================
