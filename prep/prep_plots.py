import os
import json

import numpy as np
import geopandas as gpd
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


def prep_fields_json(fields, input_ts, target_plots, out_js, data_out, idx_col='FID', ltype='unirrigated'):
    gdf = gpd.read_file(fields)
    gdf.index = [str(i) for i in gdf[idx_col]]
    gdf = gdf.loc[target_plots]
    gdf.drop(columns=['STATE', 'geometry'], inplace=True)

    dct = {i: r.to_dict() for i, r in gdf.iterrows()}

    if ltype == 'irrigated':
        required_params = REQUIRED + REQ_IRR
    elif ltype == 'unirrigated':
        required_params = REQUIRED + REQ_UNIRR

    dts, order = None, []
    first, arrays = True, {r: [] for r in required_params}
    for fid, v in dct.items():
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

    dct = {'order': order, 'plots': dct}
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

    select_fields = [str(f) for f in [1779, 1787, 1793, 1794, 1797, 1801, 1804]]

    select_fields_js = os.path.join(project_ws, 'prepped_input', '{}_fields.json'.format(project))
    input_data = os.path.join(project_ws, 'prepped_input', '{}_data.json'.format(project))

    prep_fields_json(fields_shp, src_dir, select_fields, select_fields_js, input_data,
                     idx_col='FID', ltype='irrigated')

# ========================= EOF ====================================================================
