import os
import json

from tqdm import tqdm
import numpy as np
import pandas as pd
import geopandas as gpd

from data_extraction.gridmet.gridmet import find_gridmet_points, download_gridmet

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def join_daily_timeseries(fields, gridmet_dir, landsat_table, snow, dst_dir, overwrite=False,
                          start_date=None, end_date=None, feature_id='FID', **kwargs):
    with open(snow, 'r') as f:
        snow = json.load(f)

    lst = pd.read_csv(landsat_table, parse_dates=True, index_col=0)
    start, end = lst.index[0], lst.index[-1]

    if 'params' not in kwargs.keys():
        params = set(['_'.join(x.split('_')[1:]) for x in lst.columns])
    else:
        params = kwargs['params']

    field_df = gpd.read_file(fields)
    field_df.index = field_df[feature_id]

    out_plots, bad = [], None

    for f, row in tqdm(field_df.iterrows(), total=field_df.shape[0]):

        if 'target_fields' in kwargs.keys():
            if str(f) not in kwargs['target_fields']:
                continue

        if pd.isna(row['GFID']):
            print(row[feature_id], 'was not assigned a Gridmet point')
            continue

        _file = os.path.join(dst_dir, '{}_daily.csv'.format(f))

        gridmet_file = os.path.join(gridmet_dir, 'gridmet_{}.csv'.format(int(row['GFID'])))

        try:
            gridmet = pd.read_csv(gridmet_file, index_col='date', parse_dates=True).loc[start: end]

            time_covered = ((pd.to_datetime(start_date) >= gridmet.index[0]) &
                            (pd.to_datetime(end_date) <= gridmet.index[-1]))

            if os.path.exists(_file) and not overwrite and time_covered:
                continue

            if not time_covered:
                bias_factors = fields.replace('.shp', '.json')
                download_gridmet(fields, bias_factors, gridmet_dir, start_date, end_date,
                                 overwrite=True, target_fields=[f], feature_id=feature_id)

            drop_cols = [c for c in gridmet.columns if '.' in c]
            if drop_cols:
                gridmet.drop(columns=drop_cols, inplace=True)

            swe_data = [(pd.to_datetime(d['date']), d['value']) for d in snow[f]]
            swe = pd.Series(index=[x[0] for x in swe_data], data=[x[1] for x in swe_data])
            swe = swe.sort_index()

            match_idx = [i for i in gridmet.index if i in swe.index]
            gridmet.loc[match_idx, 'obs_swe'] = swe

        except FileNotFoundError:
            print(gridmet_file, 'not found\n')
            continue

        except pd.errors.EmptyDataError:
            print(gridmet_file, 'empty\n')
            continue

        for p in params:
            gridmet.loc[lst.index, p] = lst['{}_{}'.format(f, p)]

        if start_date:
            gridmet = gridmet.loc[start_date:]
        if end_date:
            gridmet = gridmet.loc[:end_date]

        accept, bad = True, 0

        for i, r in gridmet.iterrows():
            if np.isnan(r['ndvi_irr']) and np.isnan(r['ndvi_inv_irr']):
                print('{} in {} has only nan in ndvi_irr and ndvi_inv_irr'.format(f, i.year))
                accept = False
                bad += 1
                break
            if np.isnan(r['etf_irr']) and np.isnan(r['etf_inv_irr']):
                print('{} in {} has only nan in etf_irr and etf_inv_irr'.format(f, i.year))
                accept = False
                bad += 1
                break

        if accept:
            gridmet.to_csv(_file)
            out_plots.append(f)

    print(f'{len(out_plots)} fields were successfully processed')
    print(f'{bad} fields were dropped due to missing data')


if __name__ == '__main__':
    root = '/home/dgketchum/PycharmProjects/swim-rs'
    fields_gridmet = os.path.join(root, 'tutorial', 'step_3_meteorology_extract', 'mt_sid_boulder_gfid.shp')
    met = os.path.join(root, 'tutorial', 'step_3_meteorology_extract', 'met_timeseries')
    landsat = os.path.join(root, 'tutorial/step_2_earth_engine_extract/landsat/remote_sensing.csv')
    snow = os.path.join(root, 'tutorial/step_2_earth_engine_extract/snodas/snodas.json')

    joined_timeseries = os.path.join(root, 'tutorial/step_4_model_data_prep/input_timeseries')
    if not os.path.isdir(joined_timeseries):
        os.mkdir(joined_timeseries)
    shapefile_path = os.path.join(root, 'tutorial/step_1_domain/mt_sid_boulder.shp')

    params = ['etf_inv_irr',
              'ndvi_inv_irr',
              'etf_irr',
              'ndvi_irr']
    params += ['{}_ct'.format(p) for p in params]

    join_daily_timeseries(fields=fields_gridmet,
                          gridmet_dir=met,
                          landsat_table=landsat,
                          snow=snow,
                          dst_dir=joined_timeseries,
                          overwrite=False,
                          start_date='2004-01-01',
                          end_date='2022-12-31',
                          feature_id='FID_1',
                          **{'params': params})

# ========================= EOF ====================================================================
