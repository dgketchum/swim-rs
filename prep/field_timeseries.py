import os
import json

from tqdm import tqdm
import numpy as np
import pandas as pd
import geopandas as gpd

from prep import get_ensemble_parameters, get_openet_sites
from data_extraction.gridmet.gridmet import find_gridmet_points, download_gridmet

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def join_daily_timeseries(fields, gridmet_dir, landsat_table, snow, dst_dir, overwrite=False,
                          start_date=None, end_date=None, feature_id='FID', **kwargs):
    with open(snow, 'r') as f:
        snow = json.load(f)

    lst = pd.read_csv(landsat_table, parse_dates=True, index_col=0)
    lst = lst.sort_index(axis=1)
    lst_start, lst_end = lst.index[0], lst.index[-1]

    remote_sensing_coverage = ((pd.to_datetime(start_date) >= lst_start) &
                               (pd.to_datetime(end_date) <= lst_end))
    if not remote_sensing_coverage:
        raise ValueError('Remote sensing data does not cover requested time period')

    lst = lst.loc[start_date: end_date]

    if 'params' not in kwargs.keys():
        params = set(['_'.join(x.split('_')[1:]) for x in lst.columns])
        models = set(['_'.join(x.split('_')[0]) for x in lst.columns])
    else:
        params = kwargs['params']
        models = list(set([m.split('_')[0] for m in kwargs['params']]))

    field_df = gpd.read_file(fields)
    field_df.index = field_df[feature_id]

    out_plots, bad, time_covered = [], None, False

    for f, row in tqdm(field_df.iterrows(), total=field_df.shape[0]):

        if 'target_fields' in kwargs.keys():
            if str(f) not in kwargs['target_fields']:
                continue

        if pd.isna(row['GFID']):
            print(row[feature_id], 'was not assigned a Gridmet point')
            continue

        _file = os.path.join(dst_dir, '{}_daily.csv'.format(f))

        gridmet_file = os.path.join(gridmet_dir, 'gridmet_{}.csv'.format(int(row['GFID'])))

        if os.path.exists(gridmet_file):
            gridmet = pd.read_csv(gridmet_file, index_col='date', parse_dates=True).loc[start_date: end_date]
            gridmet.index = pd.DatetimeIndex(gridmet.index)
            missing_gridmet = False
            time_covered = ((pd.to_datetime(start_date) >= gridmet.index[0]) &
                            (pd.to_datetime(end_date) <= gridmet.index[-1]))
        else:
            missing_gridmet = True

        if os.path.exists(_file) and not overwrite and time_covered:
            continue

        if not time_covered or missing_gridmet:
            bias_factors = fields.replace('.shp', '.json')
            gridmet = download_gridmet(fields, bias_factors, gridmet_dir, start_date, end_date, return_df=True,
                                       overwrite=False, append=True, target_fields=[f], feature_id=feature_id)
            gridmet = gridmet.loc[start_date: end_date]

        drop_cols = [c for c in gridmet.columns if '.' in c]
        if drop_cols:
            gridmet.drop(columns=drop_cols, inplace=True)

        swe_data = [(pd.to_datetime(d['date']), d['value']) for d in snow[f]]
        swe = pd.Series(index=[x[0] for x in swe_data], data=[x[1] for x in swe_data])
        swe = swe.sort_index()

        match_idx = [i for i in gridmet.index if i in swe.index]
        gridmet.loc[match_idx, 'obs_swe'] = swe

        for p in params:
            gridmet.loc[lst.index, p] = lst['{}_{}'.format(f, p)]

        if start_date:
            gridmet = gridmet.loc[start_date:]
        if end_date:
            gridmet = gridmet.loc[:end_date]

        accept, bad = True, 0

        chkdf = gridmet.resample('A').sum()
        for m in models:
            if np.isnan(chkdf[f'{m}_irr']) and np.isnan(chkdf[f'{m}_inv_irr']):
                print('{} in {} has only nan in ndvi_irr and ndvi_inv_irr'.format(f, i.year))
                accept = False
                bad += 1
                break

        if accept:
            gridmet.to_csv(_file)
            out_plots.append(f)
            print(f'wrote {_file}')

    print(f'{len(out_plots)} fields were successfully processed')
    print(f'{bad} fields were dropped due to missing data')


if __name__ == '__main__':

    project = '5_Flux_Ensemble'

    root = '/data/ssd2/swim'
    data = os.path.join(root, project, 'data')
    if not os.path.isdir(root):
        root = '/home/dgketchum/PycharmProjects/swim-rs'
        data = os.path.join(root, 'tutorials', project, 'data')

    landsat = os.path.join(data, 'landsat')
    remote_sensing_file = os.path.join(landsat, 'remote_sensing.csv')

    FEATURE_ID = 'field_1'

    fields_gridmet = os.path.join(data, 'gis', 'flux_fields_gfid.shp')
    met = os.path.join(data, 'met_timeseries')

    joined_timeseries = os.path.join(data, 'plot_timeseries')
    snow = os.path.join(data, 'snodas', 'snodas.json')

    sites_ = get_openet_sites(fields_gridmet)
    remote_sensing_parameters = get_ensemble_parameters()

    join_daily_timeseries(fields=fields_gridmet,
                          gridmet_dir=met,
                          landsat_table=remote_sensing_file,
                          snow=snow,
                          dst_dir=joined_timeseries,
                          overwrite=True,
                          start_date='2016-01-01',
                          end_date='2024-12-31',
                          feature_id=FEATURE_ID,
                          **{'params': remote_sensing_parameters,
                             'target_fields': sites_})

# ========================= EOF ====================================================================
