import os

import numpy as np
import pandas as pd
import geopandas as gpd

from data_extraction.gridmet.gridmet import find_gridmet_points, download_gridmet
from data_extraction.snodas.snodas import snodas_zonal_stats


def join_gridmet_remote_sensing_daily(fields, gridmet_dir, landsat_table, dst_dir, overwrite=False,
                                      start_date=None, end_date=None, **kwargs):
    lst = pd.read_csv(landsat_table, parse_dates=True, index_col=0)
    start, end = lst.index[0], lst.index[-1]
    if 'params' not in kwargs.keys():
        params = set(['_'.join(x.split('_')[1:]) for x in lst.columns])
    else:
        params = kwargs['params']

    fields = gpd.read_file(fields)
    fields.index = fields['FID']

    for f, row in fields.iterrows():

        if 'target_fields' in kwargs.keys():
            if f not in kwargs['target_fields']:
                continue

        if pd.isna(row['GFID']):
            print(row['FID'], 'was not assigned a Gridmet point')
            continue

        _file = os.path.join(dst_dir, '{}_daily.csv'.format(f))
        if os.path.exists(_file) and not overwrite:
            continue

        gridmet_file = os.path.join(gridmet_dir, 'gridmet_historical_{}.csv'.format(int(row['GFID'])))

        try:
            gridmet = pd.read_csv(gridmet_file, index_col='date', parse_dates=True).loc[start: end]
            drop_cols = [c for c in gridmet.columns if '.' in c]
            if drop_cols:
                gridmet.drop(columns=drop_cols, inplace=True)

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

        gridmet.to_csv(_file)
        print(_file)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/swim'
    if not os.path.exists(d):
        d = d = '/home/dgketchum/data/IrrigationGIS/swim'

    project = 'tongue'
    project_ws = os.path.join(d, 'examples', project)

    gridmet = os.path.join(d, 'gridmet')
    rasters_ = os.path.join(gridmet, '../data_extraction/gridmet_corrected', 'correction_surfaces_aea')
    grimet_cent = os.path.join(gridmet, 'gridmet_centroids_tongue.shp')

    fields_shp = os.path.join(project_ws, 'gis', '{}_fields.shp'.format(project))
    fields_gridmet = os.path.join(project_ws, 'gis', '{}_fields_gfid.shp'.format(project))
    gridmet_factors = os.path.join(project_ws, 'gis', '{}_fields_gfid.json'.format(project))
    met = os.path.join(project_ws, 'met_timeseries')

    # TODO: write gridmet data to a common directory, instead of project ws

    # find_gridmet_points(fields_shp, grimet_cent, rasters_, fields_gridmet, gridmet_factors, field_select=None)

    # targets = [1779, 1787, 1793, 1794, 1797, 1801, 1804]
    targets = list(range(1770, 1805))

    # download_gridmet(fields_gridmet, gridmet_factors, met, start='2000-01-01', end='2020-12-31')

    fields_shp_wgs = os.path.join(project_ws, 'gis', '{}_fields_wgs.shp'.format(project))
    snow_ts = os.path.join(project_ws, 'snow_timeseries', 'snodas_{}.json'.format(project))

    s_dir = '/data/hdd1/snodas/processed/swe'
    if not os.path.isdir(s_dir):
        s_dir = '/media/research/IrrigationGIS/climate/snodas/processed/swe'
    snodas_zonal_stats(fields_shp_wgs, s_dir, snow_ts, targets=None)

    landsat = os.path.join(project_ws, 'landsat', '{}_sensing.csv'.format(project))
    dst_dir_ = os.path.join(project_ws, 'input_timeseries')

    params = ['etf_inv_irr',
              'ndvi_inv_irr',
              'etf_irr',
              'ndvi_irr']

    params += ['{}_ct'.format(p) for p in params]

    # join_gridmet_remote_sensing_daily(fields_gridmet, met, landsat, dst_dir_, overwrite=False,
    #                                   start_date='2000-01-01', end_date='2020-12-31', **{'params': params})

# ========================= EOF ====================================================================
