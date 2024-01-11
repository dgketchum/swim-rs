import os

import numpy as np
import pandas as pd
import geopandas as gpd

from gridmet_corrected.gridmet import corrected_gridmet_clustered


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

        _file = os.path.join(dst_dir, '{}_daily.csv'.format(f))
        if os.path.exists(_file) and not overwrite:
            continue

        gridmet_file = os.path.join(gridmet_dir, 'gridmet_historical_{}.csv'.format(int(row['GFID'])))
        gridmet = pd.read_csv(gridmet_file, index_col='date', parse_dates=True).loc[start: end]

        for p in params:
            gridmet.loc[lst.index, p] = lst['{}_{}'.format(f, p)]

        if start_date:
            gridmet = gridmet.loc[start_date:]
        if end_date:
            gridmet = gridmet.loc[:end_date]

        gridmet.to_csv(_file)
        print(_file)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/et-demands'

    project = 'flux'
    project_ws = os.path.join(d, 'examples', project)

    gridmet = os.path.join(d, 'gridmet')
    rasters_ = os.path.join(gridmet, 'gridmet_corrected', 'correction_surfaces_aea')
    grimet_cent = os.path.join(gridmet, 'gridmet_centroids.shp')

    fields_shp = os.path.join(project_ws, 'gis', '{}_fields_sample.shp'.format(project))
    fields_gridmet = os.path.join(project_ws, 'gis', '{}_fields_sample_gfid.shp'.format(project))
    met = os.path.join(project_ws, 'met_timeseries')
    corrected_gridmet_clustered(fields_shp, grimet_cent, fields_gridmet, met, rasters_, start='2000-01-01',
                                end='2020-12-31')

    landsat = os.path.join(project_ws, 'landsat', '{}_sensing_sample.csv'.format(project))
    dst_dir_ = os.path.join(project_ws, 'input_timeseries')

    params = ['etf_inv_irr',
              'ndvi_inv_irr',
              'etf_irr',
              'ndvi_irr']

    params += ['{}_ct'.format(p) for p in params]

    join_gridmet_remote_sensing_daily(fields_gridmet, met, landsat, dst_dir_, overwrite=True,
                                      start_date='2000-01-01', end_date='2020-12-31', **{'params': params})
# ========================= EOF ====================================================================
