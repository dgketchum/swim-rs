import os

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterstats import zonal_stats
from tqdm import tqdm

from prep import get_flux_sites


def landsat_time_series_image(in_shp, tif_dir, years, out_csv, out_csv_ct, min_ct=100, feature_id='FID'):
    """
    Intended to process raw tif to tabular data using zonal statistics on polygons.
    See e.g., 'ndvi_export.export_ndvi() to export such images from Earth Engine. The output of this function
    should be the same format and structure as that from landsat_time_series_station() and
    landsat_time_series_multipolygon(). Ensure the .tif and
    .shp are both in the same coordinate reference system.
    :param feature_id:
    :param in_shp:
    :param tif_dir:
    :param years:
    :param out_csv:
    :param out_csv_ct:
    :param min_ct:
    :return:
    """
    gdf = gpd.read_file(in_shp)
    gdf.index = gdf[feature_id]

    adf, ctdf, first = None, None, True

    for yr in years:

        file_list, dts = get_tif_list(tif_dir, yr)

        dt_index = pd.date_range('{}-01-01'.format(yr), '{}-12-31'.format(yr), freq='D')
        df = pd.DataFrame(index=dt_index, columns=gdf.index)
        ct = pd.DataFrame(index=dt_index, columns=gdf.index)

        print('\n', yr, len(file_list))
        for dt, f in tqdm(zip(dts, file_list), total=len(file_list)):
            stats = zonal_stats(in_shp, f, stats=['mean', 'count'], nodata=0.0, categorical=False, all_touched=False)
            stats = [x['mean'] if isinstance(x['mean'], float) and x['count'] > min_ct else np.nan for x in stats]
            df.loc[dt, :] = stats
            ct.loc[dt, :] = ~pd.isna(stats)
            df.loc[dt, :] /= 1000

        df = df.astype(float).interpolate()
        df = df.interpolate(method='bfill')

        ct = ct.fillna(0)
        ct = ct.astype(int)

        if first:
            adf = df.copy()
            ctdf = ct.copy()
            first = False
        else:
            adf = pd.concat([adf, df], axis=0, ignore_index=False, sort=True)
            ctdf = pd.concat([ctdf, ct], axis=0, ignore_index=False, sort=True)

    adf.to_csv(out_csv)
    ctdf.to_csv(out_csv_ct)


def sparse_landsat_time_series(in_shp, csv_dir, years, out_csv, out_csv_ct, feature_id='FID',
                               select=None, footprint_spec=None):
    """
    """
    gdf = gpd.read_file(in_shp)
    gdf.index = gdf[feature_id]

    if select:
        gdf = gdf.loc[select]

    print(csv_dir)

    adf, ctdf, first, prev_df = None, None, True, None

    if footprint_spec is None:
        file_list = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir) if
                     x.endswith('.csv')]
    else:
        file_list = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir) if
                     x.endswith('.csv') and f'_p{footprint_spec}' in x]

    rs_years = set([os.path.basename(f).split('.')[0].split('_')[-1] for f in file_list])
    rs_years = sorted([int(y) for y in rs_years])

    print(f'{len(rs_years)} years with remote sensing data, {rs_years[0]} to {rs_years[-1]}')

    empty_yrs = [y for y in years if y not in rs_years]

    if len(empty_yrs) > 0:
        print(f'{len(empty_yrs)} years without remote sensing data, {empty_yrs[0]} to {empty_yrs[-1]}')
        dt_index = pd.date_range('{}-01-01'.format(empty_yrs[0]), '{}-12-31'.format(empty_yrs[-1]), freq='D')
        adf = pd.DataFrame(data=np.zeros((len(dt_index), len(gdf.index))) * np.nan, index=dt_index, columns=gdf.index)
        ctdf = pd.DataFrame(data=np.zeros(adf.shape).astype(int), index=dt_index, columns=gdf.index)
        first = False

    source = os.path.normpath(csv_dir).split(os.sep)

    for yr in tqdm(rs_years, total=len(rs_years), desc=f'Processing data from {source[-2]}'):

        dt_index = pd.date_range('{}-01-01'.format(yr), '{}-12-31'.format(yr), freq='D')

        df = pd.DataFrame(index=dt_index, columns=gdf.index)
        ct = pd.DataFrame(index=dt_index, columns=gdf.index)

        if footprint_spec is None:
            yr_files = [f for f in file_list if f'_{yr}' in f]
        else:
            yr_files = [f for f in file_list if  f'_{yr}' in f and f'_p{footprint_spec}' in f]

        if len(yr_files) > 0:
            for f in yr_files:
                field = pd.read_csv(f)
                sid = field.columns[0]

                if sid not in gdf.index:
                    continue

                if select and sid not in select:
                    continue

                cols = [c for c in field.columns if len(c.split('_')) == 3]
                f_idx = [c.split('_')[-1] for c in cols]
                f_idx = [pd.to_datetime(i) for i in f_idx]
                field = pd.DataFrame(columns=[sid], data=field[cols].values.T, index=f_idx)

                field = field.replace([0.0], np.nan)
                field = field.dropna()

                duplicates = field[field.index.duplicated(keep=False)]
                if not duplicates.empty:
                    field = field.resample('D').max()

                field = field.sort_index()

                valid_indices = field.dropna().index
                diffs = valid_indices.to_series().diff().dt.days
                consecutive_days = diffs[diffs == 1].index

                for day in consecutive_days:
                    prev_day = day - pd.Timedelta(days=1)
                    if prev_day in field.index:
                        if field.loc[prev_day, sid] > field.loc[day, sid]:
                            field.loc[day, sid] = np.nan
                        else:
                            field.loc[prev_day, sid] = np.nan

                field = field.sort_index()
                field[field[sid] < 0.05] = np.nan

                df.loc[field.index, sid] = field[sid]

                ct.loc[f_idx, sid] = ~pd.isna(field[sid])

            if prev_df is not None and df.loc[f'{yr}-01'].isna().all().any():
                df.loc[f'{yr}-01-01'] = prev_df.loc[f'{yr - 1}-12-31']

            df = df.replace(0.0, np.nan)
            df = df.astype(float).interpolate()
            df = df.bfill()

            ct = ct.astype(float)
            ct = ct.fillna(0.0)
            ct = ct.astype(int)

        if first:
            adf = df.copy()
            ctdf = ct.copy()
            first = False
        else:
            adf = pd.concat([adf, df], axis=0, ignore_index=False, sort=True)
            ctdf = pd.concat([ctdf, ct], axis=0, ignore_index=False, sort=True)

        prev_df = df.copy()

    adf.to_csv(out_csv)
    print(f'count {df.count().sum()} of {df.size}, mean ({df.values[~np.isnan(df.values)].mean():.2f})')
    ctdf.to_csv(out_csv_ct)


def clustered_landsat_time_series(in_shp, csv_dir, years, out_csv, out_csv_ct, feature_id='FID'):
    """
    Intended to process Earth Engine extracts of buffered point data, e.g., the area around flux tower
    stations. See e.g., ndvi_export.clustered_field_ndvi() to generate such data. The output of this function
    should be the same format and structure as that from landsat_time_series_image() and
    landsat_time_series_station().
    :param feature_id:
    :param in_shp:
    :param csv_dir:
    :param years:
    :param out_csv:
    :param out_csv_ct:
    :return:
    """
    gdf = gpd.read_file(in_shp)
    gdf.index = gdf[feature_id]

    print(csv_dir)

    adf, ctdf, first = None, None, True

    for yr in tqdm(years, desc='Processing Landsat Time Series', total=len(years)):

        # if yr != 2007:
        #     continue

        dt_index = pd.date_range('{}-01-01'.format(yr), '{}-12-31'.format(yr), freq='D')

        try:
            f = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir) if
                 x.endswith('.csv') and '_{}'.format(yr) in x][0]
        except IndexError as e:
            print(e, yr)
            continue

        field = pd.read_csv(f)
        field.index = field[feature_id]
        cols = [c for c in field.columns if len(c.split('_')) == 3]
        f_idx = [c.split('_')[-1] for c in cols]
        f_idx = [pd.to_datetime(i) for i in f_idx]
        field = pd.DataFrame(columns=field.index, data=field[cols].values.T, index=f_idx)
        duplicates = field[field.index.duplicated(keep=False)]
        if not duplicates.empty:
            field = field.resample('D').max()
        field = field.sort_index()

        # field = field[['043_000128']]

        field[field.values == 0.00] = np.nan

        # for both NDVI and ETf, values in agriculture and the vegetated land surface generally,
        # should not go below about 0.01
        # captures of these low values are likely small pixel samples on SLC OFF Landsat 7 or
        # on bad polygons that include water or some other land cover we don't want to use
        # see e.g., https://code.earthengine.google.com/5ea8bc8c6134845a8c0c81a4cdb99fc0
        # TODO: examine these thresholds, prob better to extract pixel count to filter data

        diff_back = field.diff().values
        field = pd.DataFrame(index=field.index, columns=field.columns,
                             data=np.where(diff_back < -0.1, np.nan, field.values))

        diff_for = field.shift(periods=2).diff()
        diff_for = diff_for.shift(periods=-3).values
        field = pd.DataFrame(index=field.index, columns=field.columns,
                             data=np.where(diff_for > 0.1, np.nan, field.values))

        if 'etf' in csv_dir:
            field[field.values < 0.2] = np.nan

        if 'ndvi' in csv_dir:
            field[field.values < 0.2] = np.nan

        df = field.copy()
        df = df.astype(float).interpolate()
        df = df.reindex(dt_index)

        ct = ~pd.isna(df)

        df = df.interpolate().bfill()
        df = df.interpolate().ffill()

        ct = ct.reindex(dt_index)
        ct = ct.fillna(0)
        ct = ct.astype(int)

        if first:
            adf = df.copy()
            ctdf = ct.copy()
            first = False
        else:
            adf = pd.concat([adf, df], axis=0, ignore_index=False, sort=True)
            ctdf = pd.concat([ctdf, ct], axis=0, ignore_index=False, sort=True)

    adf.to_csv(out_csv)
    ctdf.to_csv(out_csv_ct)


def join_remote_sensing(files, dst):

    first = True

    for f in files:

        param = os.path.basename(f).split('.')[0]

        if first:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            cols = ['{}_{}'.format(c, param) for c in df.columns]
            df.columns = cols
            first = False
            print(param)
        else:
            csv = pd.read_csv(f, index_col=0, parse_dates=True)
            cols = ['{}_{}'.format(c, param) for c in csv.columns]
            csv.columns = cols
            if np.all(np.isnan(csv.values)):
                raise ValueError
            df = pd.concat([csv, df], axis=1)
            print(param)

    df.to_csv(dst)


def get_tif_list(tif_dir, year):
    """ Pass list in place of tif_dir optionally """
    l = [os.path.join(tif_dir, x) for x in os.listdir(tif_dir) if
         x.endswith('.tif') and '_{}'.format(year) in x]
    dt_str = [f[-12:-4] for f in l]
    dates_ = [pd.to_datetime(d, format='%Y%m%d') for d in dt_str]
    tup_ = sorted([(f, d) for f, d in zip(l, dates_)], key=lambda x: x[1])
    l, dates_ = [t[0] for t in tup_], [t[1] for t in tup_]
    return l, dates_


if __name__ == '__main__':

    project = '4_Flux_Network'
    # project = '5_Flux_Ensemble'

    root = '/data/ssd2/swim'
    data = os.path.join(root, project, 'data')
    if not os.path.isdir(root):
        root = '/home/dgketchum/PycharmProjects/swim-rs'
        data = os.path.join(root, 'tutorials', project, 'data')

    shapefile_path = os.path.join(data, 'gis', 'flux_fields.shp')

    # input properties files
    irr = os.path.join(data, 'properties', 'calibration_irr.csv')
    ssurgo = os.path.join(data, 'properties', 'calibration_ssurgo.csv')

    landsat = os.path.join(data, 'landsat')
    extracts = os.path.join(landsat, 'extracts')
    tables = os.path.join(landsat, 'tables')

    FEATURE_ID = 'field_1'
    selected_feature = None

    models = ['ptjpl', 'eemetric', 'openet', 'geesebal', 'sims', 'disalexi', 'ssebop']
    # models = ['ssebop']

    types_ = ['irr', 'inv_irr']
    sensing_params = ['etf', 'ndvi']

    station_file = os.path.join(data, 'station_metadata.csv')

    # use 'western_only' for sites with OpenET coverage
    sites_ = get_flux_sites(station_file, crop_only=False, western_only=False, return_df=False)
    rs_files = []

    for mask_type in types_:

        for sensing_param in sensing_params:

            yrs = [x for x in range(1987, 2025)]

            if sensing_param == 'etf':

                for model in models:
                    ee_data = os.path.join(landsat, 'extracts', f'{model}_{sensing_param}', mask_type)
                    src = os.path.join(tables, '{}_{}_{}.csv'.format(model, sensing_param, mask_type))
                    src_ct = os.path.join(tables, '{}_{}_{}_ct.csv'.format(model, sensing_param, mask_type))
                    rs_files.extend([src, src_ct])
                    sparse_landsat_time_series(shapefile_path, ee_data, yrs, src, src_ct,
                                               feature_id=FEATURE_ID, select=sites_)
            else:
                ee_data = os.path.join(landsat, 'extracts', sensing_param, mask_type)
                src = os.path.join(tables, '{}_{}.csv'.format(sensing_param, mask_type))
                src_ct = os.path.join(tables, '{}_{}_ct.csv'.format(sensing_param, mask_type))
                rs_files.extend([src, src_ct])
                # TODO: consider whether there is a case where ETf needs to be interpolated
                sparse_landsat_time_series(shapefile_path, ee_data, yrs, src, src_ct,
                                           feature_id=FEATURE_ID, select=sites_)


    remote_sensing_file = os.path.join(landsat, 'remote_sensing.csv')
    join_remote_sensing(rs_files, remote_sensing_file)

# ========================= EOF ================================================================================
