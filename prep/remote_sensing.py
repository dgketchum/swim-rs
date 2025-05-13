import os

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm


def sparse_time_series(in_shp, csv_dir, years, out_csv, out_csv_ct, feature_id='FID',
                       select=None, footprint_spec=None):
    """"""
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
            yr_files = [f for f in file_list if f'_{yr}' in f and f'_p{footprint_spec}' in f]

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


if __name__ == '__main__':
    pass
# ========================= EOF ================================================================================
