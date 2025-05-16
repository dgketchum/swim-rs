import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from prep import COLUMN_MULTIINDEX


def sparse_time_series(in_shp, csv_dir, years, out_pqt, out_pqt_ct, feature_id='FID',
                       instrument='landsat', parameter='ndvi', algorithm='None', mask='no_mask',
                       interoplate=False, select=None, footprint_spec=None):
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
    rs_years = [y for y in rs_years if y in years]
    print(f'Processing years {rs_years[0]} to {rs_years[-1]}')

    empty_yrs = [y for y in years if y not in rs_years]

    target_columns = [(sid, instrument, parameter, algorithm, 'value', mask) for sid in gdf.index]
    target_columns = pd.MultiIndex.from_tuples(target_columns,
                                               names=COLUMN_MULTIINDEX)
    target_ct_columns = [(sid, instrument, parameter, algorithm, 'obs_flag', mask) for sid in gdf.index]
    target_ct_columns = pd.MultiIndex.from_tuples(target_ct_columns, names=COLUMN_MULTIINDEX)

    if len(empty_yrs) > 0:
        print(f'{len(empty_yrs)} years without remote sensing data, {empty_yrs[0]} to {empty_yrs[-1]}')
        dt_index = pd.date_range('{}-01-01'.format(empty_yrs[0]), '{}-12-31'.format(empty_yrs[-1]), freq='D')
        adf = pd.DataFrame(data=np.zeros((len(dt_index), len(gdf.index))) * np.nan,
                           index=dt_index, columns=target_columns)
        if interoplate:
            ctdf = pd.DataFrame(data=np.zeros(adf.shape).astype(int), index=dt_index, columns=target_ct_columns)

        first = False

    source = os.path.normpath(csv_dir).split(os.sep)

    for yr in tqdm(rs_years, total=len(rs_years), desc=f'Processing data from {source[-2]}'):

        dt_index = pd.date_range('{}-01-01'.format(yr), '{}-12-31'.format(yr), freq='D')

        df = pd.DataFrame(index=dt_index, columns=target_columns)
        if interoplate:
            ct = pd.DataFrame(index=dt_index, columns=target_ct_columns)

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

                target_col = (sid, instrument, parameter, algorithm, 'value', mask)

                # dicey
                cols = [c for c in field.columns if len(c.split('_')) == 3]

                if instrument == 'landsat':
                    f_idx_dt = [c.split('_')[-1] for c in cols]
                    f_idx_dt = [pd.to_datetime(i) for i in f_idx_dt]
                    f_idx = [i for i in f_idx_dt if i in df.index]
                elif instrument == 'sentinel':
                    f_idx_dt = [c[:8] for c in cols]
                    f_idx_dt = [pd.to_datetime(i) for i in f_idx_dt]
                    f_idx = [i for i in f_idx_dt if i in df.index]
                else:
                    raise ValueError

                if len(f_idx) != len(f_idx_dt):
                    out_dates = [i for i in f_idx_dt if i not in f_idx]
                    out_index = [f_idx_dt.index(i) for i in out_dates]
                    cols = [c for e, c in enumerate(cols) if e not in out_index]
                    out_datestrs = [datetime.strftime(i, '%Y%m%d') for i in f_idx_dt if i not in f_idx]
                    print(f'\nData falls outside {yr}: {out_datestrs}')

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

                df.loc[field.index, target_col] = field[sid]

                if interoplate:
                    target_ct_col = (sid, instrument, parameter, algorithm, 'obs_flag', mask)
                    ct.loc[f_idx, target_ct_col] = ~pd.isna(field[sid])

            if prev_df is not None and df.loc[f'{yr}-01'].isna().all().any() and interoplate:
                df.loc[f'{yr}-01-01'] = prev_df.loc[f'{yr - 1}-12-31']

            df = df.replace(0.0, np.nan)
            if interoplate:
                # only necessary to have interpolated daily values for NDVI in SWIM
                df = df.astype(float).interpolate()
                df = df.bfill()

                ct = ct.astype(float)
                ct = ct.fillna(0.0)
                ct = ct.astype(int)

        if first:
            adf = df.copy()

            if interoplate:
                ctdf = ct.copy()

            first = False

        else:
            adf = pd.concat([adf, df], axis=0, ignore_index=False, sort=True)

            if interoplate:
                ctdf = pd.concat([ctdf, ct], axis=0, ignore_index=False, sort=True)

        prev_df = df.copy()

    adf = adf.dropna(how='all', axis=1)
    adf.to_parquet(out_pqt, engine='pyarrow')
    print(f'wrote {out_pqt}')

    if interoplate:
        ctdf = ctdf.dropna(how='all', axis=1)
        ctdf = ctdf.astype(int)
        ctdf.to_parquet(out_pqt_ct,
                        engine='pyarrow')
        print(f'wrote {out_pqt_ct}')


def join_remote_sensing(files, dst):
    """"""
    dfs = [pd.read_parquet(f) for f in files]
    sids_list = [{c[0] for c in df.columns} for df in dfs]
    common_sids = sids_list[0].intersection(*sids_list[1:])

    valid_indices = [df.index for df in dfs if isinstance(df.index, pd.DatetimeIndex) and not df.index.empty]

    min_date = min(idx.min() for idx in valid_indices)
    max_date = max(idx.max() for idx in valid_indices)

    new_daily_index = pd.date_range(start=min_date, end=max_date, freq='D', name='datetime')

    for sid in common_sids:

        target_cols = [col for df in dfs for col in df.columns if col[0] == sid]
        target_cols = pd.MultiIndex.from_tuples(target_cols, names=COLUMN_MULTIINDEX)
        target_series =  [df[col].values for df in dfs for col in df.columns if col[0] == sid]
        site_df = pd.DataFrame(data=np.array(target_series).T, index=new_daily_index, columns=target_cols)


        if not site_df.columns.empty:
            output_filename = f"{sid}.parquet"
            output_path = os.path.join(dst, output_filename)
            site_df.to_parquet(output_path)
            print(f'{sid} ({site_df.shape}) to {output_path}')


if __name__ == '__main__':
    pass
# ========================= EOF ================================================================================
