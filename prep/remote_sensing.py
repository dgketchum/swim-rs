import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from prep import COLUMN_MULTIINDEX


def sparse_time_series(in_shp, csv_dir, years, out_pqt, feature_id='FID', instrument='landsat', parameter='ndvi',
                       algorithm='none', mask='no_mask', select=None, footprint_spec=None):
    """"""
    gdf = gpd.read_file(in_shp)
    gdf.index = gdf[feature_id]

    if select:
        dummy = select.copy()
        select = [s for s in dummy if s in gdf.index]
        dropped = [s for s in dummy if s not in select]
        print(f'dropping {dropped} not found in index')
        gdf = gdf.loc[select]

    if footprint_spec:
        gdf = gdf[gdf['grid_size'] == footprint_spec]

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

    target_columns = [(sid, instrument, parameter, 'unitless', algorithm, mask) for sid in gdf.index]
    target_columns = pd.MultiIndex.from_tuples(target_columns,
                                               names=COLUMN_MULTIINDEX)

    if len(empty_yrs) > 0:
        print(f'{len(empty_yrs)} years without remote sensing data, {empty_yrs[0]} to {empty_yrs[-1]}')
        dt_index = pd.date_range('{}-01-01'.format(empty_yrs[0]), '{}-12-31'.format(empty_yrs[-1]), freq='D')
        adf = pd.DataFrame(data=np.zeros((len(dt_index), len(gdf.index))) * np.nan,
                           index=dt_index, columns=target_columns)
        first = False

    source = os.path.normpath(csv_dir).split(os.sep)

    for yr in tqdm(rs_years, total=len(rs_years), desc=f'Processing data from {source[-2]}'):

        dt_index = pd.date_range('{}-01-01'.format(yr), '{}-12-31'.format(yr), freq='D')

        data_series_list_for_year = []

        if footprint_spec is None:
            yr_files = [f for f in file_list if f'_{yr}' in f]
        else:
            yr_files = [f for f in file_list if f'_{yr}' in f and f'_p{footprint_spec}' in f]

        if len(yr_files) > 0:
            for f in yr_files:
                field_data_csv = pd.read_csv(f)
                sid = field_data_csv.columns[0]

                if sid not in gdf.index:
                    continue

                if select and sid not in select:
                    continue

                target_col = (sid, instrument, parameter, 'unitless', algorithm, mask)

                cols = [c for c in field_data_csv.columns if len(c.split('_')) == 3]

                if instrument == 'landsat':
                    f_idx_dt = [c.split('_')[-1] for c in cols]
                    f_idx_dt = [pd.to_datetime(i) for i in f_idx_dt]
                    f_idx = [i for i in f_idx_dt if i in dt_index]
                elif instrument == 'sentinel':
                    f_idx_dt = [c[:8] for c in cols]
                    f_idx_dt = [pd.to_datetime(i) for i in f_idx_dt]
                    f_idx = [i for i in f_idx_dt if i in dt_index]
                else:
                    raise ValueError

                if len(f_idx) != len(f_idx_dt):
                    out_dates = [i for i in f_idx_dt if i not in f_idx]
                    out_index = [f_idx_dt.index(i) for i in out_dates]
                    cols = [c for e, c in enumerate(cols) if e not in out_index]
                    out_datestrs = [datetime.strftime(i, '%Y%m%d') for i in f_idx_dt if i not in f_idx]
                    print(f'\nData falls outside {yr}: {out_datestrs}')

                field = pd.DataFrame(columns=[sid], data=field_data_csv[cols].values.T, index=f_idx)

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

                data_s = pd.Series(np.nan, index=dt_index, name=target_col)
                data_s.loc[field.index] = field[sid]
                data_series_list_for_year.append(data_s)

        if data_series_list_for_year:
            df = pd.concat(data_series_list_for_year, axis=1)
            df = df.reindex(columns=target_columns)
        else:
            df = pd.DataFrame(np.nan, index=dt_index, columns=target_columns)

        if first:
            adf = df.copy()
            first = False
        else:
            if adf is None:
                adf = df.copy()
            else:
                adf = pd.concat([adf, df], axis=0, ignore_index=False)

    if adf is not None:
        adf = adf.sort_index()
        adf = adf.dropna(how='all', axis=1)
        adf.to_parquet(out_pqt, engine='pyarrow')
        print(f'wrote {out_pqt} with {adf.shape[1]} columns from {adf.index[0]} to {adf.index[-1]}')
    else:
        print(f'No data processed for {out_pqt}')


def join_remote_sensing(files, dst, station_selection='exclusive'):
    """"""
    dfs = [pd.read_parquet(f) for f in files]
    sids_list = [{c[0] for c in df.columns} for df in dfs]
    if station_selection == 'exclusive':
        common_sids = sids_list[0].intersection(*sids_list[1:])
    elif station_selection == 'inclusive':
        common_sids = sids_list[0].union(*sids_list[1:])
    else:
        raise ValueError("Must choose 'inclusive' or 'exclusive'")

    list_of_indices_same = [df.index for df in dfs]
    assert all(idx.equals(list_of_indices_same[0]) for idx in list_of_indices_same)

    valid_indices = [df.index for df in dfs if isinstance(df.index, pd.DatetimeIndex) and not df.index.empty]

    min_date = min(idx.min() for idx in valid_indices)
    max_date = max(idx.max() for idx in valid_indices)

    new_daily_index = pd.date_range(start=min_date, end=max_date, freq='D', name='datetime')

    success_ct = 0
    for sid in common_sids:

        if sid == 'S2':
            b = 1

        target_cols = [col for df in dfs for col in df.columns if col[0] == sid]
        target_cols = pd.MultiIndex.from_tuples(target_cols, names=COLUMN_MULTIINDEX)
        target_series = [df[col].values for df in dfs for col in df.columns if col[0] == sid]
        site_df = pd.DataFrame(data=np.array(target_series).T, index=new_daily_index, columns=target_cols)

        if not site_df.columns.empty:
            output_filename = f"{sid}.parquet"
            output_path = os.path.join(dst, output_filename)
            site_df.to_parquet(output_path)
            print(f'{sid} ({site_df.shape}) to {output_path}')
            success_ct += 1

    print(f'{success_ct} joined remote sensing files written to {dst}')


if __name__ == '__main__':
    pass
# ========================= EOF ================================================================================
