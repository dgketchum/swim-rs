import json
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_extraction.gridmet.gridmet import download_gridmet
from prep import COLUMN_MULTIINDEX

warnings.filterwarnings("ignore", category=FutureWarning)


def join_daily_timeseries(fields, met_dir, rs_dir, dst_dir, snow=None, overwrite=False,
                          start_date=None, end_date=None, feature_id='FID', **kwargs):
    """"""

    field_df = gpd.read_file(fields)
    field_df.index = field_df[feature_id]

    out_plots, bad, time_covered = [], None, False

    for fid, row in tqdm(field_df.iterrows(), total=field_df.shape[0]):

        if 'target_fields' in kwargs:
            if str(fid) not in kwargs['target_fields']:
                continue

        if 'met_mapping' in kwargs:
            if 'mapped_dir' not in kwargs:
                raise ValueError('Provide a mapping key and a target directory')
            met_map_key = kwargs['met_mapping']
            met_map_dir = kwargs['mapped_dir']
        else:
            met_map_key = fid
            met_map_dir = met_dir

        rs_file = os.path.join(rs_dir, f'{fid}.parquet')
        if not os.path.exists(rs_file):
            print(f'{rs_file} not found')
            continue

        rsdf = pd.read_parquet(rs_file)
        rsdf = rsdf.sort_index(axis=1)
        rs_start, rs_end = rsdf.index[0], rsdf.index[-1]

        rs_temporal_coverage = ((pd.to_datetime(start_date) >= rs_start) &
                                (pd.to_datetime(end_date) <= rs_end))

        if not rs_temporal_coverage:
            raise ValueError('Remote sensing data does not cover requested time period')

        rsdf = rsdf.loc[start_date: end_date]

        out_file = os.path.join(dst_dir, '{}.parquet'.format(fid))
        if os.path.exists(out_file) and not overwrite and time_covered:
            continue

        met_file = os.path.join(met_dir, '{}.csv'.format(met_map_key))

        if not os.path.exists(met_file):
            print(f'{met_file} not found')
            continue

        df = pd.read_csv(met_file, index_col='date', parse_dates=True).loc[start_date: end_date]
        df.index = pd.DatetimeIndex(df.index)
        time_covered = ((pd.to_datetime(start_date) >= df.index[0]) &
                        (pd.to_datetime(end_date) <= df.index[-1]))

        drop_cols = [c for c in df.columns if '.' in c]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

        # multi-index columns
        cols = [(fid, 'none', c, os.path.basename(met_dir), 'value', 'no_mask') for c in df.columns]
        cols = pd.MultiIndex.from_tuples(cols,  names=COLUMN_MULTIINDEX)
        df.columns = cols

        if snow is not None:
            with open(snow, 'r') as fp:
                snow = json.load(fp)

            swe_data = [(pd.to_datetime(d['date']), d['value']) for d in snow[fid]]
            swe = pd.Series(index=[x[0] for x in swe_data], data=[x[1] for x in swe_data])
            swe = swe.sort_index()

            match_idx = [i for i in df.index if i in swe.index]
            df.loc[match_idx, 'obs_swe'] = swe

        df = pd.concat([df, rsdf], axis=1, ignore_index=False)

        if start_date:
            df = df.loc[start_date:]
        if end_date:
            df = df.loc[:end_date]

        accept, bad = True, 0

        chkdf = df.resample('A').sum()
        for i in chkdf.index:
            for m in chkdf.columns:
                if np.isnan(chkdf.loc[i, m]):
                    print('{} in {} has only nan'.format(fid, i.year))
                    accept = False
                    bad += 1
                    break

        if accept:
            df.to_parquet(out_file)
            out_plots.append(fid)

    print(f'{len(out_plots)} fields were successfully processed')
    print(f'{bad} fields were dropped due to missing data')


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
