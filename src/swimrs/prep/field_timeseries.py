import json
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from swimrs.prep import COLUMN_MULTIINDEX

warnings.filterwarnings("ignore", category=FutureWarning)


def join_daily_timeseries(fields, met_dir, rs_dir, dst_dir, snow=None, overwrite=False,
                          start_date=None, end_date=None, feature_id='FID', **kwargs):
    """"""

    met_map_key, met_map_dir = None, None

    field_df = gpd.read_file(fields)
    field_df.index = field_df[feature_id]

    out_plots, bad, time_covered = [], None, False

    if snow is not None:
        with open(snow, 'r') as fp:
            snow_dct = json.load(fp)

    for fid, row in tqdm(field_df.iterrows(), total=field_df.shape[0],  desc="Processing Time Series files"):

        if 'target_fields' in kwargs:
            if kwargs['target_fields'] is None:
                pass
            elif str(fid) not in kwargs['target_fields']:
                continue

        if 'met_mapping' in kwargs:
            met_map_key = kwargs['met_mapping']

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

        if met_map_key is not None:
            target = row[met_map_key]
            met_file = os.path.join(met_dir, '{}.parquet'.format(target))
        else:
            met_file = os.path.join(met_dir, '{}.parquet'.format(fid))

        if not os.path.exists(met_file):
            print(f'{met_file} not found')
            continue

        df = pd.read_parquet(met_file).loc[start_date: end_date]
        df.index = pd.DatetimeIndex(df.index)
        time_covered = ((pd.to_datetime(start_date) >= df.index[0]) &
                        (pd.to_datetime(end_date) <= df.index[-1]))

        drop_cols = [c for c in df.columns if '.' in c]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

        if snow is not None:
            swe = pd.DataFrame.from_dict(snow_dct[fid])
            swe.index = pd.DatetimeIndex(swe['date'])
            swe = swe.sort_index()
            match_idx = [i for i in df.index if i in swe.index]
            df[(fid, 'none', 'swe', 'mm', 'snodas', 'no_mask')] = np.nan
            df.loc[match_idx, (fid, 'none', 'swe', 'mm', 'snodas', 'no_mask')] = swe['value']

        df = pd.concat([df, rsdf], axis=1, ignore_index=False)
        cols = pd.MultiIndex.from_tuples(df.columns, names=COLUMN_MULTIINDEX)
        df.columns = cols

        if start_date:
            df = df.loc[start_date:]
        if end_date:
            df = df.loc[:end_date]

        df.to_parquet(out_file)
        out_plots.append(fid)

    print(f'{len(out_plots)} fields were successfully processed')


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
