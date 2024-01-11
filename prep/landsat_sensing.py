import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd

from rasterstats import zonal_stats
from detecta import detect_cusum, detect_peaks, detect_onset


def landsat_time_series_image(in_shp, tif_dir, years, out_csv, out_csv_ct, min_ct=100):
    gdf = gpd.read_file(in_shp)
    gdf.index = gdf['FID']

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


def landsat_time_series_table(in_shp, csv_dir, years, out_csv, out_csv_ct):
    gdf = gpd.read_file(in_shp)
    gdf.index = gdf['FID']

    print(csv_dir)

    adf, ctdf, first = None, None, True

    for yr in years:

        dt_index = pd.date_range('{}-01-01'.format(yr), '{}-12-31'.format(yr), freq='D')
        df = pd.DataFrame(index=dt_index, columns=gdf.index)
        ct = pd.DataFrame(index=dt_index, columns=gdf.index)

        file_list = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir) if
         x.endswith('.csv') and '_{}'.format(yr) in x]

        for f in file_list:
            field = pd.read_csv(f)
            sid = field.columns[0]
            cols = [c for c in field.columns if len(c.split('_')) == 3]
            f_idx = [c.split('_')[-1] for c in cols]
            f_idx = [pd.to_datetime(i) for i in f_idx]
            field = pd.DataFrame(columns=[sid], data=field[cols].values.T, index=f_idx)
            duplicates = field[field.index.duplicated(keep=False)]
            if not duplicates.empty:
                field = field.resample('D').mean()
            field = field.sort_index()

            if 'etf' in csv_dir:
                field[field[sid] < 0.01] = np.nan

            df.loc[field.index, sid] = field[sid]

            ct.loc[f_idx, sid] = ~pd.isna(field[sid])

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
        print(yr)

    adf.to_csv(out_csv)
    ctdf.to_csv(out_csv_ct)


def join_remote_sensing(_dir, dst):
    l = [os.path.join(_dir, f) for f in os.listdir(_dir) if f.endswith('.csv')]
    first = True

    params = ['etf_inv_irr',
              'ndvi_inv_irr',
              'etf_irr',
              'ndvi_irr']

    params += ['{}_ct'.format(p) for p in params]

    for f in l:
        param = [p for p in params if p in os.path.basename(f)]
        if len(param) > 1:
            param = param[1]
        else:
            param = param[0]

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


def detect_cuttings(landsat, irr_csv, out_json, irr_threshold=0.1):
    lst = pd.read_csv(landsat, index_col=0, parse_dates=True)
    cols = list(set([x.split('_')[0] for x in lst.columns]))
    years = list(set([i.year for i in lst.index]))
    irr = pd.read_csv(irr_csv, index_col=0)
    irr.drop(columns=['LAT', 'LON'], inplace=True)

    irrigated, fields = False, {c: {} for c in cols}
    for c in cols:

        selector = '{}_ndvi_irr'.format(c)
        print('\n', c, selector)
        count, fallow = [], []

        if c not in ['US-Mj1', 'US-Mj2']:
            continue

        for yr in years:

            try:
                f_irr = irr.at[int(c), 'irr_{}'.format(yr)]
            except ValueError:
                f_irr = irr.at[c, 'irr_{}'.format(yr)]

            irrigated = f_irr > irr_threshold

            if not irrigated:
                fallow.append(yr)
                continue

            df = lst.loc['{}-01-01'.format(yr): '{}-12-31'.format(yr), [selector]]
            diff = df.diff()

            nan_ct = np.count_nonzero(np.isnan(df.values))
            if nan_ct > 200:
                print('{}: {} has {}/{} nan'.format(c, yr, nan_ct, df.values.size))
                fallow.append(yr)
                continue

            vals = df.values

            try:
                peaks = detect_peaks(vals.flatten(), mph=0.500, mpd=30, threshold=0, valley=False,
                                     show=False)

                ta, tai, _, _ = detect_cusum(vals, threshold=0.100, ending=False, show=False,
                                             drift=0.005)

                onsets = detect_onset(vals, threshold=0.550, show=False)

            except ValueError:
                print('Error', yr, c)
                continue

            irr_doys = []
            green_start_dates, cut_dates = [], []
            green_start_doys, cut_doys = [], []
            irr_dates, cut_dates, pk_dates = [], [], []

            if irrigated:
                for infl, green in zip(ta, tai):

                    off_peak = False
                    if np.all(~np.array([ons[0] < green < ons[1] for ons in onsets])):
                        off_peak = True

                    if not off_peak:
                        continue

                    sign = diff.loc[diff.index[green + 1]: diff.index[green + 10], selector].mean()

                    if sign > 0:
                        date = df.index[green]
                        green_start_doys.append(date)
                        dts = '{}-{:02d}-{:02d}'.format(date.year, date.month, date.day)
                        green_start_dates.append(dts)

                for pk in peaks:

                    on_peak = False
                    if np.any(np.array([ons[0] < pk < ons[1] for ons in onsets])):
                        on_peak = True

                    if on_peak:
                        date = df.index[pk]
                        cut_doys.append(date)
                        dts = '{}-{:02d}-{:02d}'.format(date.year, date.month, date.day)
                        cut_dates.append(dts)

                irr_doys = [[i for i in range(s.dayofyear, e.dayofyear)] for s, e in zip(green_start_doys, cut_doys)]
                irr_doys = list(np.array(irr_doys, dtype=object).flatten())
                irr_windows = [(gu, cd) for gu, cd in zip(green_start_dates, cut_dates)]

                if not irr_windows:
                    # this dense code calculates the periods when NDVI is increasing
                    roll = pd.DataFrame((diff.rolling(window=15).mean() > 0.0), columns=[selector])
                    roll = roll.loc[[i for i in roll.index if 3 < i.month < 11]]
                    roll['crossing'] = (roll[selector] != roll[selector].shift()).cumsum()
                    roll['count'] = roll.groupby([selector, 'crossing']).cumcount(ascending=True)
                    irr_doys = [i.dayofyear for i in roll[roll[selector]].index]
                    roll = roll[(roll['count'] == 0 & roll[selector])]
                    start_idx, end_idx = list(roll.loc[roll[selector] == 1].index), list(roll.loc[roll[selector] == 0].index)
                    start_idx = ['{}-{:02d}-{:02d}'.format(d.year, d.month, d.day) for d in start_idx]
                    end_idx = ['{}-{:02d}-{:02d}'.format(d.year, d.month, d.day) for d in end_idx]
                    irr_windows = [(s, e) for s, e in zip(start_idx, end_idx)]

            else:
                irr_windows = []

            count.append(len(pk_dates))

            green_start_dates = list(np.unique(np.array(green_start_dates)))

            fields[c][yr] = {'pk_count': len(pk_dates),
                             'green_ups': green_start_dates,
                             'cut_dates': cut_dates,
                             'irr_windows': irr_windows,
                             'irr_doys': irr_doys,
                             'irrigated': int(irrigated),
                             'f_irr': f_irr}

        avg_ct = np.array(count).mean()
        fields[c]['average_cuttings'] = float(avg_ct)
        fields[c]['fallow_years'] = fallow

    with open(out_json, 'w') as fp:
        json.dump(fields, fp, indent=4)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/et-demands'
    project = 'flux'
    dtype = 'extracts'
    project_ws = os.path.join(d, 'examples', project)
    tables = os.path.join(project_ws, 'landsat', 'tables')

    targets_ = ['US-FPe', 'US-Mj1', 'US-Mj2']

    types_ = ['inv_irr', 'irr']
    sensing_params = ['ndvi']

    for mask_type in types_:

        for sensing_param in sensing_params:

            yrs = [x for x in range(2000, 2021)]
            shp = os.path.join(project_ws, 'gis', '{}_fields_sample.shp'.format(project))

            ee_data, src = None, None

            ee_data = os.path.join(project_ws, 'landsat', dtype, sensing_param, mask_type)
            src = os.path.join(tables, '{}_{}_{}_sample.csv'.format(project, sensing_param, mask_type))
            src_ct = os.path.join(tables, '{}_{}_{}_ct_sample.csv'.format(project, sensing_param, mask_type))

            # landsat_time_series_table(shp, ee_data, yrs, src, src_ct)
            # landsat_time_series_image(shp, tif, yrs, src, src_ct)

    dst_ = os.path.join(project_ws, 'landsat', '{}_sensing_sample.csv'.format(project))
    join_remote_sensing(tables, dst_)

    irr_ = os.path.join(project_ws, 'properties', '{}_sample_irr.csv'.format(project))
    js_ = os.path.join(project_ws, 'landsat', '{}_cuttings.json'.format(project))
    detect_cuttings(dst_, irr_, irr_threshold=0.1, out_json=js_)

# ========================= EOF ================================================================================
