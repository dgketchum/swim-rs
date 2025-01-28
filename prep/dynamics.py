import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def plot_dynamics(landsat, irr_csv, out_json, irr_threshold=0.1, select=None):
    lst = pd.read_csv(landsat, index_col=0, parse_dates=True)
    years = list(set([i.year for i in lst.index]))
    irr = pd.read_csv(irr_csv, index_col=0)
    irr.drop(columns=['LAT', 'LON'], inplace=True)

    try:
        _ = float(irr.index[0])
        irr.index = [str(i) for i in irr.index]
    except (TypeError, ValueError):
        pass

    irrigated, fields = False, {}
    for c in tqdm(irr.index, desc='Analyzing Irrigation', total=len(irr.index)):

        if select and c not in select:
            continue

        if c not in irr.index:
            print('{} not in index'.format(c))
            continue

        if np.all(np.isnan(irr.loc[c])):
            print('{} is all nan in {}'.format(c, irr_csv))
            continue

        fields[c] = {}

        selector = '{}_ndvi_irr'.format(c)

        fallow = []

        for yr in years:

            if yr > 2022:
                continue

            irr_doys, periods = [], 0

            try:
                f_irr = irr.at[c, 'irr_{}'.format(yr)]
            except (ValueError, KeyError):
                f_irr = irr.at[c, 'irr_{}'.format(yr)]

            irrigated = f_irr > irr_threshold

            if not irrigated:
                fallow.append(yr)
                fields[c][yr] = {'irr_doys': irr_doys,
                                 'irrigated': int(irrigated),
                                 'f_irr': f_irr}
                continue

            df = lst.loc['{}-01-01'.format(yr): '{}-12-31'.format(yr), [selector]]
            df['doy'] = [i.dayofyear for i in df.index]
            df[selector] = df[selector].rolling(window=10, center=True).mean()
            # df[selector] = savgol_filter(df[selector], window_length=7, polyorder=2)

            df['diff'] = df[selector].diff()

            nan_ct = np.count_nonzero(np.isnan(df[selector].values))
            if nan_ct > 200:
                # print('{}: {} has {}/{} nan'.format(c, yr, nan_ct, df.shape[0]))
                fallow.append(yr)
                continue

            local_min_indices = df[(df['diff'] > 0) & (df['diff'].shift(1) < 0)].index

            # Find periods with positive slope for more than 10 days
            positive_slope = (df['diff'] > 0)
            groups = (positive_slope != positive_slope.shift()).cumsum()
            df['groups'] = groups
            group_counts = positive_slope.groupby(groups).sum()
            long_positive_slope_groups = group_counts[group_counts >= 10].index

            for group in long_positive_slope_groups:
                # Find the start and end indices of the group
                group_indices = positive_slope[groups == group].index
                start_index = group_indices[0]
                end_index = group_indices[-1]

                # Check if this group follows a local minimum
                if start_index in local_min_indices:
                    start_doy = (start_index - pd.Timedelta(days=5)).dayofyear
                    end_doy = (end_index - pd.Timedelta(days=5)).dayofyear
                    irr_doys.extend(range(start_doy, end_doy + 1))
                    periods += 1

                else:
                    # Otherwise, just add the days within the group
                    start_doy = start_index.dayofyear
                    end_doy = (end_index - pd.Timedelta(days=5)).dayofyear
                    irr_doys.extend(range(start_doy, end_doy + 1))
                    periods += 1

            irr_doys = sorted(list(set(irr_doys)))

            fields[c][yr] = {'irr_doys': irr_doys,
                             'irrigated': int(irrigated),
                             'f_irr': f_irr}

        fields[c]['fallow_years'] = fallow

    with open(out_json, 'w') as fp:
        print('wrote {}'.format(out_json))
        json.dump(fields, fp, indent=4)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
