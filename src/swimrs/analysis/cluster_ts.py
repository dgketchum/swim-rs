import json
import os
import random

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import StrMethodFormatter


def cluster_ndvi(csv_dir, out_json, fig_dir, sample_n=1000):
    l = [os.path.join(csv_dir, _file) for _file in os.listdir(csv_dir)]
    random.shuffle(l)
    dct = {}

    for ndvi in ['ndvi_irr', 'ndvi_inv_irr']:
        first = True
        for csv in l:
            df = pd.read_csv(csv, index_col='date', parse_dates=True)
            df = df[[ndvi]]
            name = df.columns[0]
            df['date'] = df.index
            df['year'] = df.date.dt.year
            df['date'] = df.date.dt.strftime('%m-%d')
            df.index = [x for x in range(0, df.shape[0])]
            ydf = df.set_index(['year', 'date'])[name].unstack(-2)
            ydf.columns = ['{}_{}'.format(name, c) for c in ydf.columns]
            ydf.dropna(axis=1, how='all', inplace=True)
            ydf.dropna(axis=0, how='any', inplace=True)
            if first:
                mdf = ydf.copy()
                first = False
            else:
                mdf = pd.concat([mdf, ydf], axis=1)

            if len(mdf.columns) > 5000:
                break

        colors = ['k' for _ in mdf.columns]
        years = list(set([int(c.split('_')[-1]) for c in mdf.columns]))
        ax = mdf.plot(logy=False, legend=False, alpha=0.1, color=colors, ylabel='NDVI',
                      title='{} - {}'.format(years[0], years[-1]), figsize=(30, 10))

        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))

        median_ = pd.DataFrame(mdf.median(axis=1))
        median_.columns = ['Median Daily NDVI']
        median_.plot(logy=False, legend=True, color='b', ax=ax)

        plt.ylim([0.2, 0.95])
        plt.savefig(os.path.join(fig_dir, 'stacked_{}'.format(ndvi)))
        # plt.show()

        dct[ndvi] = list(median_.values.flatten())

    with open(out_json, 'w') as f:
        json.dump(dct, f)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/swim'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/swim'

    project = 'tongue'
    data = os.path.join(d, 'examples', project)

    f_ = os.path.join(data, 'plot_timeseries')
    c = os.path.join(data, 'ndvi_signals', 'plots')
    j = os.path.join(data, 'ndvi_signals', 'median_ts.json')
    cluster_ndvi(f_, j, c, sample_n=15000)
# ========================= EOF ====================================================================
