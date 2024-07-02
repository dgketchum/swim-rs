import os
import json

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from tslearn.clustering import KShape
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def cluster_ndvi(csv_dir, out_json, fig_dir):

    l = [os.path.join(csv_dir, _file) for _file in os.listdir(csv_dir)]
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

            if len(mdf.columns) > 1000:
                break

        colors = ['k' for _ in mdf.columns]
        years = list(set([int(c.split('_')[-1]) for c in mdf.columns]))
        ax = mdf.plot(logy=False, legend=False, alpha=0.2, color=colors, ylabel='NDVI',
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


def cluster_growing_seasons(_dir, plot_fig):
    seed = 0
    np.random.seed(seed)
    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
    # Keep first 3 classes and 50 first time series
    X_train = X_train[y_train < 4]
    X_train = X_train[:50]
    np.random.shuffle(X_train)
    # For this method to operate properly, prior scaling is required
    X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
    sz = X_train.shape[1]

    # kShape clustering
    ks = KShape(n_clusters=3, verbose=True, random_state=seed)
    y_pred = ks.fit_predict(X_train)

    plt.figure()
    for yi in range(3):
        plt.subplot(3, 1, 1 + yi)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, sz)
        plt.ylim(-4, 4)
        plt.title("Cluster %d" % (yi + 1))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    f_ = '/home/dgketchum/data/IrrigationGIS/swim/examples/tongue/input_timeseries'
    c = '/home/dgketchum/data/IrrigationGIS/swim/examples/tongue/ts_cluster/plots'
    j = '/home/dgketchum/data/IrrigationGIS/swim/examples/tongue/ts_cluster/median_ts.json'
    cluster_ndvi(f_, j, c)
# ========================= EOF ====================================================================
