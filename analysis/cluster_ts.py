import numpy
import matplotlib.pyplot as plt

from tslearn.clustering import KShape
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def cluster_growing_seasons(_dir, plot_fig):
    seed = 0
    numpy.random.seed(seed)
    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
    # Keep first 3 classes and 50 first time series
    X_train = X_train[y_train < 4]
    X_train = X_train[:50]
    numpy.random.shuffle(X_train)
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
    pass
# ========================= EOF ====================================================================
