import os

import numpy as np
import pandas as pd
from pyemu import Pst, ObservationEnsemble, Ensemble

import matplotlib.pyplot as plt


def plot_obs_ts(pst):
    pst = Pst(pst)
    pst.observation_data.loc[pst.nnz_obs_names, 'observed'] = 1
    obsgrps = pst.nnz_obs_groups
    ts_obs = [i for i in pst.nnz_obs_groups if 'eta' in i]

    oe = ObservationEnsemble.from_gaussian_draw(pst=pst, num_reals=50)

    # get times and "measured values"
    nz_obs = pst.observation_data.loc[pst.nnz_obs_names, :].copy()
    nz_obs['time'] = nz_obs['time'].astype(float)
    nz_obs.sort_values(['obgnme', 'time'], inplace=True)
    nz_obs = nz_obs.loc[nz_obs['time'] > 7000]

    # to plot current model outputs
    res = pst.res.copy()
    res['time'] = pst.observation_data['time'].astype(float)
    res.sort_values(['group', 'time'], inplace=True)
    res = res.loc[res['time'] > 7000]

    for nz_group in obsgrps:
        nz_obs_group = nz_obs.loc[nz_obs.obgnme == nz_group, :]
        nz_obs_meas = res.loc[(res['group'] == nz_group) & res['weight'] != 0]

        fig, ax = plt.subplots(1, 1, figsize=(10, 2))
        # plot ensemble of time series of measured + noise
        [ax.plot(nz_obs_group.time, oe.loc[r, nz_obs_group.obsnme], color="r", lw=0.3) for r in range(0, 15)]
        # plot the envelope of noise realizations as well
        ax.fill_between(nz_obs_group.time, oe._df[nz_obs_group.obsnme].min(),
                        oe._df[nz_obs_group.obsnme].max(), color="r", zorder=0, alpha=.2)
        # plot simulated time series
        ax.plot(nz_obs_group.time, nz_obs_group.obsval, "b")
        # plot measured time series
        ax.plot(nz_obs_meas.time, nz_obs_meas.modelled, 'k', linestyle='--')
        ax.set_title(nz_group)
    plt.show()
    return


if __name__ == '__main__':
    data_root = '/media/research/IrrigationGIS/swim'
    if not os.path.exists(data_root):
        data_root = '/home/dgketchum/data/IrrigationGIS/swim'

    project = 'flux'
    pest_root = 'master'

    src = '/home/dgketchum/PycharmProjects/swim-rs'.format(project)
    d = os.path.join(src, 'examples/{}'.format(project))

    pest_dir_ = os.path.join(d, pest_root)
    pst_f = os.path.join(pest_dir_, '{}.pst'.format(project))

    plot_obs_ts(pst_f, )
# ========================= EOF ====================================================================
