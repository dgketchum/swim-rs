import os

import numpy as np
import pandas as pd
from pyemu import Pst, ObservationEnsemble

import matplotlib.pyplot as plt


def plot_tseries_ensembles(pst_dir, glob='tongue', target=1779):
    pst = Pst(os.path.join(pst_dir, '{}.pst'.format(glob)))

    def reduce_obs_ens(obj):
        obj = pd.DataFrame(obj.T.values, obj.T.index)
        obj['time'] = [int(i.split(':')[-2].split('_')[0]) for i in obj.index]
        obj = obj.loc[[i for i, t in obj['time'].items() if t in list(range(6210, 6575))]]
        obj['oname'] = [i.split(':')[1].split('_')[1] for i in obj.index]
        obj = obj.loc[[i for i in obj.index if str(target) in i]]
        return obj

    pr_oe = ObservationEnsemble.from_csv(pst=pst, filename=os.path.join(pst_dir, '{}.0.obs.csv'.format(glob)))
    pr_oe = reduce_obs_ens(pr_oe)

    noptmax = pst.control_data.noptmax
    pt_oe = ObservationEnsemble.from_csv(pst=pst, filename=os.path.join(pst_dir, '{}.{}.obs.csv'.format(glob, noptmax)))
    pt_oe = reduce_obs_ens(pt_oe)

    noise = ObservationEnsemble.from_csv(pst=pst, filename=os.path.join(pst_dir, '{}.obs+noise.csv'.format(glob)))
    noise = reduce_obs_ens(noise)

    obs = pst.observation_data.copy()
    obs['time'] = [int(i.split(':')[-2].split('_')[0]) for i in obs.index]
    obs = obs.loc[[i for i, t in obs['time'].items() if t in list(range(6210, 6575))]]
    obs = obs.loc[[i for i in obs.index if str(target) in i]]
    obs['oname'] = [i.split(':')[1].split('_')[1] for i in obs.index]
    obs['obgnme'] = obs['oname'].copy()
    ogs = obs.obgnme.unique()
    fig, axes = plt.subplots(len(ogs), 1, figsize=(10, 2 * len(ogs)))
    ogs.sort()

    # for each observation group (i.e. timeseries)
    for ax, og in zip(axes, ogs):
        # get values for x axis
        oobs = obs.loc[obs.obgnme == og, :].copy()
        oobs.loc[:, 'time'] = oobs.time.astype(float)
        oobs.sort_values(by='time', inplace=True)
        tvals = oobs.time.values
        onames = oobs.obsnme.values
        # plot prior
        [ax.plot(tvals, pr_oe.loc[pr_oe.oname == og, i].values, '0.5', lw=0.5, alpha=0.5, label='prior')
         for i in range(0, 30)]
        # plot posterior
        [ax.plot(tvals, pt_oe.loc[pr_oe.oname == og, i].values, 'b', lw=0.5, alpha=0.5, label='posterior')
         for i in range(0, 30)]
        # plot measured+noise 
        # oobs = oobs.loc[oobs.weight > 0, :]
        tvals = oobs.time.values
        onames = oobs.obsnme.values
        [ax.plot(tvals, noise.loc[noise.oname == og, i].values, 'r', lw=0.5, alpha=0.5, label='obs + noise')
         for i in range(0, 30)]
        ax.plot(oobs.time, oobs.obsval, 'r-', lw=2)
        ax.set_title(og, loc='left')

        if og == 'swe':
            ax.set_ylim(pt_oe[[i for i in range(0, 30)]].values.min(), pt_oe[[i for i in range(0, 30)]].values.max())

    fig.tight_layout()

    plt.show()


def show_phi_evolution(pst_dir, glob='tongue'):
    pst = Pst(os.path.join(pst_dir, 'tongue.pst'))

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 3.5))
    # left
    ax = axes[0]
    phi = pd.read_csv(os.path.join(pst_dir, '{}.phi.actual.csv'.format(glob)), index_col=0)
    phi.index = phi.total_runs
    phi.iloc[:, 6:].apply(np.log10).plot(legend=False, lw=0.5, color='k', ax=ax)
    ax.set_title(r'Actual $\Phi$')
    ax.set_ylabel(r'log $\Phi$')
    # right
    ax = axes[-1]
    phi = pd.read_csv(os.path.join(pst_dir, '{}.phi.meas.csv'.format(glob)), index_col=0)
    phi.index = phi.total_runs
    phi.iloc[:, 6:].apply(np.log10).plot(legend=False, lw=0.2, color='r', ax=ax)
    ax.set_title(r'Measured+Noise $\Phi$')
    fig.tight_layout()
    plt.show()

    plt.figure()
    phi.iloc[-1, 6:].hist()
    plt.title(r'Final $\Phi$ Distribution')
    plt.show()

    pr_oe = ObservationEnsemble.from_csv(pst=pst, filename=os.path.join(pst_dir, '{}.0.obs.csv'.format(glob)))
    pt_oe = ObservationEnsemble.from_csv(pst=pst, filename=os.path.join(pst_dir,
                                                                        '{}.{}.obs.csv'.format(glob,
                                                                                               pst.control_data.noptmax)))

    fig, ax = plt.subplots(1, 1)
    pr_oe.phi_vector.apply(np.log10).hist(ax=ax, fc='0.5', ec='none', alpha=0.5, density=False)
    pt_oe.phi_vector.apply(np.log10).hist(ax=ax, fc='b', ec='none', alpha=0.5, density=False)
    _ = ax.set_xlabel('$log_{10}\\phi$')
    plt.show()


if __name__ == '__main__':
    data_root = '/media/research/IrrigationGIS/swim'
    if not os.path.exists(data_root):
        data_root = '/home/dgketchum/data/IrrigationGIS/swim'

    project = 'tongue'
    src = '/home/dgketchum/PycharmProjects/swim-rs'.format(project)
    d = os.path.join(src, 'examples/{}'.format(project))

    pest_dir_ = os.path.join(d, 'pest')
    pst_f = os.path.join(pest_dir_, 'tongue.pst')

    # show_phi_evolution(pest_dir_)

    plot_tseries_ensembles(pest_dir_, )
# ========================= EOF ====================================================================
