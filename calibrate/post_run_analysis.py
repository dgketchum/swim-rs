import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import DatetimeTickFormatter, Span
from bokeh.palettes import Category10
from bokeh.plotting import figure, output_file, save
from pyemu import Pst, ObservationEnsemble

from prep.prep_plots import FLUX_SELECT


def plot_tseries_ensembles(pst_dir, glob='tongue', targets=1779, sample_n=30, idx_start=None, idx_end=None,
                           flux_file=None, start_date='2000-01-01'):
    pst = Pst(os.path.join(pst_dir, '{}.pst'.format(glob)))

    for target in targets:
        def reduce_obs_ens(obj):
            obj = pd.DataFrame(obj.T.values, obj.T.index)
            obj['time'] = [int(i.split(':')[-2].split('_')[0]) for i in obj.index]
            if idx_start and idx_end:
                obj = obj.loc[[i for i, t in obj['time'].items() if t in list(range(idx_start, idx_end))]]
            obj['oname'] = [i.split(':')[1].split('_')[1] for i in obj.index]
            obj = obj.loc[[i for i in obj.index if str(target.lower()) in i]]
            return obj

        pr_oe = ObservationEnsemble.from_csv(pst=pst, filename=os.path.join(pst_dir, '{}.0.obs.csv'.format(glob)))
        pr_oe = reduce_obs_ens(pr_oe)

        noptmax = pst.control_data.noptmax - 1
        pt_oe = ObservationEnsemble.from_csv(pst=pst, filename=os.path.join(pst_dir,
                                                                            '{}.{}.obs.csv'.format(glob, noptmax)))
        pt_oe = reduce_obs_ens(pt_oe)

        noise = ObservationEnsemble.from_csv(pst=pst, filename=os.path.join(pst_dir, '{}.obs+noise.csv'.format(glob)))
        noise = reduce_obs_ens(noise)

        obs = pst.observation_data.copy()
        obs['time'] = [int(i.split(':')[-2].split('_')[0]) for i in obs.index]
        if idx_start and idx_end:
            obs = obs.loc[[i for i, t in obs['time'].items() if t in list(range(idx_start, idx_end))]]
        obs = obs.loc[[i for i in obs.index if str(target.lower()) in i]]
        obs['oname'] = [i.split(':')[1].split('_')[1] for i in obs.index]
        obs['obgnme'] = obs['oname'].copy()
        ogs = obs.obgnme.unique()
        ogs.sort()

        start_dt = datetime.strptime(start_date, '%Y-%m-%d')

        # last two columns of prior and posterior results are 'time' and 'oname'
        samples = np.random.choice(pt_oe.columns[:-2], sample_n, replace=False)

        plots = []
        colors = Category10[10]

        # for each observation group (i.e. timeseries)
        for og in ogs:
            # get values for x axis
            oobs = obs.loc[obs.obgnme == og, :].copy()
            oobs.loc[:, 'time'] = oobs.time.astype(float)
            oobs.sort_values(by='time', inplace=True)
            tvals = [start_dt + timedelta(days=int(t)) for t in oobs.time.values]

            p = figure(title=og, x_axis_label='Time', y_axis_label='Value', width=2400, height=800,
                       x_axis_type="datetime")

            # plot prior
            for i in samples:
                p.line(tvals, pr_oe.loc[pr_oe.oname == og, i].values, line_width=0.5, alpha=0.3, color=colors[0],
                       legend_label='Prior')

            # plot posterior
            for i in samples:
                p.line(tvals, pt_oe.loc[pr_oe.oname == og, i].values, line_width=0.5, alpha=0.5,
                       color=colors[1], legend_label='Posterior')

            p.line(tvals, oobs.obsval, line_width=2, color='red', legend_label='Observed')

            p.legend.location = "top_left"
            p.xaxis.formatter = DatetimeTickFormatter(days=["%Y-%m-%d"], months=["%Y-%m-%d"], years=["%Y-%m-%d"])
            plots.append(p)

            if flux_file is not None and og == 'eta':
                end_date = start_dt + timedelta(days=len(tvals) - 1)
                dt_index = pd.DatetimeIndex(pd.date_range(start_dt, end_date, freq='D'))
                flux_df = pd.read_csv(flux_file.format(target), index_col=0, parse_dates=True)
                flux_df = flux_df.reindex(dt_index)
                p.scatter(tvals, flux_df['ET'], size=6, color='black', legend_label='Flux Obs')
                p.line(tvals, flux_df['ET_fill'], line_width=2, color='green', legend_label='Flux Fill')

            # if og == 'eta':
            #     for time in obs.loc[obs['weight'] == 1.0, 'time']:
            #         vline = Span(location=start_dt + timedelta(days=int(time)), dimension='height', line_color='red',
            #                      line_width=1, line_dash='dashed')
            #         p.add_layout(vline)

        # Create a layout and show/save the plot
        plot_dir = os.path.join(pest_dir_, 'plots')
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)

        _fig_file = os.path.join(plot_dir, 'timeseries_ensembles_{}.html'.format(target))
        output_file(_fig_file)
        save(column(*plots))


def plot_prediction_scatter(pst_dir, glob='tongue', target=1779, sample_n=30):
    pst = Pst(os.path.join(pst_dir, '{}.pst'.format(glob)))

    def reduce_obs_ens(obj):
        obj = pd.DataFrame(obj)
        obj['time'] = [int(i.split(':')[-2].split('_')[0]) for i in obj.index]
        obj = obj.loc[[i for i, t in obj['Weight'].items() if t > 0.0]]
        obj['oname'] = [i.split(':')[1].split('_')[1] for i in obj.index]
        obj = obj.loc[[i for i in obj.index if str(target.lower()) in i]]
        return obj

    pr_oe = pd.read_csv(os.path.join(pst_dir, '{}.0.base.rei'.format(glob)),
                        header=0, skiprows=3, delim_whitespace=True, index_col=0)
    pr_oe = reduce_obs_ens(pr_oe)
    print('Modelled with prior: {:.3f}'.format(pr_oe['Modelled'].values.mean()))

    noptmax = pst.control_data.noptmax - 1
    pt_oe = pd.read_csv(os.path.join(pst_dir, '{}.{}.base.rei'.format(glob, noptmax)),
                        header=0, skiprows=3, delim_whitespace=True, index_col=0)
    pt_oe = reduce_obs_ens(pt_oe)
    print('Modelled with posterior: {:.3f}'.format(pt_oe['Modelled'].values.mean()))

    obs = pst.observation_data.copy()
    obs['time'] = [int(i.split(':')[-2].split('_')[0]) for i in obs.index]
    obs = obs.loc[[i for i, t in obs['time'].items() if t in list(range(6210, 6575))]]
    obs = obs.loc[[i for i in obs.index if str(target.lower()) in i]]
    obs['oname'] = [i.split(':')[1].split('_')[1] for i in obs.index]
    obs['obgnme'] = obs['oname'].copy()
    ogs = obs.obgnme.unique()
    fig, axes = plt.subplots(len(ogs), 2, figsize=(10, 2 * len(ogs)))
    ogs.sort()

    # for each observation group (i.e. timeseries)
    for ax, og in zip(axes, ogs):
        # get values for x axis
        oobs = obs.loc[obs.obgnme == og, :].copy()
        oobs.loc[:, 'time'] = oobs.time.astype(float)
        oobs.sort_values(by='time', inplace=True)
        tvals = oobs.time.values

        # plot prior
        ax[0].scatter(pr_oe.loc['Measured'].values, pr_oe.loc['Modelled'].values,
                      c='0.5', lw=0.5, alpha=0.5, label='prior')
        ax[0].plot([0, 8], [0, 8])

        # plot posterior
        ax[1].scatter(pt_oe.loc['Measured'].values, pt_oe.loc['Modelled'].values,
                      c='b', lw=0.5, alpha=0.5, label='posterior')
        ax[1].plot([0, 8], [0, 8])

        ax[0].set_title(og, loc='left')

        if og == 'swe':
            ax.set_ylim(pt_oe[[i for i in range(0, sample_n)]].values.min(),
                        pt_oe[[i for i in range(0, sample_n)]].values.max())

        break

    fig.tight_layout()
    plt.show()


def show_phi_evolution(pst_dir, glob='tongue'):
    pst = Pst(os.path.join(pst_dir, '{}.pst'.format(glob)))

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

    project = 'flux'

    # pest_root = 'master'
    # src = '/home/dgketchum/PycharmProjects/swim-rs'
    # d = os.path.join(src, 'examples/{}'.format(project))
    # pest_dir_ = os.path.join(d, pest_root)

    flux_obs_ = os.path.join('/media/research/IrrigationGIS/climate/flux_ET_dataset/'
                             'daily_data_files/{}_daily_data.csv')

    pest_dir_ = '/media/research/IrrigationGIS/swim/examples/flux/calibrated_models/two_model_30MAY2024'
    pst_f = os.path.join(pest_dir_, '{}.pst'.format(project))

    # show_phi_evolution(pest_dir_, glob=project)

    plot_tseries_ensembles(pest_dir_, glob=project, targets=FLUX_SELECT[:2], sample_n=5, flux_file=flux_obs_)
    # plot_prediction_scatter(pest_dir_, glob=project, target='US-MJ1', sample_n=30)
# ========================= EOF ====================================================================
