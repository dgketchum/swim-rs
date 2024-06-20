import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from swim.config import ProjectConfig
from swim.input import SamplePlots
from model.etd.initialize_tracker import PlotTracker


def parameter_histogram(ini_path, parameter_distribution, plot_dir):
    config = ProjectConfig()
    config.read_config(ini_path, calibration_folder=None, parameter_dist_csv=parameter_distribution)

    fields = SamplePlots()
    fields.initialize_plot_data(config)

    params = None
    size = len(fields.input['order'])
    tracker = PlotTracker(size)
    tracker.load_soils(fields)

    param_arr = {k: np.zeros((1, size)) for k in config.forecast_parameter_groups}

    for k, v in config.forecast_parameters.items():

        group, fid = '_'.join(k.split('_')[:-1]), k.split('_')[-1]
        # PEST++ has lower-cased the FIDs
        l = [x.lower() for x in fields.input['order']]
        idx = l.index(fid)

        if params:
            value = params[k]
        else:
            value = v

        param_arr[group][0, idx] = value

    for k, v in param_arr.items():

        if k != 'aw':
            continue
        # Create a histogram plot
        mean_ = np.mean(v)
        plt.figure(figsize=(10, 6))
        sns.kdeplot(v)
        plt.axvline(mean_, color='red', linestyle='dashed', linewidth=1)
        plt.title(f'{k} Calibrated Parameter Value Histogram')
        plt.xlabel(f'{k} Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.legend(None)

        textstr = '\n'.join((
            r'$n={}$'.format(v.shape[1]),
            r'$\mu={:.2f}$'.format(mean_)))
        props = dict(boxstyle='round', facecolor='white')
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
                       verticalalignment='top', bbox=props)

        plot_path = os.path.join(plot_dir, f'{k}_histogram.png')
        # plt.savefig(plot_path)
        plt.show()


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    project = 'tongue'
    d = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}'.format(project)
    conf = os.path.join(d, '{}_swim.toml'.format(project))

    tuned = '/media/research/IrrigationGIS/swim/examples/{}/calibrated_models/model_tongue_19JUN2024/'.format(project,
                                                                                                              project)

    pars = os.path.join(tuned, '{}.4.par.csv'.format(project))

    histograms_ = os.path.join(tuned, 'histograms')

    parameter_histogram(conf, parameter_distribution=pars, plot_dir=histograms_)
# ========================= EOF ====================================================================
