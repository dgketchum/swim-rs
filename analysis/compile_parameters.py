import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parameter_histogram(input_json, plot_dir):
    with open(input_json, 'r') as f:
        param_arr = json.load(f)

    keys = list(list(param_arr['fields'].items())[0][1].keys())

    for par in keys:

        v = np.array([v[par] for k, v in param_arr['fields'].items()])

        mean_ = np.mean(v)
        plt.figure(figsize=(10, 6))
        sns.histplot(v)
        plt.axvline(mean_, color='red', linestyle='dashed', linewidth=1)
        plt.title(f'{par.upper()} Calibrated Parameter Value Histogram')
        plt.xlabel(f'{par} Value')
        plt.ylabel('Frequency')
        plt.grid(True)

        textstr = '\n'.join((
            r'$n={}$'.format(v.shape[0]),
            r'$\mu={:.2f}$'.format(mean_)))
        props = dict(boxstyle='round', facecolor='white')
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
                       verticalalignment='top', bbox=props)

        plot_path = os.path.join(plot_dir, f'{par.upper()}_histogram.png')
        plt.savefig(plot_path)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    js_ = '/home/dgketchum/Downloads/tongue_params.json'
    pdir = '/home/dgketchum/Downloads/tongue_params'

    parameter_histogram(js_, plot_dir=pdir)
# ========================= EOF ====================================================================
