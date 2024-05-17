import os
import time
import argparse

import numpy as np

from model.etd import obs_field_cycle
from swim.config import ProjectConfig
from swim.input import SamplePlots


def optimize_fields(ini_path, worker_dir):
    start_time = time.time()
    end_time = None

    config = ProjectConfig()
    config.read_config(ini_path)

    fields = SamplePlots()
    fields.initialize_plot_data(config)

    df = obs_field_cycle.field_day_loop(config, fields, debug_flag=False)

    # debug_flag=False just returns the ndarray for writing
    eta_result, swe_result = df
    for i, fid in enumerate(fields.input['order']):
        pred_eta, pred_swe = eta_result[:, i], swe_result[:, i]
        np.savetxt(os.path.join(worker_dir, 'pred', 'pred_eta_{}.np'.format(fid)), pred_eta)
        np.savetxt(os.path.join(worker_dir, 'pred', 'pred_swe_{}.np'.format(fid)), pred_swe)
        end_time = time.time()
        obs_eta = np.loadtxt(
            os.path.join('/home/dgketchum/PycharmProjects/swim-rs/examples/flux/obs/obs_eta_US-MC1.np'), dtype=float)
        rmse = np.sqrt(np.mean((pred_eta - obs_eta) ** 2))
    print('\n\nExecution time: {:.2f} seconds, mean pred ET: {:.3f}, RMSE: {:.3f}\n\n'.format(end_time - start_time,
                                                                                              np.nanmean(pred_eta),
                                                                                              rmse))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, help='Path to config file')
    parser.add_argument('--worker_dir', required=True, help='Worker directory')
    args = parser.parse_args()
    optimize_fields(args.config_path, args.worker_dir)


if __name__ == '__main__':
    main()
