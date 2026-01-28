import os
import time
import argparse

import numpy as np

from swimrs.model import obs_field_cycle
from swimrs.swim.config import ProjectConfig
from swimrs.swim.sampleplots import SamplePlots


def optimize_fields(config_path, input_data_path, worker_dir, calibration_dir):
    start_time = time.time()
    end_time = None

    config = ProjectConfig()
    config.read_config(config_path, calibration_dir_override=calibration_dir)

    config.calibrate = True
    config.input_data = input_data_path
    config.calibration_dir = calibration_dir

    fields = SamplePlots()
    fields.initialize_plot_data(config)

    config.read_calibration_parameters(sites=fields.input['order'])

    if len(config.calibrated_parameters) == 0:
        raise ValueError

    df = obs_field_cycle.field_day_loop(config, fields, debug_flag=False)

    # debug_flag=False just returns the ndarray for writing
    etf_result, swe_result = df
    for i, fid in enumerate(fields.input['order']):
        pred_eta, pred_swe = etf_result[:, i], swe_result[:, i]
        np.savetxt(os.path.join(worker_dir, 'pred', 'pred_etf_{}.np'.format(fid)), pred_eta)
        np.savetxt(os.path.join(worker_dir, 'pred', 'pred_swe_{}.np'.format(fid)), pred_swe)
        end_time = time.time()
    print('Execution time: {:.2f} seconds\n\n'.format(end_time - start_time))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, help='Path to config file')
    parser.add_argument('--input_data_path', required=True, help='Input Data JSON File Path')
    parser.add_argument('--worker_dir', required=True, help='Worker directory')
    parser.add_argument('--calibration_dir', required=False, help='Calibration (mult) directory')
    args = parser.parse_args()
    optimize_fields(args.config_path, args.input_data_path, args.worker_dir, args.calibration_dir)


if __name__ == '__main__':
    main()
