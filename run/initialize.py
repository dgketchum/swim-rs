import os

from swim.config import ProjectConfig
from swim.sampleplots import SamplePlots


def initialize_data(ini_path, project_ws, input_data=None, spinup_data=None, calibration_dir=None,
                    forecast=False, calibrate=False, forecast_file=None):
    config = ProjectConfig()
    config.read_config(ini_path, project_ws, calibration_dir=calibration_dir, forecast=forecast,
                       calibrate=calibrate, forecast_param_csv=forecast_file)

    if input_data:
        config.input_data = input_data

    if spinup_data:
        config.spinup = spinup_data

    plots_ = SamplePlots()
    plots_.initialize_plot_data(config)

    return config, plots_


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
