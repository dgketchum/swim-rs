import os

from swim.config import ProjectConfig
from swim.sampleplots import SamplePlots


def initialize_data(ini_path, input_data=None, spinup_data=None):
    config = ProjectConfig()
    config.read_config(ini_path)

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
