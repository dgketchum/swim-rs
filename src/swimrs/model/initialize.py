from swimrs.swim.config import ProjectConfig
from swimrs.swim.sampleplots import SamplePlots


def initialize_data(ini_path, input_data=None, spinup_data=None):
    """Load ProjectConfig and SamplePlots from a TOML and optional overrides.

    Parameters
    - ini_path: path to project TOML.
    - input_data: optional path to a prepped_input JSON to override config.
    - spinup_data: optional path to spinup JSON to override config.

    Returns
    - (config: ProjectConfig, plots: SamplePlots) ready for modeling workflows.
    """
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
