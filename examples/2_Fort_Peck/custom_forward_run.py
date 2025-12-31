import os

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def run():
    """Custom forward runner for PEST++ workers."""
    from swimrs.calibrate.run_mp import optimize_fields
    from swimrs.swim.config import ProjectConfig

    here = os.path.dirname(os.path.abspath(__file__))
    conf_file = os.path.join(here, 'config.toml')
    if not os.path.exists(conf_file):
        raise FileNotFoundError(f'Expected config at {conf_file}')

    cfg = ProjectConfig()
    cfg.read_config(conf_file)

    cwd = os.getcwd()
    calibration_dir = os.path.join(cwd, 'mult')
    optimize_fields(conf_file, cfg.input_data, cwd, calibration_dir)


if __name__ == '__main__':
    run()
