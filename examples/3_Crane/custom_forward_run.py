import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def run():
    """Custom forward runner for PEST++ workers.

    This imports the packaged optimize_fields function and executes it with the
    configuration from the 3_Crane project. It expects to be run in the worker
    directory; PEST++ will create a 'mult' subfolder containing parameter files.
    """
    from swimrs.calibrate.run_mp import optimize_fields
    from swimrs.swim.config import ProjectConfig

    here = os.path.dirname(os.path.abspath(__file__))
    conf_file = os.path.join(here, "3_Crane.toml")
    if not os.path.exists(conf_file):
        conf_file = os.path.join(here, "config.toml")
    if not os.path.exists(conf_file):
        raise FileNotFoundError(f"Expected config at {conf_file}")

    # Resolve input_data path from config
    cfg = ProjectConfig()
    cfg.read_config(conf_file)
    input_data_path = cfg.input_data

    # PEST++ worker CWD
    cwd = os.getcwd()
    calibration_dir = os.path.join(cwd, "mult")

    optimize_fields(conf_file, input_data_path, cwd, calibration_dir)


if __name__ == "__main__":
    run()
