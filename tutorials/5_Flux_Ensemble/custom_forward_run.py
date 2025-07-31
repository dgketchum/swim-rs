import os

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def run():
    """This script is meant to be executed by PEST++"""

    home = os.path.expanduser('~')
    root = os.path.join(home, 'PycharmProjects', 'swim-rs')

    os.environ['PYTHONPATH'] = root

    model_script = os.path.join(root, 'calibrate', 'run_mp.py')

    project_ws = os.path.join(root, 'tutorials', '5_Flux_Ensemble')

    conf_file = os.path.join(project_ws, '5_Flux_Ensemble.toml')

    cwd = os.getcwd()

    input_data = os.path.join(cwd, 'prepped_input.json')

    calibration_dir = os.path.join(cwd, 'mult')

    args = ['python' + ' {}'.format(model_script),
            '--config_path', conf_file,
            '--input_data_path', input_data,
            '--calibration_dir', calibration_dir,
            '--worker_dir', cwd]

    os.system(' '.join(args))


if __name__ == '__main__':
    run()
