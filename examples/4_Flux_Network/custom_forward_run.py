import os

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def run():
    """This script is meant to be executed by PEST++"""

    home = os.path.expanduser('~')
    root = os.path.join(home, 'code', 'swim-rs')
    src = os.path.join(root, 'src', 'swimrs')

    os.environ['PYTHONPATH'] = root

    model_script = os.path.join(src, 'calibrate', 'run_mp.py')

    project_ws = os.path.join(root, 'examples', '4_Flux_Network')

    conf_file = os.path.join(project_ws, '4_Flux_Network.toml')

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
