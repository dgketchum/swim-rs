import os

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def split_path(path):
    parts = []
    while True:
        path, folder = os.path.split(path)
        if folder:
            parts.append(folder)
        else:
            if path:
                parts.append(path)
            break
    return parts[::-1]


def run():
    """This script is meant to be executed by PEST++"""

    root = '/home/dgketchum/PycharmProjects/swim-rs'

    os.environ['PYTHONPATH'] = root

    model_script = os.path.join(root, 'calibrate', 'run_mp.py')

    project_ws = os.path.join(root, 'tutorials', '4_Flux_Network')

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
