import os

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def run():
    """This script is meant to be executed by PEST++"""
    home = os.path.expanduser('~')

    root = os.path.join(home, 'PycharmProjects', 'swim-rs')
    os.environ['PYTHONPATH'] = root

    model_script = os.path.join(root, 'run', 'run_mp.py')

    project_ws = os.path.join(root, 'tutorials', '2_Fort_Peck')

    conf_file = os.path.join(project_ws, 'data', 'tutorial_config.toml')

    cwd = os.getcwd()

    args = ['python' + ' {}'.format(model_script),
            '--project_dir', project_ws,
            '--config_path', conf_file,
            '--worker_dir', cwd]

    if 'worker' in os.path.basename(cwd):
        calibration_dir = os.path.join(os.getcwd(), 'mult')
        args += ['--calibration_dir', calibration_dir]

    os.system(' '.join(args))


if __name__ == '__main__':
    run()
