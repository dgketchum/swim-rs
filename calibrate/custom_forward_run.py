import os

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def run():
    p = '/home/dgketchum/PycharmProjects/swim-rs/run/run_mp.py'
    python_path = '/home/dgketchum/PycharmProjects/swim-rs'
    os.environ['PYTHONPATH'] = python_path

    conf_file = '/home/dgketchum/PycharmProjects/swim-rs/examples/flux/flux_swim.toml'

    cwd = os.getcwd()

    args = ['python' + ' {}'.format(p),
            '--config_path', conf_file,
            '--worker_dir', cwd]

    if 'worker' in os.path.basename(cwd):
        calibration_dir = os.path.join(os.getcwd(), 'mult')
        args += ['--calibration_dir', calibration_dir]

    os.system(' '.join(args))


if __name__ == '__main__':
    run()
