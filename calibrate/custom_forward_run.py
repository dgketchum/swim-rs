import os

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def run():
    p = '/home/dgketchum/PycharmProjects/swim-rs/run/run_field_etd_mp.py'
    python_path = '/home/dgketchum/PycharmProjects/swim-rs'
    conf_file = '/home/dgketchum/PycharmProjects/swim-rs/examples/tongue/tongue_swim.toml'
    os.environ['PYTHONPATH'] = python_path
    args = ['python' + ' {}'.format(p), '--config_path', conf_file, '--worker_dir', os.getcwd()]
    os.system(' '.join(args))


if __name__ == '__main__':
    run()
