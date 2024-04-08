import os

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def run():
    p = '/home/dgketchum/PycharmProjects/swim-rs/run/run_field_etd_mp.py'
    python_path = '/home/dgketchum/PycharmProjects/swim-rs'
    os.environ['PYTHONPATH'] = python_path
    os.system('python' + ' {}'.format(p))


if __name__ == '__main__':
    run()
