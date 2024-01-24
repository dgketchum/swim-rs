import os
import sys
import warnings
import subprocess
import multiprocessing as mp

warnings.filterwarnings("ignore", category=FutureWarning)


def run():
    p = '/home/dgketchum/PycharmProjects/swim-rs/run/run_flux_etd.py'
    python_path = '/home/dgketchum/PycharmProjects/swim-rs'
    os.environ['PYTHONPATH'] = python_path
    os.system('python' + ' {}'.format(p))


if __name__ == '__main__':
    mp.freeze_support()
    run()