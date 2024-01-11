import os
import sys
import warnings
import subprocess
import multiprocessing as mp

warnings.filterwarnings("ignore", category=FutureWarning)


def run():
    p = '/home/dgketchum/PycharmProjects/et-demands/fieldET/run_flux_et.py'
    os.system('python' + ' {}'.format(p))


if __name__ == '__main__':
    mp.freeze_support()
    run()