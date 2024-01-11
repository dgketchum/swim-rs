import os
import sys
import subprocess
import multiprocessing as mp

import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

d = '/home/dgketchum/PycharmProjects/et-demands/examples/tongue/'
field_id = '1786'


def activate_conda_environment(environment_name):
    try:
        activate_cmd = f"conda run -n {environment_name} /bin/bash"
        subprocess.run(activate_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error activating Conda environment '{environment_name}': {e}")
        sys.exit(1)


activate_conda_environment('mihm')


def run():
    p = '/home/dgketchum/PycharmProjects/et-demands/fieldET/run_field_et.py'
    os.system('python' + ' {}'.format(p))
    os.chdir("../etd_examples")


if __name__ == '__main__':
    mp.freeze_support()
    run()