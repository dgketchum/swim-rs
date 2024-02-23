import os

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def run():
    p = '/home/dgketchum/PycharmProjects/swim-rs/run/run_field_etd_mp.py'
    os.chdir("../../..")
    os.system('python' + ' {}'.format(p))


if __name__ == '__main__':
    run()
