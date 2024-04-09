import os
import shutil

import psutil

from pyemu import os_utils


def run_pst(_dir, _cmd, pst_file, num_workers, worker_root, master_dir=None, verbose=False):

    [shutil.rmtree(os.path.join(_dir, d)) for d in os.listdir(_dir) if 'worker' in d]

    os_utils.start_workers(_dir, _cmd, pst_file,
                           num_workers=num_workers,
                           worker_root=worker_root,
                           verbose=verbose,
                           master_dir=master_dir)


if __name__ == '__main__':
    project_ = 'tongue'
    d = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}/pest'.format(project_)
    exe_ = 'pestpp-ies'
    _pst = '{}.pst'.format(project_)
    _workers = psutil.cpu_count(logical=False) - 2
    worker_rt = os.path.join(d)

    run_pst(d, exe_, _pst, num_workers=_workers, worker_root=worker_rt, verbose=True)
# ========================= EOF ====================================================================
