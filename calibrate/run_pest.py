import os
import psutil

from pyemu import Pst, os_utils


def run_pst(_dir, _cmd, pst, num_workers, worker_root, master_dir):
    os_utils.start_workers(_dir, _cmd, pst,
                           num_workers=num_workers,
                           worker_root=worker_root,
                           master_dir=master_dir)


if __name__ == '__main__':
    project_ = 'tongue'
    d = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}/pest'.format(project_)
    exe_ = 'pestpp-ies'
    _pst = '{}.pst'.format(project_)
    _workers = psutil.cpu_count(logical=False) - 2
    worker_rt = os.path.join(d)

    run_pst(d, exe_, _pst, num_workers=_workers, worker_root=worker_rt, master_dir='ies')
# ========================= EOF ====================================================================
