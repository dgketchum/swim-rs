import os

from pyemu import os_utils


def run_pst(_dir, _cmd, pst_file, num_workers, worker_root, master_dir=None, verbose=False):

    os_utils.start_workers(_dir, _cmd, pst_file,
                           num_workers=num_workers,
                           worker_root=worker_root,
                           verbose=verbose,
                           master_dir=master_dir,
                           cleanup=False)


if __name__ == '__main__':
    project_ = 'tongue'
    d = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}'.format(project_)
    p_dir = os.path.join(d, 'pest')
    m_dir = os.path.join(d, 'master')
    w_dir = os.path.join(d, 'workers')
    exe_ = 'pestpp-ies'
    _pst = '{}.pst'.format(project_)
    _workers = 1

    run_pst(p_dir, exe_, _pst, num_workers=_workers, worker_root=w_dir,
            master_dir=m_dir, verbose=True)
# ========================= EOF ====================================================================
