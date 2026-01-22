import os
import shutil
import warnings

# Suppress pyemu's flopy warning - flopy is optional and not needed for SWIM-RS
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Failed to import legacy module")
    from pyemu import os_utils


def run_pst(
    _dir: str,
    _cmd: str,
    pst_file: str,
    num_workers: int,
    worker_root: str,
    master_dir: str | None = None,
    verbose: bool = True,
    cleanup: bool = True,
) -> None:
    """Run PEST++ calibration with parallel workers.

    Launches the PEST++ master and worker processes using pyemu's os_utils.
    Workers execute the forward model in parallel across multiple cores.

    Args:
        _dir: Directory containing the .pst control file.
        _cmd: PEST++ executable command (e.g., 'pestpp-ies').
        pst_file: Name of the .pst control file.
        num_workers: Number of parallel worker processes.
        worker_root: Directory for worker process files.
        master_dir: Directory for master process output. Defaults to None.
        verbose: Print progress messages. Defaults to True.
        cleanup: Clean up worker directories on completion. Defaults to True.

    Raises:
        ValueError: If the pest directory does not exist.

    Example:
        >>> run_pst(
        ...     _dir='/path/to/pest',
        ...     _cmd='pestpp-ies',
        ...     pst_file='project.pst',
        ...     num_workers=4,
        ...     worker_root='/path/to/workers',
        ...     master_dir='/path/to/master'
        ... )
    """
    try:
        os.chdir(worker_root)
        [print('rmtree: {}'.format(os.path.join(worker_root, d))) for d in os.listdir(worker_root)]
        [shutil.rmtree(os.path.join(worker_root, d)) for d in os.listdir(worker_root)]
    except FileNotFoundError:
        os.mkdir(worker_root)

    try:
        shutil.rmtree(master_dir)
        os.mkdir(master_dir)
    except FileNotFoundError:
        pass

    if not os.path.isdir(_dir):
        raise ValueError(f'The pest directory {_dir} does not exist, run pest_builder.py')

    os.chdir(_dir)

    os_utils.start_workers(_dir,
                           _cmd,
                           pst_rel_path=pst_file,
                           num_workers=num_workers,
                           worker_root=worker_root,
                           verbose=verbose,
                           master_dir=master_dir,
                           cleanup=cleanup,
                           port=5005)


if __name__ == '__main__':
    pass

# ========================= EOF ====================================================================
