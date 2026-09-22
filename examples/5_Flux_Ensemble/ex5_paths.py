"""Canonical Example 5 (paper E1) locations, derived from the project TOML.

Every on-disk location is built from ``root`` in ``5_Flux_Ensemble.toml``; change
that one line to relocate the workspace. Nothing here is read at import time
because ``ProjectConfig.read_config`` creates the project workspace directory.

    run22    internal archive tag of the published E1 calibration (Run 22)
"""

import os
from functools import cache
from pathlib import Path

from swimrs.swim.config import ProjectConfig

EX5 = Path(__file__).resolve().parent
REPO = EX5.parents[1]
CANONICAL_CONFIG = EX5 / "5_Flux_Ensemble.toml"
CANONICAL_RUN = "run22"
FINAL_DIR = REPO / "paper" / "data" / "final"
NOPTMAX = 3


@cache
def load_config(config_path=None, calibrate=True):
    cfg = ProjectConfig()
    cfg.read_config(str(config_path or CANONICAL_CONFIG), calibrate=calibrate)
    return cfg


def results_root(cfg=None):
    cfg = cfg or load_config()
    return os.path.join(cfg.project_ws, "results")


def run_dir(tag=CANONICAL_RUN, cfg=None):
    return os.path.join(results_root(cfg), tag)


def run_container(tag=CANONICAL_RUN, cfg=None):
    """Run container written by ``container_build/build_container.py --run <tag>``."""
    cfg = cfg or load_config()
    return os.path.join(cfg.data_dir, f"{cfg.project_name}_{tag}.swim")


def posterior_par_csv(tag=CANONICAL_RUN, cfg=None):
    cfg = cfg or load_config()
    return os.path.join(run_dir(tag, cfg), f"{cfg.project_name}.{NOPTMAX}.par.csv")


def calibration_log(tag=CANONICAL_RUN, cfg=None):
    cfg = cfg or load_config()
    return os.path.join(cfg.project_ws, f"nohup_{tag}_calibrate.out")
