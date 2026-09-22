"""Canonical Example 6 (paper E0 and E2) locations, derived from the project TOML.

Every on-disk location is built from ``root`` in the configuration TOML; change
that one line to relocate the workspace. ``load_config`` is not called at import
time because ``ProjectConfig.read_config`` creates the project workspace directory.

    CANONICAL_RUN   the published E2 calibration (GrassBasis, HWSD AWC recal)
    BASELINE_RUN    the superseded ETr-basis run some audits compare against
"""

import os
import tomllib
from pathlib import Path

EX6 = Path(__file__).resolve().parent
REPO = EX6.parents[1]
CANONICAL_RUN = "6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr"
CANONICAL_CONFIG = EX6 / f"{CANONICAL_RUN}.toml"
BASELINE_RUN = "6_Flux_International_LSEnsemble_POR_annual2yr"
BASELINE_CONFIG = EX6 / f"{BASELINE_RUN}.toml"
FINAL_DIR = REPO / "paper" / "data" / "final"
NOPTMAX = 3


def load_config(config_path=None, calibrate=True):
    from swimrs.swim.config import ProjectConfig

    cfg = ProjectConfig()
    cfg.read_config(str(config_path or CANONICAL_CONFIG), calibrate=calibrate)
    return cfg


def swim_root(config_path=None):
    """``root`` read from the TOML without side effects (sibling examples live under it)."""
    with open(config_path or CANONICAL_CONFIG, "rb") as fh:
        return Path(tomllib.load(fh)["root"])


def flux_root_static(config_path=None):
    """``[validation] flux_dir`` read from the TOML without side effects.

    The flux archive is an absolute path outside the workspace, so no ``{root}``
    interpolation is needed; ``load_config`` returns the interpolated value.
    """
    with open(config_path or CANONICAL_CONFIG, "rb") as fh:
        return tomllib.load(fh)["validation"]["flux_dir"]


def results_root(cfg=None):
    cfg = cfg or load_config()
    return Path(cfg.project_ws) / "results"


def run_dir(name=CANONICAL_RUN, cfg=None):
    return results_root(cfg) / name


def archive_dir(name=CANONICAL_RUN, cfg=None):
    return run_dir(name, cfg) / "archive"


def qa_root(cfg=None):
    """Container-build QA evidence (Cat 2 sidecars) for the E2 re-footing."""
    cfg = cfg or load_config()
    return Path(cfg.data_dir) / "e2_etf_refooting"


def flux_root(cfg=None):
    cfg = cfg or load_config()
    return cfg.flux_dir


def calibration_log(cfg, suffix=""):
    """``batch_runner`` stdout the chains redirect to ``{project_ws}/nohup_calibrate_<run>.out``."""
    stem = os.path.basename(cfg.container_path).removeprefix(f"{cfg.project_name}_")
    stem = stem.removesuffix(".swim")
    return Path(cfg.project_ws) / f"nohup_calibrate_{stem}{suffix}.out"
