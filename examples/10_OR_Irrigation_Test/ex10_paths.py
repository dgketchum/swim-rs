"""Shared config loader and constants for Example 10 (Oregon irrigation test)."""

import os
from pathlib import Path

from swimrs.swim.config import ProjectConfig

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "10_OR_Irrigation_Test.toml"
E5 = HERE.parent / "5_Flux_Ensemble" / "container_build"

# The four LST-based OpenET v2.1 members (SIMS and PT-JPL excluded).
LST_MODELS = ("ssebop", "eemetric", "geesebal", "disalexi")

# Single GridMET cell shared by all 24 fields (modal cell, 11 of 24 fields).
# Center from the OpenET GRIDMET_ID row-major index (SW origin, 1386 columns,
# 1/24 degree): row 491, col 144. Field centroids tagged 680670 average
# 45.531 N, -118.768 E, inside this cell.
GRIDMET_CELL = 680670
GRIDMET_CELL_LAT = 45.525
GRIDMET_CELL_LON = -118.7667


def load_config(calibrate: bool = False) -> ProjectConfig:
    cfg = ProjectConfig()
    cfg.read_config(str(CONFIG), calibrate=calibrate)
    return cfg


def etf_dir(cfg: ProjectConfig) -> str:
    return os.path.join(cfg.data_dir, "etf_v21_openet_eto")


def refet_dir(cfg: ProjectConfig) -> str:
    return os.path.join(cfg.data_dir, "openet_refet")
