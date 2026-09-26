"""Canonical Example 7 (paper E3) locations, derived from the project TOML.

Every workspace location is built from ``root`` in ``7_Applied_Water.toml``; change
that one line to relocate the workspace. The helpers below read the TOML with
``tomllib`` and interpolate its ``{key}`` templates themselves, so importing this
module or calling them has no side effects (``ProjectConfig.read_config`` creates
the project workspace directory, so ``load_config`` is only for scripts that need
the full configuration).

    RUN_NAME        the published batch IES calibration (e7cal)
    LOCAL_LABEL     evaluator label of the locally calibrated arm
    EX5_CANONICAL_RUN  the current Example 5 canonical run (``ex5_paths.CANONICAL_RUN``)
    EX5_TRANSFER_RUN   the Example 5 run whose posterior the published E3 transfer
                       uses (the canonical run; run23 since 2026-09-26, Run 22
                       outputs kept as superseded)
    TRANSFER_LABEL  evaluator label of the irrigated-class Ex5 transfer arm
    EXCLUDED_FIELD_YEARS  metered (site_id, year) rows withheld from every scorer
"""

import importlib.util
import tomllib
from pathlib import Path

EX7 = Path(__file__).resolve().parent
REPO = EX7.parents[1]
CONFIG = EX7 / "7_Applied_Water.toml"
RUN_NAME = "e7cal"
NOPTMAX = 3
LOCAL_LABEL = "calibrated"


def _ex5_canonical_run():
    """``CANONICAL_RUN`` from ``examples/5_Flux_Ensemble/ex5_paths.py``, loaded by path."""
    path = REPO / "examples" / "5_Flux_Ensemble" / "ex5_paths.py"
    spec = importlib.util.spec_from_file_location("_ex7_ex5_paths", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CANONICAL_RUN


EX5_CANONICAL_RUN = _ex5_canonical_run()
# E3 transfers the canonical Ex5 posterior. Re-footed on run23 2026-09-26 (mapping
# and forward run redone on the run23 irrigated vector); the Run 22 arm stays on disk
# at results/applied_transfer_run22_by_irrigation (see its SUPERSEDES.md).
EX5_TRANSFER_RUN = EX5_CANONICAL_RUN
TRANSFER_LABEL = f"transfer_{EX5_TRANSFER_RUN}_by_irrigation"

# Metered field-years withheld from scoring (decided 2026-09-26). The RGDSS
# parcel-to-well link (MASTER_ID) is unstable in 2019: each of these four rows
# resolves to the well and parcel that belong to a different cohort field in every
# other year, so the "truth" is another field's pumping and the one-well-to-one-parcel
# selection rule does not hold. metered_truth.csv keeps the rows as built by
# select_fields.py; every scorer drops them through drop_excluded_field_years().
# Evidence: review/_work/pipeline/suspect_2019.csv under the Ex7 workspace.
EXCLUDED_FIELD_YEARS = {
    ("SLV_013", 2019): "2019 WDID 3505024 is SLV_018's well (88 km from the field)",
    ("SLV_024", 2019): "2019 WDID 3505025 is SLV_005's well (91 km from the field)",
    ("SLV_025", 2019): "2019 WDID 2605692 is SLV_023's well (2.6 km from the field)",
    ("SLV_034", 2019): "2019 WDID 3505026 is SLV_038's well (90 km from the field)",
}


def drop_excluded_field_years(df):
    """``df`` without the rows keyed ``(site_id, year)`` in ``EXCLUDED_FIELD_YEARS``."""
    keep = [(str(s), int(y)) not in EXCLUDED_FIELD_YEARS for s, y in zip(df["site_id"], df["year"])]
    return df.loc[keep].copy()


FINAL_DIR = REPO / "paper" / "data" / "final"
TRANSFER_VECTORS = FINAL_DIR / f"e2_{EX5_TRANSFER_RUN}_transfer_vectors_by_irrigation.json"
TRUTH_CSV = EX7 / "data" / "metered_truth.csv"

# Ground-truth builds (code tracked, data untracked; see the READMEs in each dir).
SLV_DIR = REPO / "data" / "co_slv_wells"
WMIS_DIR = REPO / "data" / "idwr_wmis"
ESPA_CONTROL_IRR = WMIS_DIR / "espa_control_irrmapper.csv"


def load_config(config_path=None, calibrate=True):
    from swimrs.swim.config import ProjectConfig

    cfg = ProjectConfig()
    cfg.read_config(str(config_path or CONFIG), calibrate=calibrate)
    return cfg


def _interpolated(config_path=None):
    """``root``, ``project``, ``[paths]`` and ``[calibration]`` with ``{key}`` templates resolved."""
    with open(config_path or CONFIG, "rb") as fh:
        toml = tomllib.load(fh)
    vals = {"root": toml["root"], "project": toml["project"]}
    vals.update({k: v for k, v in toml.get("paths", {}).items() if isinstance(v, str)})
    vals.update({k: v for k, v in toml.get("calibration", {}).items() if isinstance(v, str)})
    for _ in range(10):
        resolved = {k: (v.format_map(vals) if "{" in v else v) for k, v in vals.items()}
        if resolved == vals:
            break
        vals = resolved
    return vals


def swim_root(config_path=None):
    return Path(_interpolated(config_path)["root"])


def project_ws(config_path=None):
    return Path(_interpolated(config_path)["project_workspace"])


def data_dir(config_path=None):
    return Path(_interpolated(config_path)["data"])


def gis_dir(config_path=None):
    return Path(_interpolated(config_path)["gis"])


def fields_shp(config_path=None):
    return Path(_interpolated(config_path)["fields_shapefile"])


def base_container(config_path=None):
    """The extraction container ``container_prep.py`` writes (no ETf target, no calibration)."""
    return Path(_interpolated(config_path)["container"])


def run_container(run_name=RUN_NAME, config_path=None):
    """The run container ``build_container.py --run <name>`` writes beside the base one."""
    vals = _interpolated(config_path)
    return Path(vals["data"]) / f"{vals['project']}_{run_name}.swim"


def pest_run_dir(config_path=None):
    return Path(_interpolated(config_path)["pest_run_dir"])


def results_root(config_path=None):
    return project_ws(config_path) / "results"


def run_dir(name=RUN_NAME, config_path=None):
    return results_root(config_path) / name


def archive_dir(name=RUN_NAME, config_path=None):
    return run_dir(name, config_path) / "archive"


def merged_posterior(name=RUN_NAME, config_path=None):
    """``archive_postcalibration.py`` output the calibrated-arm evaluation reads."""
    return archive_dir(name, config_path) / "4_pest_outputs" / "merged" / "merged_posterior.json"


def eval_dir(label, config_path=None):
    """``evaluate_applied_water.py --label <label>`` output directory."""
    return results_root(config_path) / f"applied_{label}"
