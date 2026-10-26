"""Repository, data-root, and frozen-package locations shared by every figure builder."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_VERSION = "1.0.0"

# Figure 1 display package only.  Kept separate from SCRIPT_VERSION so that
# revising the fig01 data contract does not perturb the byte content of the
# fig02-fig06 artifacts (two of which embed the package generator version).
FIG01_BUILDER_VERSION = "2.2.0"

REPO = Path(__file__).resolve().parents[3]
FINAL = REPO / "paper" / "data" / "final"
OUT = FINAL / "figures"

# Workspace locations come from each example's TOML-derived paths module, so the
# builder follows a relocated ``root`` for free.  ``examples/`` is not a package.
for _ex in ("5_Flux_Ensemble", "6_Flux_International", "7_Applied_Water"):
    if str(REPO / "examples" / _ex) not in sys.path:
        sys.path.append(str(REPO / "examples" / _ex))
import ex5_paths  # noqa: E402
import ex6_paths  # noqa: E402
import ex7_paths  # noqa: E402

_CFG5 = ex5_paths.load_config()
_CFG6 = ex6_paths.load_config()
_CFG6_ROOT = ex6_paths.swim_root()  # the data root; recorded paths re-root below it

E1_RUN = Path(ex5_paths.run_dir(cfg=_CFG5))
E1_ARCHIVE = E1_RUN / "archive"
E1_CONTAINER = Path(ex5_paths.run_container(cfg=_CFG5))
E1_WITHIN_STRAT = E1_RUN / "within_e1_transfer"  # six-arm within_e1_transfer.py output

# E2 sources are the GrassBasis calibration on HWSD per-site aw priors.  The frozen
# copies under paper/data/final/e2_closure_pool/ are the paper's source of record; the results
# root is read only where a per-site table was not frozen, and its per-site
# transfer table must hash-match the frozen copy.
E2_RESULTS = ex6_paths.run_dir(cfg=_CFG6)
E2_TRANSFER = ex6_paths.results_root(_CFG6) / ex6_paths.TRANSFER_RUN
E2_CONTAINER = (
    Path(_CFG6.data_dir) / "6_Flux_International_ls_ensemble_grassbasis_por_annual2yr.swim"
)
E2_FROZEN = FINAL / "e2_closure_pool"
E2_FROZEN_POOL = E2_FROZEN / "closure_pool"
E2_FROZEN_TRANSFER = E2_FROZEN / "transfer"
E2_MAPPING_META = E2_FROZEN_TRANSFER / "e3_irrigation_stratified_param_mapping_metadata.json"

E3_LOCAL = ex7_paths.eval_dir(ex7_paths.LOCAL_LABEL)
E3_TRANSFER = ex7_paths.eval_dir(ex7_paths.TRANSFER_LABEL)
E3_CONTAINER = ex7_paths.run_container()

NE_COUNTRIES = REPO / "data" / "cartographic" / "ne_110m_admin_0_countries.shp"
# Faint CONUS state boundaries for the Figure 1 map context.
NE_STATES = REPO / "data" / "cartographic" / "ne_50m_admin_1_states_provinces_lakes.shp"
# Privacy-safe E3 basin context for the Figure 1 map.
SLV_BASIN = REPO / "data" / "cartographic" / "wbd_huc8_rio_grande_headwaters.fgb"
SLV_BASIN_SOURCE = REPO / "data" / "cartographic" / "wbd_huc8_rio_grande_headwaters.SOURCE.json"
