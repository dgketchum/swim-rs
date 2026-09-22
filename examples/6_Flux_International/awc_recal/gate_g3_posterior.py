"""Gate G3 (HWSD AWC units recal): posterior ``aw`` vs the HWSD prior, vs the superseded run.

Reports, for the new run and (when present) the superseded run:

* the ``aw`` upper-bound hit rate from ``archive/5_posterior_summaries/boundary_hit_rates.csv``
  (superseded: 0.3939);
* the median posterior ``aw`` from ``posterior_site_summary.csv``;
* Spearman rank correlation of posterior site-median ``aw`` with the HWSD AWC (superseded ~ -0.05).

Informational unless the files are missing (exit 1). Writes ``awc_recal_g3.json`` into the new
run's ``5_posterior_summaries``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
import ex6_paths  # noqa: E402

DEFAULT_RUN = ex6_paths.CANONICAL_RUN
SUPERSEDED = "superseded_awc320_20260921"  # under {project_ws}/results


def summarize(run_dir: str, hwsd: pd.Series) -> dict:
    post = os.path.join(run_dir, "archive", "5_posterior_summaries")
    bhr = pd.read_csv(os.path.join(post, "boundary_hit_rates.csv"))
    pss = pd.read_csv(os.path.join(post, "posterior_site_summary.csv"))
    aw_row = bhr.loc[(bhr["parameter"] == "aw") & (bhr["lulc_group"] == "ALL")].iloc[0]
    aw = pss.loc[pss["param"] == "aw"].set_index("site_id")["median"].astype(float)
    common = aw.index.intersection(hwsd.index)
    rho, p = stats.spearmanr(aw.loc[common].values, hwsd.loc[common].values)
    return {
        "run_dir": run_dir,
        "aw_upper_hit_rate": float(aw_row["upper_hit_rate"]),
        "aw_lower_hit_rate": float(aw_row["lower_hit_rate"]),
        "aw_bounds": str(aw_row["bounds"]),
        "n_sites": int(len(aw)),
        "aw_median_mm_per_m": float(aw.median()),
        "aw_q25": float(aw.quantile(0.25)),
        "aw_q75": float(aw.quantile(0.75)),
        "frac_sites_at_400": float(np.mean(np.isclose(aw.values, 400.0, atol=1e-6))),
        "spearman_posterior_aw_vs_hwsd": float(rho),
        "spearman_p": float(p),
        "n_common_with_hwsd": int(len(common)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ex6_paths.CANONICAL_CONFIG), help="Run TOML")
    ap.add_argument("--run-name", default=DEFAULT_RUN)
    ap.add_argument("--results-root", default=None, help="default {project_ws}/results")
    ap.add_argument(
        "--old-run-dir", default=None, help=f"default <results-root>/{SUPERSEDED}/<run>"
    )
    ap.add_argument("--hwsd-csv", default=None, help="default the TOML's hwsd_csv")
    args = ap.parse_args()

    if None in (args.results_root, args.old_run_dir, args.hwsd_csv):
        cfg = ex6_paths.load_config(args.config)
        args.results_root = args.results_root or str(ex6_paths.results_root(cfg))
        args.old_run_dir = args.old_run_dir or os.path.join(
            args.results_root, SUPERSEDED, args.run_name
        )
        args.hwsd_csv = args.hwsd_csv or cfg.hwsd_csv

    hwsd = pd.read_csv(args.hwsd_csv).set_index("sid")["awc"].astype(float)
    new_dir = os.path.join(args.results_root, args.run_name)
    report = {"new": summarize(new_dir, hwsd)}
    if os.path.isdir(os.path.join(args.old_run_dir, "archive", "5_posterior_summaries")):
        report["superseded"] = summarize(args.old_run_dir, hwsd)
    out = os.path.join(new_dir, "archive", "5_posterior_summaries", "awc_recal_g3.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    print("G3 reported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
