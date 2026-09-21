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

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = "/data/ssd1/swim/6_Flux_International/results"
DEFAULT_RUN = "6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr"
DEFAULT_OLD = os.path.join(RESULTS, "superseded_awc320_20260921", DEFAULT_RUN)
DEFAULT_HWSD = (
    "/data/ssd1/swim/6_Flux_International/data/properties/6_Flux_International_hwsd_crop.csv"
)


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
    ap.add_argument("--run-name", default=DEFAULT_RUN)
    ap.add_argument("--results-root", default=RESULTS)
    ap.add_argument("--old-run-dir", default=DEFAULT_OLD)
    ap.add_argument("--hwsd-csv", default=DEFAULT_HWSD)
    args = ap.parse_args()

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
