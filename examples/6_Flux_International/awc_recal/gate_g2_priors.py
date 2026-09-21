"""Gate G2 (HWSD AWC units recal): the archived PEST ``aw_*`` priors carry the HWSD values.

Reads every ``archive/3_problem_definition/batch_*/params.csv`` of a run, joins the ``aw_{sid}``
rows to the HWSD CSV (mm/m), and asserts each prior equals the builder rule

    prior = hwsd_mm_per_m      if lower <= hwsd_mm_per_m <= upper
          = 150.0              if hwsd is NaN or below the lower bound (100)
          = 0.8 * upper        if above the upper bound (400; never reached, HWSD max 214)

and that the priors are NOT a constant (the superseded runs had 320.0 at all 66 sites). Writes
``aw_prior_table.csv`` next to the batch dirs. Exit 1 on failure.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

RESULTS = "/data/ssd1/swim/6_Flux_International/results"
DEFAULT_RUN = "6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr"
DEFAULT_HWSD = (
    "/data/ssd1/swim/6_Flux_International/data/properties/6_Flux_International_hwsd_crop.csv"
)


def expected_prior(hwsd_mm: float, lower: float, upper: float) -> float:
    if not np.isfinite(hwsd_mm) or hwsd_mm < lower:
        return 150.0
    if hwsd_mm > upper:
        return upper * 0.8
    return float(hwsd_mm)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default=DEFAULT_RUN)
    ap.add_argument("--results-root", default=RESULTS)
    ap.add_argument("--hwsd-csv", default=DEFAULT_HWSD)
    ap.add_argument("--lower", type=float, default=100.0)
    ap.add_argument("--upper", type=float, default=400.0)
    ap.add_argument("--expect-sites", type=int, default=66)
    ap.add_argument("--out-dir", default=None, help="default: the 3_problem_definition dir")
    args = ap.parse_args()

    prob = os.path.join(args.results_root, args.run_name, "archive", "3_problem_definition")
    files = sorted(glob.glob(os.path.join(prob, "batch_*", "params.csv")))
    if not files:
        print(f"G2 FAIL: no params.csv under {prob}")
        return 1
    hwsd = pd.read_csv(args.hwsd_csv).set_index("sid")["awc"].astype(float)

    rows = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        batch = os.path.basename(os.path.dirname(f))
        for name, val in df["value"].items():
            if not str(name).startswith("aw_"):
                continue
            sid = str(name)[3:]
            h = float(hwsd.get(sid, np.nan))
            exp = expected_prior(h, args.lower, args.upper)
            rows.append(
                {
                    "batch": batch,
                    "site_id": sid,
                    "hwsd_awc_mm_per_m": h,
                    "prior_aw": float(val),
                    "expected_prior": exp,
                    "fallback_150": bool(not np.isfinite(h) or h < args.lower),
                    "match": bool(abs(float(val) - exp) < 1e-3),  # float32 storage
                }
            )
    tab = pd.DataFrame(rows).sort_values(["batch", "site_id"])
    out_dir = args.out_dir or prob
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "aw_prior_table.csv")
    tab.to_csv(out, index=False)

    problems = []
    if len(tab) != args.expect_sites:
        problems.append(f"{len(tab)} aw_* rows, expected {args.expect_sites}")
    if tab["prior_aw"].nunique() <= 1:
        problems.append(f"aw prior is a constant ({tab['prior_aw'].iloc[0]}) at every site")
    bad = tab.loc[~tab["match"]]
    if len(bad):
        problems.append(f"{len(bad)} priors do not match the HWSD rule: {bad['site_id'].tolist()}")
    summary = {
        "n_sites": int(len(tab)),
        "n_fallback_150": int(tab["fallback_150"].sum()),
        "fallback_sites": tab.loc[tab["fallback_150"], "site_id"].tolist(),
        "prior_min": float(tab["prior_aw"].min()),
        "prior_median": float(tab["prior_aw"].median()),
        "prior_max": float(tab["prior_aw"].max()),
        "n_unique": int(tab["prior_aw"].nunique()),
        "table": out,
        "problems": problems,
        "pass": not problems,
    }
    with open(os.path.join(out_dir, "aw_prior_gate_g2.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"G2 {'PASS' if not problems else 'FAIL'}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
