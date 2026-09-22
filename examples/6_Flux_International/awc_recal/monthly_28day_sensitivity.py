"""S9.1 sensitivity: the E2 closure-pool monthly comparison with a 28-day (not 20-day) month rule.

Re-derives, from the frozen ``archive/6_evaluation/site_daily_timeseries/{fid}.csv`` series of
the canonical GrassBasis run and the closure-pool site list, the per-site monthly metrics of
SWIM and the interpolated Landsat ensemble against closure-corrected flux ET when a month must
carry at least ``--month-min-days`` flux-valid days (default 28; the paper's primary rule is
20). Sites keep the evaluator's gates (>= 30 daily overlap, >= 6 paired months, metrics finite
only with >= 10 months, ``calc_metrics``). The paired SWIM - benchmark difference gets the
closure-pool site bootstrap (``bootstrap_paired_median``, default 10,000 resamples, seed
20260908).

Outputs (``<closure_pool>/monthly_28day_sensitivity_persite.csv``, ``..._summary.csv``,
``..._metadata.json``). Read-only on the archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import ex6_paths  # noqa: E402
from closure_pool_summary import bootstrap_paired_median  # noqa: E402
from evaluate import calc_metrics  # noqa: E402

from swimrs.calibrate.flux_utils import paired_monthly_sums  # noqa: E402

DEFAULT_RUN = ex6_paths.CANONICAL_RUN
METRICS = ["kge", "rmse", "bias", "r2", "r"]


def site_monthly(ts_path: Path, month_min_days: int) -> dict | None:
    ts = pd.read_csv(ts_path, index_col="date", parse_dates=True)
    flux = ts["flux_ET"].dropna()
    if len(flux) < 30:
        return None
    swim = ts["swim_ET"].reindex(flux.index)
    rs = ts["benchmark_ET"].reindex(flux.index)
    s_m, f_m, r_m = paired_monthly_sums(swim, flux, rs, month_min_days=month_min_days)
    idx = f_m.index
    r_on = r_m.reindex(idx)
    mask = f_m.notna() & s_m.reindex(idx).notna() & r_on.notna()
    months = idx[mask]
    if len(months) < 6:
        return None
    obs = f_m.loc[months].to_numpy(float)
    m_s = calc_metrics(obs, s_m.reindex(months).to_numpy(float))
    m_r = calc_metrics(obs, r_on.loc[months].to_numpy(float))
    row = {"n_months": int(len(months))}
    for k in METRICS:
        row[f"{k}_swim"] = m_s[k]
        row[f"{k}_rs"] = m_r[k]
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ex6_paths.CANONICAL_CONFIG), help="Run TOML")
    ap.add_argument("--run-name", default=DEFAULT_RUN)
    ap.add_argument("--results-root", default=None, help="default {project_ws}/results")
    ap.add_argument(
        "--closure-pool", default=None, help="default <archive>/6_evaluation/closure_pool"
    )
    ap.add_argument("--month-min-days", type=int, default=28)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--out", default=None, help="default: the closure-pool directory")
    args = ap.parse_args()

    if args.results_root is None:
        args.results_root = str(ex6_paths.results_root(ex6_paths.load_config(args.config)))
    cat6 = Path(args.results_root) / args.run_name / "archive" / "6_evaluation"
    cp = Path(args.closure_pool) if args.closure_pool else cat6 / "closure_pool"
    out = Path(args.out) if args.out else cp
    out.mkdir(parents=True, exist_ok=True)
    ts_dir = cat6 / "site_daily_timeseries"
    sites = pd.read_csv(cp / "closure_pool_sites.csv", index_col="fid")
    pool = list(sites.index)
    primary = pd.read_csv(cat6 / "monthly_paired_metrics.csv", index_col="fid")
    n_primary_finite = int(primary["kge_swim"].reindex(pool).notna().sum())

    rows = []
    for fid in pool:
        r = site_monthly(ts_dir / f"{fid}.csv", args.month_min_days)
        if r is None:
            continue
        rows.append({"fid": fid, **r})
    persite = pd.DataFrame(rows).set_index("fid")
    finite = persite.loc[persite["kge_swim"].notna()]
    dropped = sorted(set(pool) - set(finite.index))

    rng = np.random.default_rng(args.seed)
    summ = []
    for k in METRICS:
        d = (finite[f"{k}_swim"] - finite[f"{k}_rs"]).to_numpy(float)
        summ.append(
            {
                "metric": k,
                "month_min_days": args.month_min_days,
                "swim_median": float(finite[f"{k}_swim"].median()),
                "rs_median": float(finite[f"{k}_rs"].median()),
                **bootstrap_paired_median(d, rng, args.reps),
            }
        )
    summary = pd.DataFrame(summ)
    tag = f"monthly_{args.month_min_days}day_sensitivity"
    persite.to_csv(out / f"{tag}_persite.csv")
    summary.to_csv(out / f"{tag}_summary.csv", index=False)
    meta = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_name": args.run_name,
        "month_min_days": args.month_min_days,
        "min_months": 6,
        "metrics_finite_min_months": 10,
        "reps": args.reps,
        "seed": args.seed,
        "n_pool_sites": len(pool),
        "n_sites_primary_finite_20day": n_primary_finite,
        "n_sites_finite": int(len(finite)),
        "sites_dropped_vs_pool": dropped,
        "sites_dropped_vs_primary": sorted(
            set(primary.index[primary["kge_swim"].notna()]).intersection(pool) - set(finite.index)
        ),
        "closure_pool_sites_sha256": hashlib.sha256(
            (cp / "closure_pool_sites.csv").read_bytes()
        ).hexdigest(),
    }
    with open(out / f"{tag}_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))
    print(summary.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
