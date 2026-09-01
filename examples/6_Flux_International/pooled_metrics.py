"""True concatenated-pool SWIM vs RS metrics for Example 6.

Implements the Volk et al. (2024) pooled-comparison conventions directly:
  - Pooled Pearson r2 + linregress slope over the concatenated obs/mod arrays
    (one regression across all site-days / site-months, not a per-site mean).
  - sqrt(n)-weighted per-station MBE / MAE / RMSE.
  - A pooled KGE computed on the full concatenated pool (a diagnostic for paper
    Experiment 2), reusing evaluate.calc_metrics for an exact definition match.

Reads the per-site combined CSVs already written by ``evaluate.py`` (``et_act``
= SWIM daily ETa, ``et_rs`` = RS-derived daily ETa) and pairs them against the
QAQC flux ET_corr truth with the same gates as ``evaluate.py``:
  - daily: paired days require finite flux+SWIM+RS, >=10 per site
  - monthly: monthly sums where the month has >=20 valid daily flux obs;
    months require finite flux+SWIM+RS, >=6 per site

Usage:
    python pooled_metrics.py [--results-dir DIR]
"""

import argparse
import os

import numpy as np
import pandas as pd
from evaluate import calc_metrics, load_flux_et, load_flux_sources
from scipy import stats

from swimrs.calibrate.flux_utils import paired_monthly_sums, passes_site_minimum

RESULTS_DIR_DEFAULT = (
    "/data/ssd1/swim/6_Flux_International/results/6_Flux_International_LSEnsemble_POR_annual2yr"
)
MIN_DAILY_OBS = 10
MIN_MONTHLY_OBS = 6
MIN_DAILY_FOR_MONTHLY = 30
MIN_DAYS_PER_MONTH = 20


def sqrt_n_weighted_mean(values, counts):
    """sqrt(n)-weighted mean across stations (Volk methodology)."""
    w = np.sqrt(counts)
    return float(np.sum(values * w) / np.sum(w))


def pooled_stats(all_obs, all_mod):
    """Pooled Pearson r2 + slope on concatenated arrays (Volk methodology)."""
    mask = np.isfinite(all_obs) & np.isfinite(all_mod)
    obs, mod = all_obs[mask], all_mod[mask]
    if len(obs) < 10:
        return {"r2": np.nan, "slope": np.nan}
    r, _ = stats.pearsonr(obs, mod)
    slope, _, _, _, _ = stats.linregress(obs, mod)
    return {"r2": r**2, "slope": slope}


def _load_site(results_dir, fid):
    """Return the per-site combined CSV (et_act, et_rs) or None."""
    path = os.path.join(results_dir, f"{fid}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = df.index.normalize()
    return df


def _accumulate(results_dir, fids, monthly, flux_sources=None):
    """Pair each site daily/monthly; return per-station stats + pooled arrays."""
    flux_sources = flux_sources or {}
    station = {"swim": [], "rs": []}
    pooled_obs = {"swim": [], "rs": []}
    pooled_mod = {"swim": [], "rs": []}

    for fid in fids:
        flux = load_flux_et(fid, flux_sources.get(fid))
        if flux.empty or not passes_site_minimum(flux):
            continue
        site = _load_site(results_dir, fid)
        if site is None or "et_act" not in site or "et_rs" not in site:
            continue
        swim = site["et_act"]
        rs = site["et_rs"]

        if monthly:
            common = swim.index.intersection(flux.index)
            if len(common) < MIN_DAILY_FOR_MONTHLY:
                continue
            flux_d = flux.loc[common]
            swim_d = swim.loc[common]
            rs_d = rs.reindex(common)
            # sums over flux-valid days only, matching evaluate_monthly()
            s_m, f_m, r_m = paired_monthly_sums(
                swim_d, flux_d, rs_d, month_min_days=MIN_DAYS_PER_MONTH
            )
            obs, sv, rv = f_m.values, s_m.values, r_m.values
            min_obs = MIN_MONTHLY_OBS
        else:
            common = swim.index.intersection(flux.index)
            if len(common) < MIN_DAILY_OBS:
                continue
            obs = flux.loc[common].values
            sv = swim.loc[common].values
            rv = rs.reindex(common).values
            min_obs = MIN_DAILY_OBS

        mask = np.isfinite(obs) & np.isfinite(sv) & np.isfinite(rv)
        if int(mask.sum()) < min_obs:
            continue
        o, sv, rv = obs[mask], sv[mask], rv[mask]

        for name, mod in (("swim", sv), ("rs", rv)):
            station[name].append(
                {
                    "fid": fid,
                    "n": len(o),
                    "mbe": float(np.mean(mod - o)),
                    "mae": float(np.mean(np.abs(mod - o))),
                    "rmse": float(np.sqrt(np.mean((mod - o) ** 2))),
                }
            )
            pooled_obs[name].append(o)
            pooled_mod[name].append(mod)

    return station, pooled_obs, pooled_mod


def _report(label, unit, station, pooled_obs, pooled_mod):
    print(f"\n{'=' * 96}")
    print(f"{label}  (concatenated pool, SWIM vs RS)")
    print(f"{'=' * 96}")
    header = (
        f"{'Model':<6} {'Nsta':>5} {'Npts':>7} {'r2_pool':>8} {'slope':>6} "
        f"{'KGE_pool':>9} {'MBE_w':>9} {'MAE_w':>9} {'RMSE_w':>9}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for name in ("swim", "rs"):
        ss = station[name]
        if not ss:
            continue
        counts = np.array([s["n"] for s in ss])
        obs = np.concatenate(pooled_obs[name])
        mod = np.concatenate(pooled_mod[name])
        ps = pooled_stats(obs, mod)
        km = calc_metrics(obs, mod)  # KGE/r2/bias/rmse on the full pool
        mbe_w = sqrt_n_weighted_mean(np.array([s["mbe"] for s in ss]), counts)
        mae_w = sqrt_n_weighted_mean(np.array([s["mae"] for s in ss]), counts)
        rmse_w = sqrt_n_weighted_mean(np.array([s["rmse"] for s in ss]), counts)
        mean_obs = float(np.mean(obs))
        print(
            f"{name.upper():<6} {len(ss):>5} {int(counts.sum()):>7} {ps['r2']:>8.3f} "
            f"{ps['slope']:>6.3f} {km['kge']:>9.3f} "
            f"{mbe_w:>+8.3f} {mae_w:>8.3f} {rmse_w:>8.3f}"
        )
        rows.append(
            {
                "model": name,
                "n_stations": len(ss),
                "n_points": int(counts.sum()),
                "r2_pooled": ps["r2"],
                "slope": ps["slope"],
                "kge_pooled": km["kge"],
                "r2_pooled_r2score": km["r2"],
                "bias_pooled": km["bias"],
                "rmse_pooled": km["rmse"],
                "mbe_weighted": mbe_w,
                "mae_weighted": mae_w,
                "rmse_weighted": rmse_w,
                "mean_obs": mean_obs,
            }
        )
    if rows:
        print(f"\nMean observed flux ET: {rows[0]['mean_obs']:.2f} {unit}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", default=RESULTS_DIR_DEFAULT)
    parser.add_argument(
        "--shapefile",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "gis", "flux_crop_pub_66_150m.shp"
        ),
        help="Cohort shapefile declaring per-site flux network/et_col",
    )
    args = parser.parse_args()

    # Site list = the per-site CSVs evaluate.py already wrote.
    fids = sorted(
        f[:-4]
        for f in os.listdir(args.results_dir)
        if f.endswith(".csv") and not f.startswith("evaluation")
    )
    print(f"Results dir: {args.results_dir}")
    print(f"Sites with per-site CSVs: {len(fids)}")

    flux_sources = {}
    if os.path.exists(args.shapefile):
        flux_sources = load_flux_sources(args.shapefile)
    else:
        print(f"WARNING: shapefile not found ({args.shapefile}); flux resolves by search order")

    d_station, d_obs, d_mod = _accumulate(
        args.results_dir, fids, monthly=False, flux_sources=flux_sources
    )
    daily_df = _report("DAILY", "mm/day", d_station, d_obs, d_mod)
    daily_df.to_csv(os.path.join(args.results_dir, "pooled_metrics_daily.csv"), index=False)

    m_station, m_obs, m_mod = _accumulate(
        args.results_dir, fids, monthly=True, flux_sources=flux_sources
    )
    monthly_df = _report("MONTHLY", "mm/month", m_station, m_obs, m_mod)
    monthly_df.to_csv(os.path.join(args.results_dir, "pooled_metrics_monthly.csv"), index=False)


if __name__ == "__main__":
    main()
