"""E2 closure-corrected pool: re-cut of the Cat 6 evaluation archive on the 47 EBR sites.

Decision 2026-09-08: the paper's E2 evaluation pool is the closure-corrected towers only
(``flux_et_col == "ET_corr"``: energy-balance-ratio corrected LE, flux-data-qaqc strict branch).
The 16 raw-ET towers (no Rn and/or G, so no closure correction was possible) do not appear in
the paper. Calibration ran on the full 66-site cohort and is not redone here.

Every table is re-derived from the frozen per-site files under
``archive/6_evaluation`` (``daily_paired_metrics.csv``, ``monthly_paired_metrics.csv``,
``overpass_split_metrics.csv``, ``internal_baseline_comparison.csv``,
``derived_uncal_baseline_persite.csv``, ``site_daily_timeseries/``) and the transfer refresh
directory. No forward run, no evaluator rerun, no calibration. The per-member benchmark table is
the only step that opens the container (read-only) — it re-pairs SWIM against each ensemble
member on the pool via ``derived_metrics.collect``; skip it with ``--skip-members``.

Outputs (``<archive>/6_evaluation/closure_pool/`` by default):

* ``closure_pool_sites.csv``        — the pool with country/continent (from the site-id prefix,
                                       which also fixes the ``US``/``USA`` label defect),
                                       region, irrigation class, paired days/months
* ``headline_aggregates.csv``       — medians/means for SWIM, benchmark and the paired
                                       difference; ``tier`` in {closure_corrected, raw, all}
* ``paired_delta_bootstrap.csv``    — median paired SWIM − benchmark difference with a 95 %
                                       site-bootstrap interval, per basis and tier
* ``group_summary_metrics.csv``     — medians by region / country / continent / irrigation
                                       class / land cover within the pool
* ``overpass_split_summary.csv``    — retrieval-day vs between-retrieval medians on the pool
* ``transfer_refresh_summary.csv``  — the five transfer configurations re-cut on the pool
                                       (daily; monthly where per-site files exist)
* ``transfer_winrates.csv``         — Ex5-transferred win rates vs each comparator on the pool
* ``pooled_metrics_daily.csv`` / ``pooled_metrics_monthly.csv`` — concatenated-pool and
                                       √n-weighted station metrics (Volk methodology)
* ``internal_baseline_comparison_summary.csv`` — GrassBasis vs the frozen baseline on the pool
* ``uncalibrated_baseline_summary.csv`` — default-parameter forward baseline on the pool
* ``classifier_transition_summary.csv`` — metric change by classifier transition on the pool
* ``derived_per_model_benchmarks.csv`` / ``derived_murphy_decomp.csv`` — per-member tables
* ``closure_pool_metadata.json``    — counts, irrigation site-years, input hashes, git sha

Usage::

    uv run --project /home/dgketchum/code/swim-rs python \\
        examples/6_Flux_International/closure_pool_summary.py \\
        [--verify-all-sites] [--skip-members]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ex6_paths  # noqa: E402
from evaluate import calc_metrics  # noqa: E402
from evaluation_summary import (  # noqa: E402
    METRICS,
    MIN_DAILY,
    MIN_DAYS_PER_MONTH,
    MIN_MONTHS,
    group_medians,
)
from pooled_metrics import pooled_stats, sqrt_n_weighted_mean  # noqa: E402

from swimrs.calibrate.flux_utils import paired_monthly_sums  # noqa: E402

REPO = ex6_paths.REPO
DEFAULT_CONFIG = ex6_paths.CANONICAL_CONFIG
DEFAULT_RUN_NAME = ex6_paths.CANONICAL_RUN
TRANSFER_NEW_NAME = "e2_run22_transfer_by_irrigation_to_grassbasis"  # under {project_ws}/results
TRANSITION_NAME = "irrigation_classifier_transition.csv"  # under the QA root

CLOSURE_TIER = {"ET_corr": "closure_corrected", "ET": "raw"}
PRIMARY_TIER = "closure_corrected"

# FLUXNET-style site ids carry the ISO 3166-1 alpha-2 country code as their prefix.
COUNTRY_BY_PREFIX = {
    "AR": "Argentina",
    "AU": "Australia",
    "BE": "Belgium",
    "CA": "Canada",
    "CH": "Switzerland",
    "CR": "Costa Rica",
    "DE": "Germany",
    "FR": "France",
    "IT": "Italy",
    "US": "United States",
}
CONTINENT_BY_COUNTRY = {
    "Argentina": "South America",
    "Australia": "Oceania",
    "Belgium": "Europe",
    "Canada": "North America",
    "Switzerland": "Europe",
    "Costa Rica": "North America",
    "Germany": "Europe",
    "France": "Europe",
    "Italy": "Europe",
    "United States": "North America",
}

TRANSFER_CONFIGS = {
    "e3_uncal": "E3 uncalibrated/default",
    "ex5_transfer": "Ex5 transferred",
    "ex5_transfer_strat": "Ex5 stratified transfer",
    "e3_cal": "E3 calibrated",
    "ls_ensemble": "LS ensemble",
}
TRANSFER_METRICS = ["kge", "r2", "rmse", "bias", "mae"]


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def site_geography(fids) -> pd.DataFrame:
    """Country and continent for each site id, from its ISO prefix (raises on an unknown one)."""
    rows = []
    for fid in fids:
        prefix = str(fid).split("-")[0]
        if prefix not in COUNTRY_BY_PREFIX:
            raise KeyError(f"no country mapping for site prefix {prefix!r} ({fid})")
        country = COUNTRY_BY_PREFIX[prefix]
        rows.append({"fid": fid, "country": country, "continent": CONTINENT_BY_COUNTRY[country]})
    return pd.DataFrame(rows).set_index("fid")


def closure_tier(daily: pd.DataFrame) -> pd.Series:
    """Map the archived ``flux_et_col`` to a closure tier; raises on an unmapped column."""
    unknown = sorted(set(daily["flux_et_col"]) - set(CLOSURE_TIER))
    if unknown:
        raise ValueError(f"unmapped flux_et_col values: {unknown}")
    return daily["flux_et_col"].map(CLOSURE_TIER).rename("closure_tier")


def select_pool(daily: pd.DataFrame, tier: str = PRIMARY_TIER) -> list[str]:
    tiers = closure_tier(daily)
    return sorted(tiers.index[tiers == tier].tolist())


def headline_rows(frame: pd.DataFrame, basis: str, tier: str) -> list[dict]:
    """Medians and means of every metric for SWIM, the benchmark, and the paired difference.

    ``n_sites`` counts rows with a finite SWIM KGE (monthly frames carry NaN-metric rows for
    sites that pass the month gate with fewer than 10 months); ``n_rows`` is the frame length.
    """
    rows = []
    finite = frame["kge_swim"].notna() & frame["kge_rs"].notna()
    for model in ("swim", "rs"):
        row = {
            "basis": basis,
            "tier": tier,
            "model": model,
            "n_rows": int(len(frame)),
            "n_sites": int(finite.sum()),
        }
        for k in METRICS:
            row[f"{k}_median"] = float(frame[f"{k}_{model}"].median())
            row[f"{k}_mean"] = float(frame[f"{k}_{model}"].mean())
        rows.append(row)
    row = {
        "basis": basis,
        "tier": tier,
        "model": "swim_minus_rs_paired",
        "n_rows": int(len(frame)),
        "n_sites": int(finite.sum()),
    }
    for k in METRICS:
        d = frame[f"{k}_swim"] - frame[f"{k}_rs"]
        row[f"{k}_median"] = float(d.median())
        row[f"{k}_mean"] = float(d.mean())
    rows.append(row)
    return rows


def bootstrap_paired_median(deltas: np.ndarray, rng: np.random.Generator, reps: int) -> dict:
    """Median paired difference with a 95 % percentile site-bootstrap interval."""
    d = np.asarray(deltas, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return {"n_sites": 0, "median": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    medians = np.median(d[rng.integers(0, n, size=(reps, n))], axis=1)
    return {
        "n_sites": int(n),
        "median": float(np.median(d)),
        "ci_low": float(np.percentile(medians, 2.5)),
        "ci_high": float(np.percentile(medians, 97.5)),
        "frac_sites_swim_higher": float(np.mean(d > 0)),
    }


def paired_delta_table(frame: pd.DataFrame, basis: str, tier: str, reps: int, seed: int):
    """One bootstrap row per metric; the RNG is seeded once per (basis, tier) block."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in METRICS:
        d = (frame[f"{k}_swim"] - frame[f"{k}_rs"]).to_numpy(float)
        rows.append(
            {"basis": basis, "tier": tier, "metric": k, **bootstrap_paired_median(d, rng, reps)}
        )
    return pd.DataFrame(rows)


def overpass_summary(split: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in METRICS:
        rows.append(
            {
                "metric": k,
                "n_sites": int(len(split)),
                "swim_retrieval_median": float(split[f"{k}_swim_retrieval"].median()),
                "rs_retrieval_median": float(split[f"{k}_rs_retrieval"].median()),
                "swim_between_median": float(split[f"{k}_swim_between"].median()),
                "rs_between_median": float(split[f"{k}_rs_between"].median()),
                "paired_swim_minus_rs_retrieval_median": float(
                    split[f"{k}_swim_minus_rs_retrieval"].median()
                ),
                "paired_swim_minus_rs_between_median": float(
                    split[f"{k}_swim_minus_rs_between"].median()
                ),
                "support_interaction_median": float(split[f"{k}_support_interaction"].median()),
            }
        )
    return pd.DataFrame(rows)


def transfer_daily_summary(persite: pd.DataFrame) -> pd.DataFrame:
    """Median metrics of each transfer configuration (``<cfg>_<metric>`` columns) on a pool."""
    rows = []
    for key, label in TRANSFER_CONFIGS.items():
        cols = {m: f"{key}_{m}" for m in TRANSFER_METRICS}
        if not all(c in persite.columns for c in cols.values()):
            continue
        row = {
            "config": label,
            "basis": "daily",
            "n_sites": int(persite[cols["kge"]].notna().sum()),
        }
        for m, c in cols.items():
            row[f"{m}_med"] = float(persite[c].median())
        rows.append(row)
    return pd.DataFrame(rows)


def transfer_monthly_summary(series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Median monthly metrics for configurations with a per-site monthly frame.

    ``series`` maps config label -> frame with ``kge``, ``r2``, ``rmse``, ``bias`` columns
    (``mae`` when present) indexed by site; the frame is already restricted to the pool.
    """
    rows = []
    for label, frame in series.items():
        finite = frame["kge"].notna()
        row = {"config": label, "basis": "monthly", "n_sites": int(finite.sum())}
        for m in TRANSFER_METRICS:
            row[f"{m}_med"] = float(frame[m].median()) if m in frame else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _suffix_frame(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    """Pick the ``<metric>_<model>`` columns of a per-site frame and strip the suffix."""
    cols = {f"{m}_{model}": m for m in TRANSFER_METRICS if f"{m}_{model}" in frame.columns}
    return frame[list(cols)].rename(columns=cols)


def win_rate(ref: pd.Series, other: pd.Series) -> tuple[float, int]:
    """Fraction of common sites where ``ref`` is strictly higher (both finite)."""
    common = ref.index.intersection(other.index)
    a, b = ref.loc[common], other.loc[common]
    valid = a.notna() & b.notna()
    n = int(valid.sum())
    if n == 0:
        return np.nan, 0
    return float((a[valid] > b[valid]).mean()), n


def transfer_winrates(persite: pd.DataFrame, monthly: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ref_label = TRANSFER_CONFIGS["ex5_transfer"]
    rows = []
    for key, label in TRANSFER_CONFIGS.items():
        if key == "ex5_transfer":
            continue
        row = {"reference": ref_label, "comparison": f"{ref_label} vs {label}"}
        for m in ("r2", "kge"):
            w, n = win_rate(persite[f"ex5_transfer_{m}"], persite[f"{key}_{m}"])
            row["n_daily"] = n
            row[f"daily_{m}_win"] = w
        if ref_label in monthly and label in monthly:
            for m in ("r2", "kge"):
                w, n = win_rate(monthly[ref_label][m], monthly[label][m])
                row["n_monthly"] = n
                row[f"monthly_{m}_win"] = w
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_from_timeseries(ts_dir: Path, fids, monthly: bool) -> pd.DataFrame:
    """Concatenated-pool and √n-weighted station metrics from the archived daily series.

    Mirrors ``pooled_metrics.py``: daily pairs are days with finite flux, SWIM and benchmark ET;
    monthly pairs are ``paired_monthly_sums`` totals over flux-valid days (≥ 20 days/month,
    ≥ 6 months per site, benchmark month NaN unless complete).
    """
    station = {"swim": [], "rs": []}
    pooled_obs = {"swim": [], "rs": []}
    pooled_mod = {"swim": [], "rs": []}
    for fid in fids:
        ts = pd.read_csv(ts_dir / f"{fid}.csv", index_col="date", parse_dates=True)
        flux = ts["flux_ET"].dropna()
        swim = ts["swim_ET"].reindex(flux.index)
        rs = ts["benchmark_ET"].reindex(flux.index)
        if monthly:
            if len(flux) < 30:
                continue
            s_m, f_m, r_m = paired_monthly_sums(swim, flux, rs, month_min_days=MIN_DAYS_PER_MONTH)
            obs, sv, rv = f_m.to_numpy(float), s_m.to_numpy(float), r_m.to_numpy(float)
            min_obs = MIN_MONTHS
        else:
            obs, sv, rv = flux.to_numpy(float), swim.to_numpy(float), rs.to_numpy(float)
            min_obs = MIN_DAILY
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
    rows = []
    for name in ("swim", "rs"):
        ss = station[name]
        if not ss:
            continue
        counts = np.array([s["n"] for s in ss])
        obs = np.concatenate(pooled_obs[name])
        mod = np.concatenate(pooled_mod[name])
        ps = pooled_stats(obs, mod)
        km = calc_metrics(obs, mod)
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
                "mbe_weighted": sqrt_n_weighted_mean(np.array([s["mbe"] for s in ss]), counts),
                "mae_weighted": sqrt_n_weighted_mean(np.array([s["mae"] for s in ss]), counts),
                "rmse_weighted": sqrt_n_weighted_mean(np.array([s["rmse"] for s in ss]), counts),
                "mean_obs": float(np.mean(obs)),
            }
        )
    return pd.DataFrame(rows)


def baseline_comparison_summary(comp: pd.DataFrame) -> pd.DataFrame:
    """Per-series medians and paired differences of ``internal_baseline_comparison.csv`` rows."""
    out = []
    for basis, suffix in (("daily", ""), ("monthly", "_monthly")):
        for name in [
            "grassbasis_swim",
            "baseline_it3_swim",
            "baseline_it4_swim",
            "grassbasis_rs",
            "baseline_rs",
        ]:
            col = f"kge_{name}{suffix}"
            if col not in comp:
                continue
            row = {"basis": basis, "series": name, "n_sites": int(comp[col].notna().sum())}
            for k in METRICS:
                row[f"{k}_median"] = float(comp[f"{k}_{name}{suffix}"].median())
            out.append(row)
        for a, b in (
            ("grassbasis_swim", "baseline_it3_swim"),
            ("grassbasis_swim", "baseline_it4_swim"),
            ("grassbasis_rs", "baseline_rs"),
        ):
            if f"kge_{b}{suffix}" not in comp:
                continue
            row = {
                "basis": basis,
                "series": f"{a}_minus_{b}_paired",
                "n_sites": int((comp[f"kge_{a}{suffix}"] - comp[f"kge_{b}{suffix}"]).notna().sum()),
            }
            for k in METRICS:
                row[f"{k}_median"] = float(
                    (comp[f"{k}_{a}{suffix}"] - comp[f"{k}_{b}{suffix}"]).median()
                )
            out.append(row)
    return pd.DataFrame(out)


def irrigation_counts(transition: pd.DataFrame, fids) -> dict:
    """Ever-irrigated sites and irrigated site-years (corrected classifier) over the modelled years."""
    t = transition.loc[transition["site"].isin(list(fids))]
    per_site = t.groupby("site")["irrigated_corrected"].agg(["sum", "count"])
    return {
        "n_sites": int(len(per_site)),
        "n_site_years_modelled": int(per_site["count"].sum()),
        "ever_irrigated_sites": int((per_site["sum"] > 0).sum()),
        "ever_irrigated_site_ids": sorted(per_site.index[per_site["sum"] > 0].tolist()),
        "irrigated_site_years": int(per_site["sum"].sum()),
    }


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# container-backed per-member benchmarks
# ---------------------------------------------------------------------------


def member_benchmarks(config_path: Path, results_dir: Path, fids):
    """SWIM vs each ensemble member and the Murphy decomposition, paired on the pool."""
    import derived_metrics as dm
    import evaluate as ev

    from swimrs.container import SwimContainer

    cfg = ev._load_config(config_path)
    container = SwimContainer.open(ev._default_container_path(cfg), mode="r")
    try:
        specs = dm._rs_model_specs(container, cfg)
        flux_sources = ev.load_flux_sources(cfg.fields_shapefile, cfg.feature_id_col)
        per_site = dm.collect(cfg, container, str(results_dir), list(fids), specs, flux_sources)
    finally:
        container.close()
    return dm.per_model_benchmarks(per_site, specs), dm.murphy(per_site, specs), len(per_site)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    p.add_argument("--results-root", default=None, help="default {project_ws}/results")
    p.add_argument(
        "--transfer-new", default=None, help=f"default <results-root>/{TRANSFER_NEW_NAME}"
    )
    p.add_argument("--transition-csv", default=None, help=f"default <qa-root>/{TRANSITION_NAME}")
    p.add_argument("--out", default=None, help="default <archive>/6_evaluation/closure_pool")
    p.add_argument("--reps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument(
        "--skip-members", action="store_true", help="skip the container-backed member table"
    )
    p.add_argument(
        "--verify-all-sites",
        action="store_true",
        help="recompute the all-site pooled table from the time series and compare to the archive",
    )
    args = p.parse_args()

    if None in (args.results_root, args.transfer_new, args.transition_csv):
        cfg = ex6_paths.load_config(Path(args.config))
        results_root = Path(args.results_root) if args.results_root else ex6_paths.results_root(cfg)
        args.results_root = str(results_root)
        args.transfer_new = args.transfer_new or str(results_root / TRANSFER_NEW_NAME)
        args.transition_csv = args.transition_csv or str(ex6_paths.qa_root(cfg) / TRANSITION_NAME)
    results_dir = Path(args.results_root) / args.run_name
    cat6 = results_dir / "archive" / "6_evaluation"
    out = Path(args.out) if args.out else cat6 / "closure_pool"
    out.mkdir(parents=True, exist_ok=True)
    t_new = Path(args.transfer_new)
    problems = []

    daily_all = pd.read_csv(cat6 / "daily_paired_metrics.csv", index_col="fid")
    monthly_all = pd.read_csv(cat6 / "monthly_paired_metrics.csv", index_col="fid")
    groups_all = pd.read_csv(cat6 / "site_groups.csv", index_col="site_id")
    tiers = closure_tier(daily_all)
    pool = select_pool(daily_all)
    raw = select_pool(daily_all, "raw")
    print(f"daily-scored sites {len(daily_all)}: {len(pool)} closure-corrected, {len(raw)} raw")

    daily = daily_all.loc[pool]
    monthly = monthly_all.loc[monthly_all.index.intersection(pool)]
    geo = site_geography(daily_all.index)

    # ---- site listing ----
    sites = pd.DataFrame(index=pd.Index(pool, name="fid"))
    sites["flux_network"] = daily["flux_network"]
    sites["flux_et_col"] = daily["flux_et_col"]
    sites["closure_tier"] = tiers.loc[pool]
    sites["country"] = geo.loc[pool, "country"]
    sites["continent"] = geo.loc[pool, "continent"]
    sites["region"] = groups_all.loc[pool, "region"]
    sites["lulc"] = groups_all.loc[pool, "lulc"]
    sites["irrigation_class"] = groups_all.loc[pool, "irrigation_class"]
    sites["n_paired_days"] = daily["n"]
    sites["n_paired_months"] = monthly_all["n"].reindex(pool)
    sites["monthly_metrics_finite"] = monthly_all["kge_swim"].reindex(pool).notna()
    sites["kge_swim_daily"] = daily["kge_swim"]
    sites["kge_rs_daily"] = daily["kge_rs"]
    sites["bias_swim_daily"] = daily["bias_swim"]
    sites["bias_rs_daily"] = daily["bias_rs"]
    sites.to_csv(out / "closure_pool_sites.csv")
    dropped = pd.DataFrame(index=pd.Index(raw, name="fid"))
    dropped["flux_network"] = daily_all.loc[raw, "flux_network"]
    dropped["country"] = geo.loc[raw, "country"]
    dropped["continent"] = geo.loc[raw, "continent"]
    dropped["n_paired_days"] = daily_all.loc[raw, "n"]
    dropped.to_csv(out / "excluded_raw_tier_sites.csv")

    # ---- headline + bootstrap ----
    frames = {
        ("daily", PRIMARY_TIER): daily,
        ("daily", "raw"): daily_all.loc[raw],
        ("daily", "all"): daily_all,
        ("monthly", PRIMARY_TIER): monthly,
        ("monthly", "raw"): monthly_all.loc[monthly_all.index.intersection(raw)],
        ("monthly", "all"): monthly_all,
    }
    headline = pd.DataFrame(
        [r for (basis, tier), f in frames.items() for r in headline_rows(f, basis, tier)]
    )
    headline.to_csv(out / "headline_aggregates.csv", index=False)
    boot = pd.concat(
        [
            paired_delta_table(f, basis, tier, args.reps, args.seed)
            for (basis, tier), f in frames.items()
        ]
    )
    boot.to_csv(out / "paired_delta_bootstrap.csv", index=False)

    # ---- groups within the pool (country normalised from the site prefix) ----
    groups = groups_all.loc[pool, ["region", "lulc", "irrigation_class"]].copy()
    groups["country"] = geo.loc[pool, "country"]
    groups["continent"] = geo.loc[pool, "continent"]
    grp = pd.concat(
        [group_medians(daily, groups, "daily"), group_medians(monthly, groups, "monthly")]
    )
    grp.to_csv(out / "group_summary_metrics.csv", index=False)

    # ---- overpass split ----
    split = pd.read_csv(cat6 / "overpass_split_metrics.csv", index_col="fid")
    overpass_summary(split.loc[split.index.intersection(pool)]).to_csv(
        out / "overpass_split_summary.csv", index=False
    )

    # ---- transfer refresh ----
    persite = pd.read_csv(t_new / "transfer_comparison_persite.csv", index_col=0)
    persite = persite.loc[persite.index.intersection(pool)]
    monthly_series = {}
    for key, fname in (
        ("ex5_transfer", "evaluation_monthly_metrics.csv"),
        ("ex5_transfer_strat", "evaluation_monthly_metrics_strat.csv"),
    ):
        f = t_new / fname
        if f.exists():
            m = pd.read_csv(f, index_col=0)
            monthly_series[TRANSFER_CONFIGS[key]] = _suffix_frame(
                m.loc[m.index.intersection(pool)], "swim"
            )
        else:
            problems.append(f"transfer monthly file missing: {f}")
    monthly_series[TRANSFER_CONFIGS["e3_cal"]] = _suffix_frame(monthly, "swim")
    monthly_series[TRANSFER_CONFIGS["ls_ensemble"]] = _suffix_frame(monthly, "rs")
    transfer = pd.concat(
        [transfer_daily_summary(persite), transfer_monthly_summary(monthly_series)]
    )
    transfer.insert(0, "application", "grassbasis_refresh_closure_pool")
    transfer.to_csv(out / "transfer_refresh_summary.csv", index=False)
    transfer_winrates(persite, monthly_series).to_csv(out / "transfer_winrates.csv", index=False)

    # ---- pooled (Volk-style) ----
    ts_dir = cat6 / "site_daily_timeseries"
    pooled_d = pooled_from_timeseries(ts_dir, pool, monthly=False)
    pooled_m = pooled_from_timeseries(ts_dir, pool, monthly=True)
    pooled_d.to_csv(out / "pooled_metrics_daily.csv", index=False)
    pooled_m.to_csv(out / "pooled_metrics_monthly.csv", index=False)
    verify = {}
    if args.verify_all_sites:
        for basis, fname in (
            ("daily", "pooled_metrics_daily.csv"),
            ("monthly", "pooled_metrics_monthly.csv"),
        ):
            mine = pooled_from_timeseries(
                ts_dir, daily_all.index, monthly=basis == "monthly"
            ).set_index("model")
            ref = pd.read_csv(cat6 / fname).set_index("model")
            common = [c for c in ref.columns if c in mine.columns]
            delta = (mine[common] - ref[common]).abs()
            verify[basis] = {
                "n_points_match": bool((mine["n_points"] == ref["n_points"]).all()),
                "n_stations_match": bool((mine["n_stations"] == ref["n_stations"]).all()),
                "max_abs_delta": float(delta.drop(columns=["n_points", "n_stations"]).max().max()),
            }
            if not verify[basis]["n_points_match"] or verify[basis]["max_abs_delta"] > 1e-3:
                problems.append(
                    f"all-site pooled {basis} does not reproduce the archive: {verify[basis]}"
                )

    # ---- internal baseline comparison ----
    comp = pd.read_csv(cat6 / "internal_baseline_comparison.csv", index_col="fid")
    baseline_comparison_summary(comp.loc[comp.index.intersection(pool)]).to_csv(
        out / "internal_baseline_comparison_summary.csv", index=False
    )

    # ---- uncalibrated default-parameter baseline ----
    # Two forward runs of the default-parameter model exist in the archive, scored on different
    # day sets: transfer_ex5_params.py (``e3_uncal``, the rs-gated mask shared with every other
    # row of the closure pool: flux, SWIM and the Landsat-ensemble benchmark all finite) and
    # derived_metrics.py (``derived_uncal_baseline_persite.csv``, flux + SWIM finite only). The
    # CANONICAL row is the rs-gated one, so the uncalibrated model is scored on exactly the days
    # as the calibrated model, the benchmark and the transfers it is compared with (HANDOFF
    # HWSD_AWC_UNITS_RECAL section 6.5). The flux+SWIM-mask summary is kept as a secondary file.
    unc_rows = []
    for state, key in (("cal", "e3_cal"), ("uncal", "e3_uncal")):
        cols = {k: f"{key}_{k}" for k in ("r2", "rmse", "bias", "kge")}
        row = {
            "series": state,
            "forward_run": "transfer_ex5_params.py",
            "mask": "rs_gated_paired_days",
            "n_sites": int(persite[cols["kge"]].notna().sum()),
        }
        for k, c in cols.items():
            row[f"{k}_median"] = float(persite[c].median())
        unc_rows.append(row)
    row = {
        "series": "cal_minus_uncal_paired",
        "forward_run": "transfer_ex5_params.py",
        "mask": "rs_gated_paired_days",
        "n_sites": int(len(persite)),
    }
    for k in ("r2", "rmse", "bias", "kge"):
        row[f"{k}_median"] = float((persite[f"e3_cal_{k}"] - persite[f"e3_uncal_{k}"]).median())
    unc_rows.append(row)
    pd.DataFrame(unc_rows).to_csv(out / "uncalibrated_baseline_summary.csv", index=False)

    unc = pd.read_csv(cat6 / "derived_uncal_baseline_persite.csv", index_col="fid")
    unc = unc.loc[unc.index.intersection(pool)]
    unc_rows = []
    for state in ("cal", "uncal"):
        row = {
            "series": state,
            "forward_run": "derived_metrics.py",
            "mask": "flux_and_swim_finite",
            "n_sites": int(unc[f"kge_{state}"].notna().sum()),
        }
        for k in ("r2", "rmse", "bias", "kge"):
            row[f"{k}_median"] = float(unc[f"{k}_{state}"].median())
        unc_rows.append(row)
    row = {
        "series": "cal_minus_uncal_paired",
        "forward_run": "derived_metrics.py",
        "mask": "flux_and_swim_finite",
        "n_sites": int(len(unc)),
    }
    for k in ("r2", "rmse", "bias", "kge"):
        row[f"{k}_median"] = float((unc[f"{k}_cal"] - unc[f"{k}_uncal"]).median())
    unc_rows.append(row)
    pd.DataFrame(unc_rows).to_csv(
        out / "uncalibrated_baseline_summary_flux_swim_mask.csv", index=False
    )
    both = persite.join(unc, how="inner")
    uncal_mask_delta = {
        k: float((both[f"e3_uncal_{k}"] - both[f"{k}_uncal"]).abs().max())
        for k in ("r2", "rmse", "bias", "kge")
    }
    if (persite["e3_uncal_kge"].notna().sum()) != len(pool):
        problems.append(
            f"canonical uncalibrated row scored {int(persite['e3_uncal_kge'].notna().sum())} "
            f"sites, pool has {len(pool)}"
        )

    # ---- classifier transitions ----
    trv = pd.read_csv(cat6 / "classifier_transition_vs_metrics.csv", index_col="site")
    trv = trv.loc[trv.index.intersection(pool)]
    tr_summary = trv.groupby("transition").agg(
        n_sites=("n_years", "count"),
        **{
            c: (c, "median") for c in trv.columns if c.startswith(("kge_", "r2_", "bias_", "rmse_"))
        },
    )
    tr_summary.to_csv(out / "classifier_transition_summary.csv")

    # ---- per-member benchmarks (container read-only) ----
    n_member_sites = None
    if not args.skip_members:
        bench, mur, n_member_sites = member_benchmarks(Path(args.config), results_dir, pool)
        bench.to_csv(out / "derived_per_model_benchmarks.csv", index=False)
        mur.to_csv(out / "derived_murphy_decomp.csv", index=False)
        if n_member_sites != len(pool):
            problems.append(f"member pairing scored {n_member_sites} sites, pool has {len(pool)}")

    # ---- metadata ----
    transition = pd.read_csv(args.transition_csv)
    inputs = [
        cat6 / "daily_paired_metrics.csv",
        cat6 / "monthly_paired_metrics.csv",
        cat6 / "overpass_split_metrics.csv",
        cat6 / "internal_baseline_comparison.csv",
        cat6 / "derived_uncal_baseline_persite.csv",
        cat6 / "classifier_transition_vs_metrics.csv",
        cat6 / "site_groups.csv",
        t_new / "transfer_comparison_persite.csv",
        Path(args.transition_csv),
    ]
    meta = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "decision": (
            "2026-09-08: E2 evaluation pool = closure-corrected (EBR) towers only; the raw-ET tier "
            "does not appear in the paper. Calibration ran on the full 66-site cohort."
        ),
        "pool_definition": {"column": "flux_et_col", "value": "ET_corr", "tier": PRIMARY_TIER},
        "n_daily_sites_all": int(len(daily_all)),
        "n_daily_sites_pool": int(len(pool)),
        "n_daily_sites_raw_excluded": int(len(raw)),
        "n_monthly_rows_pool": int(len(monthly)),
        "n_monthly_finite_pool": int(monthly["kge_swim"].notna().sum()),
        "n_paired_days_pool": int(daily["n"].sum()),
        "n_paired_days_all": int(daily_all["n"].sum()),
        "countries_pool": sites["country"].value_counts().to_dict(),
        "continents_pool": sites["continent"].value_counts().to_dict(),
        "regions_pool": sites["region"].value_counts().to_dict(),
        "irrigation_class_pool": sites["irrigation_class"].value_counts().to_dict(),
        "irrigation_counts_pool": irrigation_counts(transition, pool),
        "irrigation_counts_all_cohort": irrigation_counts(transition, transition["site"].unique()),
        "configured_pool_note": (
            "US-Ne1/US-Ne2/US-Ne3 are closure-corrected in the cohort shapefile but have no paired "
            "days in the modelled period, so the configured closure-corrected pool is 50 sites "
            "and the evaluated pool is 47."
        ),
        "bootstrap": {
            "replicates": args.reps,
            "seed": args.seed,
            "rng_scope": "one RNG per (basis, tier) block",
        },
        "member_benchmark_sites": n_member_sites,
        "uncalibrated_baseline": {
            "canonical_file": "uncalibrated_baseline_summary.csv",
            "canonical_forward_run": (
                "transfer_ex5_params.py e3_uncal (default parameters, container AWC), scored on "
                "the rs-gated paired days shared with E3 calibrated, the transfers and the "
                "LS ensemble"
            ),
            "secondary_file": "uncalibrated_baseline_summary_flux_swim_mask.csv",
            "secondary_forward_run": (
                "derived_metrics.py run_uncalibrated_model, scored on flux + SWIM finite days "
                "(no benchmark gate); differs from the canonical row only by the day mask"
            ),
            "max_abs_persite_delta_between_masks": uncal_mask_delta,
        },
        "verify_all_sites": verify,
        "inputs_sha256": {str(f): sha256_file(f) for f in inputs if f.exists()},
        "problems": problems,
    }
    (out / "closure_pool_metadata.json").write_text(json.dumps(meta, indent=2))

    print(headline.loc[headline["tier"] == PRIMARY_TIER].to_string(index=False))
    print(f"\nwrote {out}")
    if problems:
        print("\nPROBLEMS:")
        for pr in problems:
            print(" -", pr)
        sys.exit(1)


if __name__ == "__main__":
    main()
